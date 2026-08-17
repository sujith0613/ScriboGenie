import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import logging

PI_HOME = os.path.dirname(os.path.abspath(__file__))
if os.name != 'nt':
    sys.path.insert(0, os.path.join(PI_HOME, "scribogenie_env", "Lib", "site-packages"))

import numpy as np
import cv2
import threading
import queue
import time
import json
import random
import hashlib
import tkinter as tk
from tkinter import Canvas, Frame, Button, Label
from PIL import Image, ImageDraw, ImageOps
from itertools import product
from spellchecker import SpellChecker
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed
import subprocess
import http.server
import socketserver

import confusions as confusion_lib
import recognizer
import recommend
import context_lm


try:
    import evdev
    from evdev import ecodes
except ImportError:
    evdev = None

LOG_DIR = os.path.join(PI_HOME, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DEBUG_CAPTURE = os.environ.get("SCRIBO_DEBUG") == "1"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "scribogenie.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ScriboGenie")


def _install_exception_hooks():
    """Capture every unhandled exception (main thread, background threads and
    tkinter UI callbacks) into the log so a crash is never silent."""
    import traceback as _tb

    def _log(tb):
        try:
            log.error("UNHANDLED EXCEPTION\n%s", "".join(_tb.format_list(tb)))
        except Exception:
            pass

    def _thread_hook(args):
        _log(args.exc_traceback)

    def _tk_hook(exc, val, tb):
        log.error("UNHANDLED Tkinter CALLBACK EXCEPTION: %s: %s",
                  exc.__name__, val)
        _log(tb)

    threading.excepthook = _thread_hook
    sys.excepthook = lambda t, v, tb: _log(tb)
    try:
        tk.Tk.report_callback_exception = _tk_hook
    except Exception:
        pass


if DEBUG_CAPTURE:
    DEBUG_DIR = os.path.join(LOG_DIR, "debug", f"run_{int(time.time())}")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    log.info("Debug capture ON -> %s", DEBUG_DIR)
else:
    DEBUG_DIR = None

MODEL_PATH = os.path.join(PI_HOME, "models", "recog", "model.onnx")
LOGICAL_W, LOGICAL_H = 800, 370
CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
EMNIST_CORRECTIONS = {'0': 'o', '8': 'r', '5': 's', '1': 'l', '2': 'z', '6': 'b', '9': 'g'}
# Full dyslexia confusion map: research pairs (confusions module) + digit/letter
# look-alikes from EMNIST. Used by dyslexia_aware_correction() to expand a
# recognized word into every plausible variant before SpellChecker filtering.
# Maps each letter to a STRING of all alternative letters (the correction loop
# iterates its characters), so a letter can have several look-alikes at once.
confusions: dict[str, str] = {}
for _a, _b in confusion_lib.all_pairs():
    confusions[_a] = confusions.get(_a, "") + _b
    confusions[_b] = confusions.get(_b, "") + _a
for _a, _b in [('1', 'l'), ('l', '1'), ('0', 'o'), ('o', '0'),
               ('8', 'r'), ('r', '8'), ('5', 's'), ('s', '5'),
               ('2', 'z'), ('z', '2'), ('6', 'b'), ('b', '6'),
               ('9', 'g'), ('g', '9')]:
    if _b not in confusions.get(_a, ""):
        confusions[_a] = confusions.get(_a, "") + _b
    if _a not in confusions.get(_b, ""):
        confusions[_b] = confusions.get(_b, "") + _a
spell = SpellChecker()

try:
    _SORTED = sorted(spell.word_frequency.dictionary.items(), key=lambda x: -x[1])
    _WORDS = [w.upper() for w, f in _SORTED if w.isalpha() and len(w) >= 3][:3000]
    LESSON_WORDS = {
        1: [w for w in _WORDS if len(w) == 3],
        2: [w for w in _WORDS if len(w) == 4],
        3: [w for w in _WORDS if len(w) == 5],
        4: [w for w in _WORDS if len(w) == 6],
        5: [w for w in _WORDS if len(w) == 7],
        6: [w for w in _WORDS if len(w) >= 8],
    }
    log.info("Loaded SpellChecker word bank: %d common words, %d–%d per level",
             len(_WORDS), min(len(v) for v in LESSON_WORDS.values()),
             max(len(v) for v in LESSON_WORDS.values()))
except Exception as e:
    log.warning("SpellChecker word bank failed (%s), using hardcoded word lists", e)
    LESSON_WORDS = {
        1: ["BAT", "CAT", "RAT", "HAT", "MAT", "PAT", "SAT", "BED", "RED", "PEN", "TEN", "PIG", "BIG", "DIG", "WIG"],
        2: ["BOOK", "LOOK", "TAKE", "CARE", "GAME", "NAME", "TIME", "MICE", "RING", "KING", "SING", "DUCK", "FISH", "SHIP"],
        3: ["HAPPY", "SUNNY", "MONEY", "HONEY", "FUNNY", "CANDY", "PARTY", "SILLY", "APPLE", "TIGER", "ROBOT"],
        4: ["BETTER", "LETTER", "SUMMER", "WINTER", "NUMBER", "BUTTER", "KITTEN", "MITTEN", "PENCIL", "PAPER"],
        5: ["SCIENCE", "READING", "WRITING", "DRAWING", "HOMEWORK", "PICTURE", "ANIMALS", "PLANETS"],
        6: ["BEAUTIFUL", "DIFFICULT", "WONDERFUL", "REMEMBER", "CHILDREN", "EDUCATION"]
    }

# Lowercased pool of known words used for next-word prediction and the
# "meaningful word" gate. Kept modest so one LM forward pass per candidate
# stays fast on the desktop.
_WORD_BANK = sorted({w.lower() for words in LESSON_WORDS.values() for w in words})

# Dyslexia-confusion correction: bound the number of letter substitutions tried
# per word. The naive full enumeration is 2^len(word); capping at 2 keeps it
# ~len^2 instead of exponential, so long words don't stall the prediction loop.
MAX_SUBSTITUTIONS = 2

# Full SpellChecker dictionary (the "is this a real word?" gate). Much larger
# than _WORD_BANK so common words like "wow" are accepted in sentence mode.
_KNOWN_WORDS = sorted(w.lower() for w, _ in _SORTED if w.isalpha())

# Common-word gate for the b/d and p/q mirror disambiguation: a mirror reading
# only gets the dictionary bonus if it is a word the child is expected to know
# (top-frequency bank). The full dictionary would also admit obscure words like
# "gob", making "gob" and "god" equally "real" and letting the weak char-LM
# (which happens to prefer "gob is great") win.
_WORD_BANK_SET = set(w.lower() for w in _WORD_BANK)

# When True the app runs in sentence mode: the child writes words freely, the
# recognized words are sent to the char-LM, which then suggests the most likely
# next word. If a written word is not meaningful, the child is asked to rewrite.
SENTENCE_MODE = True

mobile_clients = set()
mobile_loop = None
lesson_state = {"level": 1, "word": "BAT", "score": 0, "streak": 0}
_APP_INSTANCE = None

# ---------------------------------------------------------------------------
# Persistent progress store (score, streak, ratings, badges, active days).
# Stored as a JSON file so badges/awards/recognitions survive restarts and are
# consistent between the desktop and the mobile companion.
# ---------------------------------------------------------------------------
PROGRESS_FILE = os.path.join(PI_HOME, "data", "progress.json")

from progress_store import ProgressStore  # noqa: E402

_progress = ProgressStore(PROGRESS_FILE)


def save_progress():
    _progress.save()


def progress_state():
    return _progress.state()


def record_word_accepted():
    return _progress.record_word_accepted()


def record_word_rejected():
    _progress.record_word_rejected()


def record_rating(rating_id, word):
    return _progress.record_rating(rating_id, word)

def get_next_lesson(level=None):
    global lesson_state
    if level is None:
        level = lesson_state["level"]
    else:
        lesson_state["level"] = level
    level = min(max(level, 1), 6)
    word = random.choice(LESSON_WORDS[level])
    lesson_state["word"] = word
    lesson_state["level"] = level
    return word

def analyze_attempt(written, target):
    written = written.strip().upper()
    target = target.strip().upper()
    if written == target:
        lesson_state["score"] += 10
        lesson_state["streak"] += 1
        return "correct", f"+10 pts (streak: {lesson_state['streak']})"
    else:
        lesson_state["streak"] = 0
        return "wrong", f"Expected: {target}, Got: {written}"

async def ws_handler(websocket):
    global lesson_state
    mobile_clients.add(websocket)
    try:
        await websocket.send(json.dumps({
            "type": "sync",
            "state": progress_state(),
            "sentence": getattr(_APP_INSTANCE, "_last_full_sentence", "")
        }))
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type", "")
            if msg_type == "rate":
                rating = data.get("rating", "great")
                word = data.get("word", "")
                unlocked = record_rating(rating, word)
                await broadcast({"type": "sync", "state": progress_state()})
                for b in unlocked:
                    await broadcast({"type": "badge", "badge": b})
            elif msg_type in ("request_audio", "speak"):
                t = data.get("text") or data.get("word", "")
                if t and _tts_queue_global is not None:
                    _tts_queue_global.put(str(t))
            elif msg_type == "clear_pi":
                if _APP_INSTANCE:
                    _APP_INSTANCE.root.after(0, _APP_INSTANCE.clear)
    except Exception as e:
        log.warning(f"WebSocket handler error: {e}")
    finally:
        mobile_clients.discard(websocket)

async def serve_ws():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        await asyncio.Future()

def start_websocket_server():
    global mobile_loop
    mobile_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(mobile_loop)
    mobile_loop.run_until_complete(serve_ws())

def start_http_server():
    mobile_dir = os.path.join(PI_HOME, "mobile")
    if not os.path.isdir(mobile_dir):
        return
    os.chdir(mobile_dir)
    with socketserver.TCPServer(("0.0.0.0", 8000), http.server.SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()

async def broadcast(data):
    if not mobile_clients:
        return
    msg = json.dumps(data) if isinstance(data, dict) else data
    await asyncio.gather(*(s.send(msg) for s in mobile_clients.copy()), return_exceptions=True)

def send_to_mobile_sync(data):
    if not mobile_loop or not mobile_loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(broadcast(data), mobile_loop)


def dump_debug(name, payload, gray=None):
    """Persist a debug snapshot under logs/debug/run_<ts>/ when SCRIBO_DEBUG=1.

    When `gray` (HxW uint8 canvas) is given, the source image is also saved as
    a PNG beside the JSON so real-handwriting localization can be inspected.
    """
    if not DEBUG_DIR:
        return
    try:
        fname = os.path.join(DEBUG_DIR,
                             f"{int(time.time() * 1000)}_{name}.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info("debug dump: %s", os.path.relpath(fname, LOG_DIR))
        if gray is not None:
            pname = os.path.join(DEBUG_DIR,
                                 f"{int(time.time() * 1000)}_{name}.png")
            import cv2 as _cv2
            _cv2.imwrite(pname, gray)
            log.info("debug dump: %s", os.path.relpath(pname, LOG_DIR))
    except Exception as e:
        log.error("debug dump failed: %s", e)

def apply_emnist_context_correction(chars_str):
    if not chars_str or not any(c.isalpha() for c in chars_str): return chars_str
    return "".join([EMNIST_CORRECTIONS.get(c, c) for c in chars_str])

def dyslexia_aware_correction(word):
    if len(word) <= 1: return word
    # Fast path: already a dictionary word -> keep it. known() is a cheap dict
    # lookup, unlike correction() which runs an expensive edit-distance-2 search
    # over the whole dictionary.
    if spell.known([word]):
        return word
    # Bounded dyslexia-confusion search. The naive 2^len(word) enumeration
    # explodes for words over ~4-5 letters (product over every confusion +
    # spell.correction per variant), stalling the prediction loop. Instead try
    # only a bounded number of letter substitutions (k <= MAX_SUB) and return
    # the first variant the spellchecker accepts, so cost is ~len*2^MAX_SUB
    # worst case, not 2^len. Each variant is tested with known() (a dict
    # lookup) rather than correction() (an edit-distance-2 scan of the whole
    # dictionary, ~0.6s per call), which is what actually blew up the loop.
    pos = [i for i, ch in enumerate(word) if ch in confusions]
    import itertools as _it
    for k in range(1, min(len(pos), MAX_SUBSTITUTIONS) + 1):
        for combo in _it.combinations(pos, k):
            for subs in _it.product(*[confusions[word[i]] for i in combo]):
                cand = list(word)
                for i, ch in zip(combo, subs):
                    cand[i] = ch
                c = "".join(cand)
                if spell.known([c]):
                    return c
    return spell.correction(word) or word

class TTSSpeaker:
    TTS_MODEL = "KittenML/kitten-tts-micro-0.8"
    TTS_VOICE = "Bella"
    TTS_SPEED = 0.9
    def __init__(self):
        self.q = queue.Queue(maxsize=3)
        self._model = None
        self._model_err = None
        self._last_enqueue = 0.0
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            try:
                t = self.q.get()
                self._say(t)
            except Exception:
                log.warning("TTS thread error", exc_info=True)
    def _get_model(self):
        if self._model is not None or self._model_err is not None:
            return self._model
        try:
            from kittentts import KittenTTS
            self._model = KittenTTS(self.TTS_MODEL)
            log.info("Loaded TTS model %s", self.TTS_MODEL)
        except Exception as e:
            self._model_err = e
            log.warning("TTS engine unavailable (%s); skipping speech", e)
            return None
        return self._model
    def _say(self, text):
        if not text:
            return
        model = self._get_model()
        if model is not None:
            try:
                import sounddevice as sd
                audio = model.generate(text, voice=self.TTS_VOICE,
                                       speed=self.TTS_SPEED)
                sd.play(audio, 24000)
                sd.wait()
                return
            except Exception:
                log.warning("KittenTTS speech failed; falling back", exc_info=True)
        try:
            if os.name == 'nt':
                subprocess.run(["powershell", "-Command",
                    f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"],
                    stderr=subprocess.DEVNULL, timeout=15)
            else:
                subprocess.run(["espeak", "-s", "140", text],
                               stderr=subprocess.DEVNULL, timeout=15)
        except Exception:
            log.warning("TTS unavailable (espeak missing); skipping speech")
    def speak(self, text, throttle=0.0):
        """Queue `text` for the TTS thread without ever blocking the caller.

        `throttle` seconds: skip if a speak was enqueued more recently than
        that (used for clue speech so the child isn't talked at every stroke).
        When the queue is full the oldest pending item is dropped, so the UI
        thread can never stall waiting for TTS to finish playing.
        """
        if not text:
            return
        now = time.time()
        if throttle > 0 and (now - self._last_enqueue) < throttle:
            return
        self._last_enqueue = now
        try:
            self.q.put_nowait(str(text))
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(str(text))
            except (queue.Empty, queue.Full):
                pass

_tts_queue_global = None

class PredictorWorker(threading.Thread):
    def __init__(self, tq, rq, stop):
        super().__init__(daemon=True)
        self.tq, self.rq, self.stop = tq, rq, stop
        self.result_callback = None
    def run(self):
        recognizer._get_session()  # warm the ONNX recognizer session
        log.info("Loaded recognizer model from %s", MODEL_PATH)
        while not self.stop.is_set():
            try:
                task = self.tq.get(timeout=0.2)
                if not task:
                    break
                pil_img, cw, ch, ts, scale, gen, sentence_prefix = task
                img = np.array(pil_img)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                _T = {"prep": 0.0, "comp": 0.0, "segment": 0.0,
                      "crop": 0.0, "recognize": 0.0, "correct": 0.0,
                      "recommend": 0.0, "lm": 0.0}
                _t0 = time.time()
                _t = _t0

                # connected components are the cheap ground-truth of ink blobs:
                # they always find every blob, but merge touching letters. Use
                # them both as the fallback box source and as a sanity floor
                # for the YOLO box count.
                thr = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 15, 8)
                num, labs, stats, cents = cv2.connectedComponentsWithStats(thr)
                comp_boxes = [(int(stats[i][0]), int(stats[i][1]),
                               int(stats[i][0] + stats[i][2]),
                               int(stats[i][1] + stats[i][3]))
                              for i in range(1, num) if stats[i][4] >= 12]
                x_boxes = [(b[0], b[2]) for b in comp_boxes]
                y_boxes = [(b[1], b[3]) for b in comp_boxes]
                _T["comp"] = time.time() - _t; _t = time.time()

                # Segmentation is purely component-based (YOLO was removed):
                # connectedComponents always finds every ink blob, then
                # segment_components groups them into lines and merge_letters
                # fuses multi-stroke letters (dot+stem, two-stroke w/y, ...)
                # into one crop per real letter. The gap-based merge never
                # splits letters that physically touch (e.g. rn -> m).
                lines = recognizer.segment_components(x_boxes, y_boxes)
                letter_boxes = []
                line_letter_idxs = []
                for line in lines:
                    merged = recognizer.merge_letters(line, x_boxes, y_boxes)
                    start = len(letter_boxes)
                    letter_boxes.extend(merged)
                    line_letter_idxs.append(
                        list(range(start, len(letter_boxes))))

                crops = [gray[y0:y1, x0:x1] for (x0, y0, x1, y1) in letter_boxes]
                _T["segment"] = time.time() - _t; _t = time.time()
                per_letter = recognizer.recognize_crops(crops)
                recognized = [max(pos, key=pos.get) for pos in per_letter]
                _T["recognize"] = time.time() - _t; _t = time.time()

                # recommendation layer: group into words, combine recognizer
                # probs with the char-LM context, surface suggested spellings
                suggestions = []
                words_info = []
                next_words = []
                rec_sentence = ""
                full_sentence = ""
                # flat fallback kept for when word-grouping fails below.
                # Correct word-by-word (not the whole joined string): the
                # dyslexia-confusion search is ~2^len and explodes on a
                # multi-word string with spaces, stalling the prediction loop.
                raw = apply_emnist_context_correction("".join(recognized))
                corrected = " ".join(
                    dyslexia_aware_correction(w)
                    for w in raw.split()) if raw else raw
                try:
                    rec_all = recommend.recommend_all(
                        per_letter, letter_boxes, lines=line_letter_idxs,
                        known=_WORD_BANK_SET)
                    for w_ in rec_all["words"]:
                        if w_["best"] != w_["spelling"] and \
                           w_["margin"] >= recommend.SUGGEST_MARGIN:
                            suggestions.append({
                                "word": w_["best"],
                                "alternatives": w_["alternatives"],
                                "margin": w_["margin"],
                                "offsets": w_["offsets"],
                            })
                        words_info.append({
                            "spelling": w_["spelling"],
                            "best": w_["best"],
                            "margin": w_["margin"],
                            "offsets": w_["offsets"],
                            "indices": w_["indices"],
                        })
                    rec_sentence = rec_all["sentence"]
                    # raw/corrected reflect word spacing: join each word's
                    # greedy spelling with a space instead of flattening the
                    # whole string (which produced "iam" for "i am").
                    # `corrected` uses the CONTEXT-RESOLVED best spelling per
                    # word (recommend_all rebuilds sentence from `best`, so a
                    # b/d or p/q mirror confusion is already resolved there),
                    # with the dictionary correction as a last resort for words
                    # the recommendation layer left untouched.
                    spellings = [w_["spelling"] for w_ in rec_all["words"]]
                    bests = [w_["best"] for w_ in rec_all["words"]]
                    raw = apply_emnist_context_correction(" ".join(spellings))
                    corrected = " ".join(
                        b if b != s else dyslexia_aware_correction(s)
                        for s, b in zip(spellings, bests))
                    # next-word prediction: the current (last) word's recognized
                    # letters constrain the candidates; the words before it are
                    # the LM context. Remaining slots fill with context-based
                    # guesses so the list is never empty.
                    try:
                        full_sentence = (sentence_prefix + " " + rec_sentence).strip()
                        if rec_all["words"]:
                            current = rec_all["words"][-1]
                            prefix = current["spelling"]
                            context = " ".join(
                                w_["spelling"] for w_ in rec_all["words"][:-1])
                            next_words = recommend.predict_current_word(
                                context, prefix, _WORD_BANK, k=5,
                                full_sentence=full_sentence)
                        else:
                            next_words = recommend.next_word(
                                full_sentence, _WORD_BANK, k=5)
                    except Exception:
                        next_words = []
                        log.error("Next-word prediction failed", exc_info=True)
                    _T["lm"] = time.time() - _t; _t = time.time()
                    # sentence-completion gate: does the model assign high
                    # probability to a period/EOS? Strict, so it under-triggers.
                    complete = False
                    try:
                        causal = context_lm.get_causal()
                        if causal is not None and full_sentence:
                            complete = causal.sentence_is_complete(full_sentence)
                    except Exception:
                        log.error("Sentence-end scoring failed", exc_info=True)
                except Exception:
                    rec_sentence = ""
                    log.error("Recommendation failed", exc_info=True)

                dump_debug("prediction", {
                    "raw": raw,
                    "corrected": corrected,
                    "sentence": rec_sentence,
                    "words": words_info,
                    "suggestions": suggestions,
                    "next_words": next_words,
                    "complete": complete,
                    "per_letter": per_letter,
                    "x_boxes": x_boxes,
                    "y_boxes": y_boxes,
                    "lines": line_letter_idxs,
                    "letter_boxes": letter_boxes,
                    "timings_ms": {k: round(v * 1000) for k, v in _T.items()},
                    "wall_ms": round((time.time() - _t0) * 1000),
                }, gray=gray)
                self.rq.put((ts, {"raw": raw,
                                  "corrected": corrected,
                                  "gen": gen,
                                  "sentence": rec_sentence,
                                  "words": words_info,
                                  "suggestions": suggestions,
                                  "next_words": next_words,
                                  "complete": complete,
                                  "lines": line_letter_idxs}))
                if self.result_callback:
                    self.result_callback()
            except queue.Empty:
                continue
            except Exception:
                import traceback
                log.error(f"Prediction error:\n{traceback.format_exc()}")

class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ScriboGenie Pro Station")
        self.root.geometry("800x480")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0F0F0")
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self._focused = True
        self._last_draw_time = 0.0
        self._last_pred_timer = time.time()
        self._last_enqueued_sig = None
        self.root.bind("<FocusIn>", lambda e: setattr(self, '_focused', True))
        self.root.bind("<FocusOut>", lambda e: setattr(self, '_focused', False))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.root.bind("<Control-Delete>", lambda e: self.clear())

        self.status_bar = Label(root, text="[OK] System Ready", bd=1, relief="sunken",
                                anchor="w", fg="green")
        self.status_bar.pack(side="bottom", fill="x")

        if not SENTENCE_MODE:
            lesson_frame = Frame(root, bg="#E8F5E9")
            lesson_frame.pack(fill="x", padx=10, pady=(5, 0))
            self.lbl_lesson = Label(lesson_frame,
                text=f"Lesson: {lesson_state['word']} (Level {lesson_state['level']})",
                font=("Arial", 11, "bold"), bg="#E8F5E9", fg="#2E7D32")
            self.lbl_lesson.pack(side="left", padx=10)
            self.lbl_score = Label(lesson_frame,
                text=f"Score: {lesson_state['score']}",
                font=("Arial", 11), bg="#E8F5E9", fg="#1565C0")
            self.lbl_score.pack(side="right", padx=10)

        sentence_frame = Frame(root, bg="#FFF8E1")
        sentence_frame.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_sentence = Label(sentence_frame,
            text="Sentence: --",
            font=("Arial", 11, "bold"), bg="#FFF8E1", fg="#B26A00",
            anchor="w")
        self.lbl_sentence.pack(side="left", padx=10)
        self.lbl_next = Label(sentence_frame,
            text="Next word: --",
            font=("Arial", 11, "italic"), bg="#FFF8E1", fg="#6A1B9A",
            anchor="w")
        self.lbl_next.pack(side="left", padx=10)

        self.controls = Frame(root)
        self.controls.pack(side="bottom", fill="x", pady=5)
        Button(self.controls, text="Clear Canvas", command=self.clear,
               width=15).pack(side="left", padx=10)
        Button(self.controls, text="Speak Word", command=self.speak_word,
               width=15).pack(side="left", padx=10)
        Button(self.controls, text="Undo", command=self.undo,
               width=6).pack(side="left", padx=2)
        Button(self.controls, text="Redo", command=self.redo,
               width=6).pack(side="left", padx=2)
        Button(self.controls, text="Eraser", command=self.toggle_eraser,
               width=7).pack(side="left", padx=2)
        Button(self.controls, text="Next Word", command=self.next_word,
               width=12).pack(side="left", padx=10)

        self.result_panel = Frame(root, bg="#F3F0FF", height=50)
        self.result_panel.pack(side="bottom", fill="x", padx=10, pady=5)

        self.canvas_frame = Frame(root, bg="white", bd=2, relief="sunken")
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = Canvas(self.canvas_frame, bg="white",
                             width=LOGICAL_W, height=LOGICAL_H, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.lbl_cor = Label(self.result_panel, text="Corrected: --",
                             font=("Arial", 12, "bold"), bg="#F3F0FF", fg="#2E7D32")
        self.lbl_cor.pack(side="left", padx=20)
        self.lbl_raw = Label(self.result_panel, text="Raw: --",
                             font=("Arial", 12), bg="#F3F0FF", fg="#555555")
        self.lbl_raw.pack(side="left", padx=20)

        self.image = Image.new("RGB", (LOGICAL_W, LOGICAL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        global _APP_INSTANCE
        _APP_INSTANCE = self
        self.speaker = TTSSpeaker()
        global _tts_queue_global
        _tts_queue_global = self.speaker.q
        self.tq, self.rq = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.worker = PredictorWorker(self.tq, self.rq, self.stop)
        self.worker.start()
        self.worker.result_callback = lambda: self._wake_ui()
        self._word_gen = 0
        self._ui_poll_after_id = None
        self._ui_poll_interval = 50  # ms; main-thread rq poller
        self._start_ui_poller()
        self.sentence_words = []
        self._sentence_accepted = False
        self._last_full_sentence = ""
        self._predict_after_id = None
        self._stroke_groups = []
        self._undone_strokes = []
        self._eraser = False
        self._current_segments = []
        self._stroke_first_point = None
        self._stroke_color = "black"
        self._pen_width = 4
        self._eraser_size = 20
        self._wacom_found = False
        self._init_wacom()
        if not self._wacom_found:
            self.canvas.bind("<ButtonPress-1>", self._mouse_press)
            self.canvas.bind("<B1-Motion>", self._mouse_draw)
            self.canvas.bind("<ButtonRelease-1>", self._mouse_release)
        if SENTENCE_MODE:
            self.speaker.speak("Write any word to start your sentence.")
        else:
            self.speaker.speak(f"Write the word {lesson_state['word']}")

    def _mouse_press(self, e):
        self._last_draw_time = time.time()
        self._last_x, self._last_y = e.x, e.y

    def _mouse_draw(self, e):
        self._last_draw_time = time.time()
        color = "white" if self._eraser else "black"
        self._stroke_color = color
        w = self._eraser_size if self._eraser else self._pen_width
        if getattr(self, '_last_x', None) is not None:
            self._current_segments.append((self._last_x, self._last_y, e.x, e.y))
            self.canvas.create_line(self._last_x, self._last_y, e.x, e.y,
                                    width=w, fill=color, capstyle="round", smooth=True, tags="stroke")
            self.draw.line([self._last_x, self._last_y, e.x, e.y], fill=color, width=w)
        else:
            self._stroke_first_point = (e.x, e.y)
            r = w // 2
            self.canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, fill=color, tags="stroke")
            self.draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=color)
        self._last_x, self._last_y = e.x, e.y

    def _mouse_release(self, e):
        self._last_x = None
        self._finalize_stroke()
        self._schedule_prediction(0.2)

    def _init_wacom(self):
        if not evdev:
            return
        for p in evdev.list_devices():
            dev = evdev.InputDevice(p)
            if "Wacom" in dev.name or "CTL" in dev.name:
                self._wacom_found = True
                threading.Thread(target=self._wacom_loop, args=(dev,),
                                 daemon=True).start()

    def _wacom_loop(self, dev):
        rx, ry = 0, 0
        last_dispatch = 0.0
        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY and event.value == 1:
                if event.code == ecodes.BTN_0:
                    self.root.after(0, self.undo)
                elif event.code == ecodes.BTN_1:
                    self.root.after(0, self.redo)
                elif event.code == ecodes.BTN_2:
                    self.root.after(0, self.clear)
                elif event.code == ecodes.BTN_3:
                    self.root.after(0, self.speak_word)
                elif event.code == ecodes.BTN_STYLUS:
                    self.root.after(0, self.toggle_eraser)
                elif event.code == ecodes.BTN_STYLUS2:
                    self.root.after(0, self.speak_word)
            elif event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X:
                    rx = event.value
                elif event.code == ecodes.ABS_Y:
                    ry = event.value
                elif event.code == ecodes.ABS_PRESSURE:
                    now = time.monotonic()
                    if event.value > 100 and now - last_dispatch > 0.008:
                        last_dispatch = now
                        self.root.after(0, lambda rx=rx, ry=ry: self._wacom_draw_raw(rx, ry))
                    elif event.value <= 100:
                        self.root.after(0, lambda: self._wacom_pen_up())

    def _wacom_pen_up(self):
        self._wacom_last = None
        self._finalize_stroke()
        self._schedule_prediction(0.2)

    def _wacom_draw_raw(self, raw_x, raw_y):
        if not self._focused:
            return
        self._last_draw_time = time.time()
        color = "white" if self._eraser else "black"
        self._stroke_color = color
        w = self._eraser_size if self._eraser else self._pen_width
        cvs_x = self.canvas.winfo_rootx()
        cvs_y = self.canvas.winfo_rooty()
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        local_x = (raw_x / 15200) * scr_w - cvs_x
        local_y = (raw_y / 9500) * scr_h - cvs_y
        if getattr(self, '_wacom_last', None) is not None:
            lx, ly = self._wacom_last
            self._current_segments.append((lx, ly, local_x, local_y))
            self.canvas.create_line(lx, ly, local_x, local_y,
                                    width=w, fill=color, capstyle="round", smooth=True, tags="stroke")
            self.draw.line([lx, ly, local_x, local_y], fill=color, width=w)
        else:
            self._stroke_first_point = (local_x, local_y)
            r = w // 2
            self.canvas.create_oval(local_x-r, local_y-r, local_x+r, local_y+r,
                                    fill=color, tags="stroke")
            self.draw.ellipse([local_x-r, local_y-r, local_x+r, local_y+r], fill=color)
        self._wacom_last = (local_x, local_y)

    def _sentence_lines(self, latest) -> list[str]:
        """Group recognized words into display lines (one row per written
        line). Uses the worker's per-line letter indices and each word's letter
        indices; falls back to a single flat sentence.
        """
        lines = latest.get("lines") or []
        words = latest.get("words") or []
        if not lines or not words:
            return [latest.get("sentence", "")]
        line_of_word: list[int] = []
        for wi, w in enumerate(words):
            w_idxs = w.get("indices") or []
            found = -1
            for li, lidx in enumerate(lines):
                if lidx and any(i in lidx for i in w_idxs):
                    found = li
                    break
            line_of_word.append(found if found >= 0 else max(len(lines) - 1, 0))
        rows: list[list[str]] = [[] for _ in lines]
        for wi, w in enumerate(words):
            li = line_of_word[wi]
            if li < len(rows):
                rows[li].append(w.get("best") or w.get("spelling") or "")
        return [" ".join(r) for r in rows]

    def _render_sentence(self, latest):
        rows = self._sentence_lines(latest)
        shown = "\n".join(rows) if rows else latest.get("sentence", "")
        shown = shown.strip() or "--"
        self.lbl_sentence.config(text=f"Sentence: {shown}")
        next_words = latest.get("next_words", [])
        if next_words:
            nw_text = " | ".join(f"{w['word']}" for w in next_words)
        else:
            nw_text = "--"
        self.lbl_next.config(text=f"Next word: {nw_text}")

    def _handle_sentence_result(self, latest, raw_text, corrected_text):
        """Accept recognized words into the running sentence.

        The recognized words are already sent to the char-LM, which returns the
        next-word suggestions. Every recognized word is accepted so prediction
        continues for the next character rather than early-stopping on a single
        "meaningful" letter.
        """
        words_info = latest.get("words", [])
        sentence = latest.get("sentence", "")
        next_words = latest.get("next_words", [])
        running = list(self.sentence_words)

        if words_info:
            accepted = [w["best"] for w in words_info]
        else:
            # fall back to the whole corrected string when segmentation failed
            accepted = [t for t in corrected_text.lower().split() if t.isalpha()]

        # accept the words, add to the running sentence, persist
        for accepted_word in accepted:
            if accepted_word and (not running or
                                  running[-1].lower() != accepted_word.lower()):
                running.append(accepted_word)
        self.sentence_words = running
        self._last_full_sentence = " ".join(running)
        unlocked = record_word_accepted()

        # always keep suggesting the next word so prediction never stops early
        complete = False
        if next_words:
            top = next_words[0]["word"]
            self.speaker.speak(f"Good! Try {top}.", throttle=4.0)
            accept_msg = f"Good! Try: {top}"
        else:
            self.speaker.speak("Good!", throttle=4.0)
            accept_msg = "Good!"
        self.lbl_cor.config(text=f"Corrected: {corrected_text} — {accept_msg}",
                            fg="#2E7D32")
        send_to_mobile_sync({
            "type": "recognition",
            "raw": raw_text,
            "corrected": corrected_text,
            "correct": True,
            "complete": complete,
            "feedback": accept_msg,
            "sentence": " ".join(running).strip(),
            "lines": self._sentence_lines(latest),
            "suggestions": latest.get("suggestions", []),
            "next_words": next_words,
            "state": progress_state(),
        })
        for b in unlocked:
            send_to_mobile_sync({"type": "badge", "badge": b})

    def _current_raw_display(self):
        return getattr(self, '_raw_display', "Raw: --")

    def _set_raw_display(self, raw_text, sentence):
        self._raw_display = f"Raw: {raw_text}  |  Recognized: {sentence}"
        self.lbl_raw.config(text=self._raw_display)

    def _wake_ui(self):
        """Called from the worker thread whenever a result lands on rq.

        The old code did root.after(0, self._check_results) here, but calling
        after() from a non-main thread is unreliable in tkinter (callbacks get
        lost/coalesced), so the UI only repainted at word boundaries. Instead we
        just touch a flag; the always-running main-thread poller below picks it
        up immediately and drains rq.
        """
        self._ui_dirty = True

    def _start_ui_poller(self):
        """Continuously drain rq on the MAIN thread so the sentence/prediction
        UI repaints as results arrive, not only when a stroke is released.

        Runs on a short fixed timer regardless of worker activity. Whenever the
        worker signals dirty, we schedule an immediate drain; otherwise we keep
        the steady cadence so nothing is ever missed.
        """
        self._ui_dirty = True

        def _poll():
            if self.stop.is_set():
                return
            if self._ui_dirty or not self.rq.empty():
                self._ui_dirty = False
                try:
                    self._check_results()
                except Exception:
                    import traceback
                    log.error("UI poller error:\n%s", traceback.format_exc())
            self._ui_poll_after_id = self.root.after(
                self._ui_poll_interval, _poll)

        self.root.after(0, _poll)

    def _check_results(self):
        log.debug("CHECK_RESULTS fired, rq size=%d _word_gen=%d",
                  self.rq.qsize(), getattr(self, '_word_gen', -1))
        latest = None
        while not self.rq.empty():
            _, res = self.rq.get()
            latest = res
        if latest:
            gen = latest.get("gen", -1)
            if gen != self._word_gen:
                log.debug("Stale prediction (gen %d != %d), skipping", gen, self._word_gen)
            else:
                raw_text = latest.get("raw", "")
                corrected_text = latest.get("corrected", "")
                if corrected_text == getattr(self, '_last_result', ''):
                    log.debug("Skipping duplicate prediction result")
                else:
                    self._last_result = corrected_text
                    self.lbl_cor.config(text=f"Corrected: {corrected_text}")
                    self._set_raw_display(raw_text, latest.get("sentence", ""))
                    log.debug("UI update: raw=%r cor=%r", raw_text, corrected_text)

                    # sentence mode: render the recognized sentence and the
                    # char-LM's next-word suggestions for the child
                    self._render_sentence(latest)

                    if SENTENCE_MODE:
                        self._handle_sentence_result(latest, raw_text, corrected_text)
                        self._requeue_prediction()
                        return

                    target = lesson_state["word"].upper()
                    corrected = corrected_text.strip().upper()

                    wrong_chars = []
                    min_len = min(len(corrected), len(target))
                    for i in range(min_len):
                        if corrected[i] != target[i]:
                            wrong_chars.append({"position": i+1, "expected": target[i], "got": corrected[i]})
                    if len(corrected) < len(target):
                        for i in range(len(corrected), len(target)):
                            wrong_chars.append({"position": i+1, "expected": target[i], "got": "(missing)"})
                    if len(corrected) > len(target):
                        for i in range(len(target), len(corrected)):
                            wrong_chars.append({"position": i+1, "expected": "(end)", "got": corrected[i]})

                    if corrected == target:
                        lesson_state["score"] += 10
                        lesson_state["streak"] += 1
                        if lesson_state["streak"] >= 3 and lesson_state["level"] < 6:
                            lesson_state["level"] += 1
                            lesson_state["streak"] = 0
                            send_to_mobile_sync({
                                "type": "level_up", "new_level": lesson_state["level"]
                            })
                        self.lbl_score.config(text=f"Score: {lesson_state['score']}")
                        self.speaker.speak("Correct! Well done!")
                        send_to_mobile_sync({
                            "type": "recognition",
                            "raw": raw_text,
                            "corrected": corrected_text,
                            "correct": True,
                            "score": lesson_state["score"],
                            "streak": lesson_state["streak"],
                            "sentence": latest.get("sentence", ""),
                            "suggestions": latest.get("suggestions", [])
                        })
                        self.root.after(1500, self.next_word)
                        return
                    else:
                        lesson_state["streak"] = 0
                        feedback_parts = []
                        for wc in wrong_chars[:3]:
                            feedback_parts.append(f"Character {wc['position']} should be {wc['expected']}, not {wc['got']}")
                        feedback = ". ".join(feedback_parts)
                        self.lbl_cor.config(text=f"Corrected: {corrected_text} — {feedback}", fg="#D32F2F")
                        if wrong_chars:
                            self.speaker.speak(feedback)
                        send_to_mobile_sync({
                            "type": "recognition",
                            "raw": raw_text,
                            "corrected": corrected_text,
                            "correct": False,
                            "wrong_chars": wrong_chars,
                            "feedback": feedback,
                            "sentence": latest.get("sentence", ""),
                            "suggestions": latest.get("suggestions", [])
                        })

        self._requeue_prediction()

    def _requeue_prediction(self):
        now = time.time()
        if not self._stroke_groups:
            return
        # Wait for the pen to settle (>=0.2s idle) so we predict a stable image.
        # Reschedule instead of returning when throttled: if we returned here,
        # nothing would wake the loop again until the next pen-up, which is why
        # erase/rewrite edits sometimes never re-predicted. Cancel the previous
        # timer first so repeated callers don't stack duplicate after()s.
        if now - self._last_draw_time < 0.2:
            remaining = max(0.0, 0.2 - (now - self._last_draw_time))
            self._schedule_prediction(remaining + 0.001)
            return
        if now - self._last_pred_timer < 0.2:
            remaining = max(0.0, 0.2 - (now - self._last_pred_timer))
            self._schedule_prediction(remaining + 0.001)
            return
        if self.tq.qsize() >= 2:
            self._schedule_prediction(0.05)
            return
        # Don't re-enqueue a prediction for an unchanged canvas: once the pen is
        # idle the image is stable, so returning here stops the loop (the next
        # pen-up re-enters it via _schedule_prediction). Without this guard the
        # reschedule above would re-run the worker on the same image forever.
        sig = hashlib.md5(self.image.tobytes()).hexdigest()
        if sig == getattr(self, '_last_enqueued_sig', None):
            return
        self._last_enqueued_sig = sig
        self._last_pred_timer = now
        prefix = " ".join(self.sentence_words)
        self.tq.put((self.image.copy(), 0, 0, now, 1.0, self._word_gen, prefix))
        log.debug(f"Prediction queued (queue size: {self.tq.qsize()})")

    def _schedule_prediction(self, delay=1.0):
        try:
            self.root.after_cancel(self._predict_after_id)
        except: pass
        self._predict_after_id = self.root.after(int(delay * 1000), self._check_results)

    def clear(self):
        self.canvas.delete("stroke")
        self.image = Image.new("RGB", (LOGICAL_W, LOGICAL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        self._last_x = None
        self._last_draw_time = time.time()
        while not self.rq.empty():
            try: self.rq.get_nowait()
            except queue.Empty: break
        while not self.tq.empty():
            try: self.tq.get_nowait()
            except queue.Empty: break
        self._last_pred_timer = 0
        self._last_enqueued_sig = None
        self._word_gen += 1
        self._last_result = ""
        self.sentence_words = []
        self._last_full_sentence = ""
        self.lbl_sentence.config(text="Sentence: --")
        self._stroke_groups.clear()
        self._undone_strokes.clear()
        self._current_segments = []
        self._stroke_first_point = None
        self.lbl_cor.config(text="Corrected: --")
        self.lbl_raw.config(text="Raw: --")
        self.lbl_next.config(text="Next word: --")

    def speak_word(self):
        if SENTENCE_MODE:
            text = self._last_full_sentence or getattr(self, '_last_result', "") or "Write any word to start your sentence."
            self.speaker.speak(text)
        else:
            self.speaker.speak(lesson_state["word"])

    def _finalize_stroke(self):
        if not self._current_segments and not self._stroke_first_point:
            return
        w = self._eraser_size if self._eraser else self._pen_width
        self._stroke_groups.append({
            "segments": self._current_segments[:],
            "first_point": self._stroke_first_point,
            "color": self._stroke_color,
            "width": w,
        })
        self._current_segments = []
        self._stroke_first_point = None
        self._undone_strokes.clear()

    def undo(self):
        if not self._stroke_groups:
            return
        group = self._stroke_groups.pop()
        self._undone_strokes.append(group)
        self._word_gen += 1
        self._last_result = ""
        self._rebuild_from_strokes()
        self._schedule_prediction(0.15)
        log.debug("Undo: %d strokes remaining", len(self._stroke_groups))

    def redo(self):
        if not self._undone_strokes:
            self.speaker.speak("Nothing to redo")
            return
        group = self._undone_strokes.pop()
        self._stroke_groups.append(group)
        self._word_gen += 1
        self._last_result = ""
        self._rebuild_from_strokes()
        self._schedule_prediction(0.15)
        log.debug("Redo")

    def _rebuild_from_strokes(self):
        self.canvas.delete("stroke")
        self.image = Image.new("RGB", (LOGICAL_W, LOGICAL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        for group in self._stroke_groups:
            color = group["color"]
            w = group.get("width", 4)
            if group["first_point"]:
                x, y = group["first_point"]
                r = w // 2
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, tags="stroke")
                self.draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
            for seg in group["segments"]:
                x1, y1, x2, y2 = seg
                self.canvas.create_line(x1, y1, x2, y2,
                                        width=w, fill=color, capstyle="round", smooth=True, tags="stroke")
                self.draw.line(seg, fill=color, width=w)

    def toggle_eraser(self):
        self._eraser = not self._eraser
        status = f"ERASER ON (size={self._eraser_size})" if self._eraser else "ERASER OFF"
        self.status_bar.config(text=f"[{status}]", fg="red" if self._eraser else "green")
        log.info("Eraser toggled %s", status)

    def next_word(self):
        if SENTENCE_MODE:
            self.sentence_words = []
            self._last_full_sentence = ""
            self.lbl_sentence.config(text="Sentence: --")
            self.lbl_next.config(text="Next word: --")
            self._last_result = ""
            self.clear()
            self.speaker.speak("Write a word to start your sentence.")
            send_to_mobile_sync({
                "type": "clear_sentence",
                "state": progress_state(),
            })
            return
        word = get_next_lesson(lesson_state["level"])
        self.lbl_lesson.config(text=f"Lesson: {word} (Level {lesson_state['level']})")
        self.lbl_score.config(text=f"Score: {lesson_state['score']}")
        while not self.rq.empty():
            try: self.rq.get_nowait()
            except queue.Empty: break
        while not self.tq.empty():
            try: self.tq.get_nowait()
            except queue.Empty: break
        self._last_pred_timer = 0
        self._stroke_groups.clear()
        self._undone_strokes.clear()
        self._current_segments = []
        self._stroke_first_point = None
        self.lbl_cor.config(text="Corrected: --")
        self.clear()
        self.speaker.speak(f"Write the word {word}")
        send_to_mobile_sync({
            "type": "lesson",
            "word": word,
            "level": lesson_state["level"],
            "mode": "copy",
            "score": lesson_state["score"]
        })

if __name__ == "__main__":
    if not SENTENCE_MODE:
        word = get_next_lesson(1)
        print(f"[ScriboGenie] Lesson: {word}")

    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    time.sleep(0.3)

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    print(f"[ScriboGenie] WebSocket: ws://0.0.0.0:8765")
    print(f"[ScriboGenie] Mobile PWA: http://192.168.4.1:8000/")

    def _preload_lm_sessions():
        try:
            context_lm._get_session()
            log.info("Preloaded char-LM session")
        except Exception as e:
            log.warning("char-LM preload failed: %s", e)
        try:
            context_lm.get_causal()
            log.info("Preloaded causal-LM session")
        except Exception as e:
            log.warning("causal-LM preload failed: %s", e)

    threading.Thread(target=_preload_lm_sessions, daemon=True).start()

    _install_exception_hooks()
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
