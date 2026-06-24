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
import tensorflow as tf

try:
    import evdev
    from evdev import ecodes
except ImportError:
    evdev = None

LOG_DIR = os.path.join(PI_HOME, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "scribogenie.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ScriboGenie")

MODEL_PATH = os.path.join(PI_HOME, "models", "myCnn.h5")
LOGICAL_W, LOGICAL_H = 800, 370
CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
EMNIST_CORRECTIONS = {'0': 'o', '8': 'r', '5': 's', '1': 'l', '2': 'z', '6': 'b', '9': 'g'}
confusions = {'b':'d','d':'b','p':'q','q':'p','i':'l','l':'i','1':'l','l':'1','0':'o','o':'0','r':'8','8':'r'}
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

mobile_clients = set()
mobile_loop = None
lesson_state = {"level": 1, "word": "BAT", "score": 0, "streak": 0}
_APP_INSTANCE = None

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
            "type": "lesson", "word": lesson_state["word"],
            "level": lesson_state["level"], "mode": "copy",
            "score": lesson_state["score"]
        }))
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type", "")
            if msg_type == "attempt":
                written = data.get("written", "")
                target = lesson_state["word"]
                result, msg = analyze_attempt(written, target)
                if result == "correct":
                    stars = min(lesson_state["streak"], 3)
                    await broadcast({
                        "type": "reward", "stars": stars, "message": msg,
                        "score": lesson_state["score"], "level": lesson_state["level"]
                    })
                else:
                    await broadcast({
                        "type": "wrong_attempt", "feedback_text": msg,
                        "expected": target, "got": written
                    })
            elif msg_type == "get_lesson":
                level = data.get("level", lesson_state["level"])
                word = get_next_lesson(level)
                await websocket.send(json.dumps({
                    "type": "lesson", "word": word, "level": level,
                    "mode": "copy", "score": lesson_state["score"]
                }))
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

def apply_emnist_context_correction(chars_str):
    if not chars_str or not any(c.isalpha() for c in chars_str): return chars_str
    return "".join([EMNIST_CORRECTIONS.get(c, c) for c in chars_str])

def dyslexia_aware_correction(word):
    if len(word) <= 1: return word
    variants = [''.join(p) for p in product(*[[c, confusions[c]] if c in confusions else [c] for c in word])]
    valid = [w for w in variants if spell.correction(w) == w]
    return valid[0] if valid else (spell.correction(word) or word)

class TTSSpeaker:
    def __init__(self):
        self.q = queue.Queue(maxsize=3)
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            t = self.q.get()
            self._say(t)
    def _say(self, text):
        if os.name == 'nt':
            subprocess.run(["powershell", "-Command",
                f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"],
                stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["espeak", "-s", "140", text], stderr=subprocess.DEVNULL)
    def speak(self, text):
        if text:
            self.q.put(str(text))

_tts_queue_global = None

class PredictorWorker(threading.Thread):
    def __init__(self, tq, rq, stop):
        super().__init__(daemon=True)
        self.tq, self.rq, self.stop = tq, rq, stop
        self.result_callback = None
    def run(self):
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))

        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2d')(inputs)
        x = tf.keras.layers.BatchNormalization(name='batch_normalization')(x)
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2d_1')(x)
        x = tf.keras.layers.BatchNormalization(name='batch_normalization_1')(x)
        shortcut = tf.keras.layers.Conv2D(64, 1, padding='same', name='conv2d_2')(inputs)
        x = tf.keras.layers.Add(name='add')([x, shortcut])
        x = tf.keras.layers.MaxPooling2D(2, name='max_pooling2d')(x)
        x = tf.keras.layers.Dropout(0.25, name='dropout')(x)

        x_res = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2d_3')(x)
        x_res = tf.keras.layers.BatchNormalization(name='batch_normalization_2')(x_res)
        x_res = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2d_4')(x_res)
        x_res = tf.keras.layers.BatchNormalization(name='batch_normalization_3')(x_res)
        shortcut_1 = tf.keras.layers.Conv2D(128, 1, padding='same', name='conv2d_5')(x)
        x = tf.keras.layers.Add(name='add_1')([x_res, shortcut_1])
        x = tf.keras.layers.MaxPooling2D(2, name='max_pooling2d_1')(x)
        x = tf.keras.layers.Dropout(0.25, name='dropout_1')(x)

        x_res = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv2d_6')(x)
        x_res = tf.keras.layers.BatchNormalization(name='batch_normalization_4')(x_res)
        x_res = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv2d_7')(x_res)
        x_res = tf.keras.layers.BatchNormalization(name='batch_normalization_5')(x_res)
        shortcut_2 = tf.keras.layers.Conv2D(256, 1, padding='same', name='conv2d_8')(x)
        x = tf.keras.layers.Add(name='add_2')([x_res, shortcut_2])
        x = tf.keras.layers.MaxPooling2D(2, name='max_pooling2d_2')(x)
        x = tf.keras.layers.Dropout(0.25, name='dropout_2')(x)

        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='dense')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout_3')(x)
        outputs = tf.keras.layers.Dense(62, activation='softmax', name='dense_1')(x)

        model = tf.keras.Model(inputs, outputs)
        model.load_weights(MODEL_PATH, by_name=True)
        log.info("Loaded model weights from %s", MODEL_PATH)
        while not self.stop.is_set():
            try:
                task = self.tq.get(timeout=0.2)
                if not task:
                    break
                pil_img, cw, ch, ts, scale, gen = task
                img = np.array(pil_img)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 8)
                num, labs, stats, cents = cv2.connectedComponentsWithStats(thr)
                indices = [i for i in range(1, num) if stats[i][4] >= 12]
                indices.sort(key=lambda i: stats[i][0])
                arrays = []
                for i in indices:
                    x, y, w, h = stats[i][:4]
                    crop = gray[y:y+h, x:x+w]
                    resized = cv2.resize(crop, (20, 20)).astype(np.float32)
                    arr = np.zeros((28, 28), dtype=np.float32)
                    arr[4:24, 4:24] = (255.0 - resized) / 255.0
                    arrays.append(arr.reshape(1, 28, 28, 1))
                recognized = []
                if arrays:
                    batch = np.vstack(arrays)
                    preds = model.predict_on_batch(batch)
                    for p in preds:
                        recognized.append(CHAR_LIST[np.argmax(p)])
                raw = apply_emnist_context_correction("".join(recognized))
                self.rq.put((ts, {"raw": raw, "corrected": dyslexia_aware_correction(raw), "gen": gen}))
                if self.result_callback:
                    self.result_callback()
            except queue.Empty:
                continue
            except Exception:
                import traceback
                log.error(f"Prediction error:\n{traceback.format_exc()}")

def draw_overlay(canvas, raw_text):
    canvas.delete("overlay")
    canvas.create_text(10, 10, text=f"Recognized: {raw_text}", anchor="nw",
                       font=("Arial", 14, "bold"), fill="green", tags="overlay")

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
        self.root.bind("<FocusIn>", lambda e: setattr(self, '_focused', True))
        self.root.bind("<FocusOut>", lambda e: setattr(self, '_focused', False))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.root.bind("<Control-Delete>", lambda e: self.clear())

        self.status_bar = Label(root, text="[OK] System Ready", bd=1, relief="sunken",
                                anchor="w", fg="green")
        self.status_bar.pack(side="bottom", fill="x")

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
        self.lbl_raw = Label(self.result_panel, text="Raw: --",
                             font=("Arial", 12), bg="#F3F0FF")
        self.lbl_raw.pack(side="left", padx=20)
        self.lbl_cor = Label(self.result_panel, text="Corrected: --",
                             font=("Arial", 12, "bold"), bg="#F3F0FF", fg="#2E7D32")
        self.lbl_cor.pack(side="left", padx=20)

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
        self.worker.result_callback = lambda: self.root.after(0, self._check_results)
        self._word_gen = 0
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
            self.canvas.bind("<B1-Motion>", self._mouse_draw)
            self.canvas.bind("<ButtonRelease-1>", self._mouse_release)
        self.speaker.speak(f"Write the word {lesson_state['word']}")

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
        self._schedule_prediction(2.0)

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
        self._schedule_prediction(2.0)

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

    def _check_results(self):
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
                    self.lbl_raw.config(text=f"Raw: {raw_text}")
                    self.lbl_cor.config(text=f"Corrected: {corrected_text}")
                    draw_overlay(self.canvas, raw_text)

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
                            "streak": lesson_state["streak"]
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
                            "feedback": feedback
                        })

        now = time.time()
        if now - self._last_draw_time < 1.0:
            return
        if self._stroke_groups and (not hasattr(self, '_last_pred_timer') or now - self._last_pred_timer >= 2.0) and self.tq.qsize() < 2:
            self._last_pred_timer = now
            self.tq.put((self.image.copy(), 0, 0, now, 1.0, self._word_gen))
            log.debug(f"Prediction queued (queue size: {self.tq.qsize()})")

    def _schedule_prediction(self, delay=2.0):
        try:
            self.root.after_cancel(self._predict_after_id)
        except: pass
        self._predict_after_id = self.root.after(int(delay * 1000), self._check_results)

    def clear(self):
        self.canvas.delete("overlay")
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
        self._word_gen += 1
        self._stroke_groups.clear()
        self._undone_strokes.clear()
        self._current_segments = []
        self._stroke_first_point = None
        self.lbl_raw.config(text="Raw: --")
        self.lbl_cor.config(text="Corrected: --")

    def speak_word(self):
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
        self._rebuild_from_strokes()
        log.debug("Undo: %d strokes remaining", len(self._stroke_groups))

    def redo(self):
        if not self._undone_strokes:
            self.speaker.speak("Nothing to redo")
            return
        group = self._undone_strokes.pop()
        self._stroke_groups.append(group)
        self._rebuild_from_strokes()
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
        self.lbl_raw.config(text="Raw: --")
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
    word = get_next_lesson(1)
    print(f"[ScriboGenie] Lesson: {word}")

    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    time.sleep(0.3)

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    print(f"[ScriboGenie] WebSocket: ws://0.0.0.0:8765")
    print(f"[ScriboGenie] Mobile PWA: http://192.168.4.1:8000/")

    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
