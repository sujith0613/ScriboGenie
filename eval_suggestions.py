"""End-to-end recommendation eval: recognizer x char-LM combined accuracy.

For each eval case {sentence, pos, correct, alt}:
  1. Render the sentence onto a 800x370 canvas using EMNIST glyphs (val split)
     placed like the app's handwriting (words separated by a gap).
  2. Run the full ScriboGenie pipeline: adaptive threshold -> connected
     components (area>=12, x-sorted) -> per-component top-K recognizer probs
     -> word grouping -> char-LM combined scoring.
  3. Measure whether the COMBINED score prefers `correct` over `alt` at the
     confusion position (P_rec * P_lm^alpha), i.e. the recommendation would
     suggest the right letter.

Reports per-pair accuracy, the macro gate (>=85%), overall, and suggestion
precision/recall. Contrast with eval_lm.py (LM-only) -- this measures the
whole recognizer+LM path on rendered handwriting.
"""

import json
import os
import random

import numpy as np

import recognizer as rec
import recommend
import context_lm

_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(_ROOT, "data", "corpus", "eval_cases.json")
GLYPH = "/home/sujith/projects/pi-learn-station/data/processed/val62_uint8.npz"

GATE = 0.85
LOGICAL_W, LOGICAL_H = 800, 370
PEN = 5                    # render pen thickness (dilate kernel)
SCALE = 2                  # 28px glyph -> 56px on canvas
WORD_GAP = 26              # extra x gap between words (px)
CHAR_GAP = 4               # intra-word gap (px)
MIN_AREA = 12
TOP_K = 5

PAIRS = [("b", "d"), ("b", "p"), ("d", "q"), ("d", "t"),
         ("f", "v"), ("g", "k"), ("p", "q"), ("s", "z")]

_char2id = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_glyph_pool: dict[str, list[np.ndarray]] | None = None


def load_glyph_pool():
    """group val glyphs by their target lowercase letter (upright, ink=255)."""
    global _glyph_pool
    if _glyph_pool is not None:
        return _glyph_pool
    d = np.load(GLYPH)
    X, y = d["X"], d["y"]
    pool: dict[str, list[np.ndarray]] = {c: [] for c in "abcdefghijklmnopqrstuvwxyz"}
    for img, lab in zip(X, y):
        ch = rec.char_from_index(int(lab))
        if ch in pool:
            pool[ch].append(img)
    # drop letters with no samples
    pool = {k: v for k, v in pool.items() if v}
    _glyph_pool = pool
    return pool


def render_sentence(sentence: str, rng: random.Random) -> np.ndarray:
    """Render a sentence onto a white canvas using random upright glyphs.

    Scale is chosen so the whole sentence fits on the line (letters shrink for
    long sentences, as a child would write smaller). Returns a gray uint8
    canvas (ink dark ~0, paper light ~255).
    """
    pool = load_glyph_pool()
    canvas = np.full((LOGICAL_H, LOGICAL_W), 255, np.uint8)
    import cv2
    nchars = sum(1 for ch in sentence if ch != " ")
    nwords = sentence.count(" ") + 1
    # fit: nchars*28*s + (nchars-1)*CHAR_GAP + nwords*WORD_GAP <= LOGICAL_W-40
    avail = LOGICAL_W - 40 - nwords * WORD_GAP - (max(nchars - 1, 0)) * CHAR_GAP
    scale = max(0.9, min(SCALE, avail / (max(nchars, 1) * 28)))
    y0 = (LOGICAL_H - int(28 * scale)) // 2
    x = 20
    for ch in sentence:
        if ch == " ":
            x += WORD_GAP
            continue
        if ch not in pool:
            x += CHAR_GAP
            continue
        img = rng.choice(pool[ch])            # upright, ink=255 (28x28)
        big = cv2.resize(img, (int(28 * scale), int(28 * scale)),
                         interpolation=cv2.INTER_NEAREST)
        if x + big.shape[1] >= LOGICAL_W - 10:
            break
        # big has ink=255 (letter), bg=0; thicken ink, then put dark on white
        ink = (big > 128).astype(np.uint8) * 255
        pen = max(3, int(PEN * scale / SCALE))  # scale pen with letters
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (pen, pen)))
        paper = 255 - ink
        canvas[y0:y0 + big.shape[0], x:x + big.shape[1]] = paper
        x += big.shape[1] + CHAR_GAP
    return canvas


def pipeline(gray: np.ndarray):
    """Run the full segmentation + recognition + recommendation pipeline.

    Returns dict with per-component top-K, x_boxes, and recommend.recommend_all
    output.
    """
    import cv2
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 8)
    num, _l, stats, _c = cv2.connectedComponentsWithStats(thr)
    indices = [i for i in range(1, num) if stats[i][4] >= MIN_AREA]
    indices.sort(key=lambda i: stats[i][0])
    crops = []
    x_boxes = []
    for i in indices:
        x, y, w, h = stats[i][:4]
        crops.append(gray[y:y + h, x:x + w].astype(np.float32))
        x_boxes.append((int(x), int(x + w)))
    per_letter = rec.recognize_crops(crops, TOP_K)
    rec_all = recommend.recommend_all(per_letter, x_boxes, top_k=TOP_K)
    return {"per_letter": per_letter, "x_boxes": x_boxes, "rec": rec_all}


def word_containing(offsets, char_pos):
    for wi, (ws, we) in enumerate(offsets):
        if ws <= char_pos < we:
            return wi, char_pos - ws
    return None


def _orig_word_and_ci(sentence, pos):
    """Word ordinal + intra-word char index for `pos` in the ORIGINAL
    sentence (word-ordered, so we can align to the rendered pipeline)."""
    wi = 0
    cur = 0
    for word in sentence.split(" "):
        if cur <= pos < cur + len(word):
            return wi, pos - cur
        cur += len(word) + 1
        wi += 1
    return None, None


def evaluate(alpha=recommend.ALPHA, margin=recommend.SUGGEST_MARGIN,
             n=None, seed=0, balanced=True):
    cases = json.load(open(EVAL))
    if n:
        if balanced:
            by_pair: dict[tuple, list] = {}
            for c in cases:
                k = (c["correct"], c["alt"]) if (c["correct"], c["alt"]) in PAIRS \
                    else (c["alt"], c["correct"])
                by_pair.setdefault(k, []).append(c)
            per_pair = max(1, n // len(PAIRS))
            chosen = []
            for p in PAIRS:
                chosen += by_pair.get(p, [])[:per_pair]
            rng = random.Random(seed)
            rng.shuffle(chosen)
            cases = chosen
        else:
            cases = cases[:n]
    rng = random.Random(seed)
    stats = {p: [0, 0] for p in PAIRS}
    surfaced = 0
    surfaced_right = 0
    skipped = 0

    for c in cases:
        sent = c["sentence"]
        pos, correct, alt = c["pos"], c["correct"], c["alt"]
        owi, ci = _orig_word_and_ci(sent, pos)
        if owi is None:
            skipped += 1
            continue
        gray = render_sentence(sent, rng)
        try:
            pipe = pipeline(gray)
        except Exception:
            skipped += 1
            continue
        rec_all = pipe["rec"]
        words = rec_all["words"]
        if owi >= len(words):
            skipped += 1
            continue
        wi = owi
        ws, we = words[wi]["offsets"]
        word_probs = [pipe["per_letter"][i] for i in words[wi]["indices"]]
        if not word_probs or ci >= len(word_probs):
            skipped += 1
            continue

        # candidate spellings differing only at the confusion offset
        greedy = "".join(max(p, key=p.get) for p in word_probs)
        if ci >= len(greedy):
            skipped += 1
            continue
        spell_c = greedy[:ci] + correct + greedy[ci + 1:]
        spell_a = greedy[:ci] + alt + greedy[ci + 1:]
        sc_c = recommend.combine_log_score(
            spell_c, rec_all["sentence"], ws, we, word_probs, alpha)
        sc_a = recommend.combine_log_score(
            spell_a, rec_all["sentence"], ws, we, word_probs, alpha)

        key = (correct, alt) if (correct, alt) in PAIRS else (alt, correct)
        if key not in stats:
            skipped += 1
            continue
        stats[key][1] += 1
        if sc_c > sc_a:
            stats[key][0] += 1

        # suggestion surfacing: did we surface a corrected spelling, and was it right?
        best = max((spell_c, sc_c), (spell_a, sc_a), (greedy,
                   recommend.combine_log_score(greedy, rec_all["sentence"],
                                               ws, we, word_probs, alpha)),
                   key=lambda t: t[1])[0]
        if best != greedy:
            surfaced += 1
            if best == spell_c:
                surfaced_right += 1

    _report(stats, surfaced, surfaced_right, skipped)


def _report(stats, surfaced, surfaced_right, skipped):
    correct_total = 0
    total = 0
    macro = []
    print(f"{'pair':<6}{'tok_ok':>6}{'acc':>8}")
    for p in PAIRS:
        ok, tot = stats[p]
        acc = ok / tot if tot else 0
        macro.append(acc)
        correct_total += ok
        total += tot
        print(f"{p[0]+'/'+p[1]:<6}{tot:>6}{acc:>8.1%}")
    macro_mean = sum(macro) / len(macro)
    overall = correct_total / total if total else 0
    prec = surfaced_right / surfaced if surfaced else 0
    print(f"\noverall={overall:.1%}  macro={macro_mean:.1%}  "
          f"gate={GATE:.0%} -> {'PASS' if macro_mean >= GATE else 'FAIL'}")
    print(f"worst pair acc={min(macro):.1%}  skipped={skipped}")
    print(f"suggestions surfaced={surfaced}  precision={prec:.1%}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=recommend.ALPHA)
    ap.add_argument("--margin", type=float, default=recommend.SUGGEST_MARGIN)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balanced", action="store_true")
    a = ap.parse_args()
    evaluate(alpha=a.alpha, margin=a.margin, n=a.n, seed=a.seed,
             balanced=not a.no_balanced)