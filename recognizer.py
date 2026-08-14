"""Per-component recognizer (ONNX myCnn port, 62 EMNIST ByClass classes).

Normalization matches the ScriboGenie pipeline exactly:
    crop from gray -> cv2.resize(crop, (20,20)) (squash, not aspect-preserving)
    -> arr[4:24,4:24] = (255.0 - crop) / 255.0  (inverted grayscale, 4px border)

Returns, per component, an OrderedDict {lowercase_letter: prob} for the top-K
letters. Digits/uppercase are mapped to lowercase so the app (lowercase words)
just works (digits go through EMNIST_CORRECTIONS, uppercase is lowercased).

The ONNX model is a faithful port of ScriboGenie's myCnn.h5 (which is a Git-LFS
pointer not fetchable without sudo/git-lfs).
"""

import os
from collections import OrderedDict

import numpy as np
import onnxruntime as ort

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_ROOT, "models", "recog", "model.onnx")

# EMNIST ByClass class order (62): 0-9 digits, 10-35 A-Z, 36-61 a-z.
CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Recognized digits are almost always the letter that looks like it.
EMNIST_CORRECTIONS = {'0': 'o', '8': 'r', '5': 's', '1': 'l', '2': 'z',
                      '6': 'b', '9': 'g'}
TOP_K = 5

_session = None


def _get_session():
    global _session
    if _session is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"recognizer.onnx missing: {_MODEL_PATH}")
        _session = ort.InferenceSession(
            _MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def char_from_index(idx: int) -> str:
    ch = CHAR_LIST[idx]
    if ch in EMNIST_CORRECTIONS:
        return EMNIST_CORRECTIONS[ch]
    return ch.lower()


def normalize_component(crop: np.ndarray) -> np.ndarray:
    """Normalize a segmented GRAYSCALE crop exactly like ScriboGenie.

    crop: 2D float array of the component cut from the grayscale image
    (0..255, ink dark ~0, paper light ~255).

    Returns a 28x28 float32 array in [0,1] with ink=1, in a 4px border.
    """
    if crop.ndim != 2 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros((28, 28), dtype=np.float32)
    import cv2
    resized = cv2.resize(crop, (20, 20)).astype(np.float32)
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[4:24, 4:24] = (255.0 - resized) / 255.0
    return arr


def letter_probs_from_scores(row: np.ndarray,
                             top_k: int = TOP_K) -> OrderedDict:
    """Map one model output row (62 logits or already-softmaxed probs) to a
    top-K {lowercase_letter: prob} dict, applying EMNIST corrections + case
    folding. Shared by the TF app path and the ONNX eval so results agree."""
    row = np.asarray(row, dtype=np.float64)
    probs = np.exp(row) / np.exp(row).sum()
    order = np.argsort(row)[::-1][:top_k]
    d = OrderedDict()
    for idx in order:
        d[char_from_index(int(idx))] = float(probs[idx])
    return d


def group_words(x_boxes: list[tuple[int, int]]) -> list[list[int]]:
    """Group per-component x-ranges (x0, x1) into words by horizontal gap.

    A gap between consecutive components is a word boundary when it exceeds a
    multiple of the median component width (scale-robust) and a floor px.
    Returns a list of word groups, each a list of component indices.
    """
    if not x_boxes:
        return []
    widths = [x1 - x0 for x0, x1 in x_boxes]
    median_w = sorted(widths)[len(widths) // 2] if widths else 1.0
    threshold = max(24.0, 1.2 * median_w)

    words: list[list[int]] = [[0]]
    for i in range(1, len(x_boxes)):
        gap = x_boxes[i][0] - x_boxes[i - 1][1]
        if gap > threshold:
            words.append([i])
        else:
            words[-1].append(i)
    return words


def build_sentence_and_offsets(word_spellings: list[str]
                               ) -> tuple[str, list[tuple[int, int]]]:
    """Join word spellings with single spaces; return sentence + per-word
    (start, end) char offsets into it."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cur = 0
    for w in word_spellings:
        parts.append(w)
        start = cur
        end = cur + len(w)
        offsets.append((start, end))
        cur = end + 1  # trailing space slot
    return " ".join(parts), offsets


def recognize_crops(crops: list[np.ndarray],
                    top_k: int = TOP_K) -> list[OrderedDict]:
    """Batch-recognize grayscale component crops into top-K {letter: prob}."""
    if not crops:
        return []
    sess = _get_session()
    inp = sess.get_inputs()[0].name
    batch = np.stack([normalize_component(c) for c in crops])
    batch = batch[..., np.newaxis].astype(np.float32)  # NHWC (N,28,28,1)
    logits = sess.run(None, {inp: batch})[0]  # (N, 62)
    return [letter_probs_from_scores(row, top_k) for row in logits]