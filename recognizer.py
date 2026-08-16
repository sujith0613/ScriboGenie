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
    folding. Shared by the TF app path and the ONNX eval so results agree.

    Handles both output conventions: if the row is already a normalized
    probability distribution (values in [0,1], sum ~= 1), it is used directly;
    otherwise it is treated as raw logits and softmaxed. When multiple EMNIST
    indices fold to the same lowercase letter (e.g. 'I' and 'i'), their
    probabilities are summed rather than letting the later low-scoring index
    overwrite the confident one.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.size == 0:
        return OrderedDict()
    if float(np.max(row)) <= 1.0 and float(np.min(row)) >= 0.0 \
            and abs(float(row.sum()) - 1.0) < 1e-3:
        probs = row.copy()
    else:
        probs = np.exp(row) / np.exp(row).sum()
    agg: dict[str, float] = {}
    for idx, p in enumerate(probs):
        ch = char_from_index(idx)
        agg[ch] = agg.get(ch, 0.0) + float(p)
    order = sorted(agg.items(), key=lambda kv: -kv[1])[:top_k]
    d = OrderedDict()
    for ch, p in order:
        d[ch] = p
    return d


def segment_components(x_boxes: list[tuple[int, int]],
                       y_boxes: list[tuple[int, int]]) -> list[list[int]]:
    """Cluster per-component boxes into lines, each line = component indices
    sorted left-to-right.

    A component belongs to the current line when its y-range overlaps the
    running line band (the line's current min_y0/max_y1 expanded by a band
    derived from the median component height) or the vertical gap to that band
    is small. A separate, taller/typed component that is merely near another
    letter's band joins it too, so a dot sitting just above its stem (i, j)
    stays on the same line instead of becoming a phantom line/letter.
    Lines are returned top-to-bottom.
    """
    if not x_boxes:
        return []
    heights = [y1 - y0 for y0, y1 in y_boxes]
    median_h = sorted(heights)[len(heights) // 2] if heights else 1.0
    # Small vertical tolerance: enough for slightly misaligned letters and for
    # a dot to sit with its own stem, but small enough that a real second line
    # (even one written close below on a short canvas) does not merge into the
    # first. This was 0.75*median_h, which swallowed any line spaced by up to
    # 3/4 of a letter height and interleaved two rows into one ("cdaotg").
    band = max(6.0, 0.3 * median_h)

    # Work in reading order: cluster by y first, then sort each line by x.
    items = sorted(range(len(x_boxes)),
                   key=lambda i: (y_boxes[i][0] + y_boxes[i][1]) / 2.0)
    lines: list[list[int]] = []
    line_y0: list[int] = []  # running min_y0 per line
    line_y1: list[int] = []  # running max_y1 per line
    for i in items:
        y0, y1 = y_boxes[i]
        h = y1 - y0
        # Only a small component (a dot over an i/j) may join a line with a
        # vertical gap; a full-height letter must actually overlap the line's
        # y-band, otherwise a second line written close below would merge into
        # the first and interleave its letters into one row.
        tol = band if h < 0.5 * median_h else 0.0
        placed = False
        for li, line in enumerate(lines):
            if not line:
                continue
            # join when this component's y-range is within `tol` of the
            # line's running band (overlap OR a small vertical gap for dots).
            if y1 >= line_y0[li] - tol and y0 <= line_y1[li] + tol:
                line.append(i)
                line_y0[li] = min(line_y0[li], y0)
                line_y1[li] = max(line_y1[li], y1)
                placed = True
                break
        if not placed:
            lines.append([i])
            line_y0.append(y0)
            line_y1.append(y1)

    # Re-attach orphan dots (i, j): a lone small component that the tight band
    # left on its own belongs to the line directly below it whose horizontal
    # range it overlaps (preferring the closest line), so the dot+stem still
    # form one crop and one line.
    if len(lines) > 1:
        small: list[int] = []
        big: list[int] = []
        for li, line in enumerate(lines):
            if (len(line) == 1 and
                    (line_y1[li] - line_y0[li]) <= 0.5 * median_h):
                small.append(li)
            else:
                big.append(li)
        for si in small:
            dot = lines[si][0]
            x0, x1 = x_boxes[dot]
            y1 = y_boxes[dot][1]
            best, best_gap = None, None
            for bi in big:
                lx0 = min(x_boxes[j][0] for j in lines[bi])
                lx1 = max(x_boxes[j][1] for j in lines[bi])
                ov = min(x1, lx1) - max(x0, lx0)
                if ov <= 0:
                    continue
                gap = line_y0[bi] - y1
                if gap < 0:  # the dot sits below this line -> not its stem
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = bi, gap
            if best is not None:
                lines[best].append(dot)
                lines[best].sort(key=lambda j: x_boxes[j][0])
                line_y0[best] = min(line_y0[best], y_boxes[dot][0])
                line_y1[best] = max(line_y1[best], y_boxes[dot][1])
        lines = [lines[li] for li in range(len(lines)) if li not in small]

    for line in lines:
        line.sort(key=lambda i: x_boxes[i][0])
    return lines


def merge_letters(line: list[int],
                  x_boxes: list[tuple[int, int]],
                  y_boxes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge components of a line that are strokes of the same letter.

    Two consecutive components (in x) belong to the same letter only when their
    x-ranges actually overlap (gap <= 0). Merging requires an actual x-overlap
    plus either a clear y-overlap (side-by-side fragments of one multi-stroke
    letter) or a substantial shared x-column (vertically stacked fragments of a
    single tall stroke, e.g. the dot+stem of an ``i``).

    A positive x-gap is never merged. Distinct letters written close together
    (a child's tight handwriting) otherwise collapse into one box, silently
    dropping letters. Returns merged (x0, y0, x1, y1) boxes in left-to-right
    order.
    """
    if not line:
        return []
    merged: list[tuple[int, int, int, int]] = []
    for i in line:
        x0, x1 = x_boxes[i]
        y0, y1 = y_boxes[i]
        if merged:
            lx0, ly0, lx1, ly1 = merged[-1]
            if x0 <= lx1:  # x-ranges overlap (gap <= 0)
                ov_w = min(x1, lx1) - max(x0, lx0)
                narrow_w = min(x1 - x0, lx1 - lx0)
                y_overlap = not (y1 < ly0 or ly1 < y0)
                # Fuse side-by-side fragments (y-overlap) OR stacked fragments
                # of one tall letter (substantial shared x-column).
                if y_overlap or ov_w >= 0.5 * max(1, narrow_w):
                    merged[-1] = (min(lx0, x0), min(ly0, y0),
                                  max(lx1, x1), max(ly1, y1))
                    continue
        merged.append((x0, y0, x1, y1))
    return merged


def group_words(x_boxes: list[tuple[int, int]],
                y_boxes: list[tuple[int, int]] | None = None
                ) -> list[list[int]]:
    """Group per-component x-ranges (x0, x1) into words by horizontal gap.

    A gap between consecutive components is a word boundary when it exceeds a
    multiple of the median letter HEIGHT (scale-robust) and a floor px. Letter
    height (x-height) is far more stable than width for a child's handwriting
    (an ``m`` and an ``i`` differ hugely in width, little in height), so
    thresholding on height separates real word gaps from intra-word gaps even
    when letters are large. When ``y_boxes`` is omitted, falls back to the
    legacy width-based threshold. Returns a list of word groups, each a list
    of component indices.
    """
    if not x_boxes:
        return []
    if y_boxes is not None:
        heights = [y1 - y0 for y0, y1 in y_boxes]
        median_h = sorted(heights)[len(heights) // 2] if heights else 1.0
        threshold = max(40.0, 0.5 * median_h)
    else:
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
    # Compendious EMNIST-VGG: NCHW (N,1,28,28), normalize (x-0.1307)/0.3081,
    # then transpose H<->W (EMNIST stored-rotated convention).
    batch = ((batch - 0.1307) / 0.3081)[:, np.newaxis, ...].astype(np.float32)
    batch = batch.transpose(0, 1, 3, 2)
    logits = sess.run(None, {inp: batch})[0]  # (N, 62)
    return [letter_probs_from_scores(row, top_k) for row in logits]