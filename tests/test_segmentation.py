"""Segmentation tests for line-aware word/line splitting.

Renders synthetic multi-line handwriting (EMNIST glyph pool) and asserts:
  1. segment_components separates components into lines by y.
  2. merge_letters merges overlapping multi-stroke letters into one box.
  3. recommend_all with per-line structure groups words per line and keeps
     char offsets valid for the (joined) sentence.

Run:  .venv/bin/python -m unittest tests.test_segmentation -v
"""

import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import recognizer as rec
import recommend
import eval_suggestions as ev

LOGICAL_W, LOGICAL_H = 800, 370


def _render_lines(lines: list[str], rng: random.Random,
                  line_gap: int = 130) -> np.ndarray:
    """Render each string on its own row (top-to-bottom) onto a white canvas."""
    pool = ev.load_glyph_pool()
    canvas = np.full((LOGICAL_H, LOGICAL_W), 255, np.uint8)
    import cv2
    y = 40
    for sentence in lines:
        nchars = sum(1 for ch in sentence if ch != " ")
        nwords = sentence.count(" ") + 1
        avail = LOGICAL_W - 40 - nwords * ev.WORD_GAP - \
            (max(nchars - 1, 0)) * ev.CHAR_GAP
        scale = max(0.9, min(ev.SCALE, avail / (max(nchars, 1) * 28)))
        h = int(28 * scale)
        y0 = y
        x = 20
        for ch in sentence:
            if ch == " ":
                x += ev.WORD_GAP
                continue
            if ch not in pool:
                x += ev.CHAR_GAP
                continue
            img = rng.choice(pool[ch])
            big = cv2.resize(img, (int(28 * scale), int(28 * scale)),
                             interpolation=cv2.INTER_NEAREST)
            if x + big.shape[1] >= LOGICAL_W - 10:
                break
            ink = (big > 128).astype(np.uint8) * 255
            pen = max(3, int(ev.PEN * scale / ev.SCALE))
            ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            (pen, pen)))
            paper = 255 - ink
            canvas[y0:y0 + big.shape[0], x:x + big.shape[1]] = paper
            x += big.shape[1] + ev.CHAR_GAP
        y += h + line_gap
    return canvas


def _components(gray: np.ndarray):
    import cv2
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 8)
    num, _l, stats, _c = cv2.connectedComponentsWithStats(thr)
    boxes = []
    for i in range(1, num):
        if stats[i][4] < ev.MIN_AREA:
            continue
        x, y, w, h = stats[i][:4]
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


class SegmentationTest(unittest.TestCase):

    def test_two_lines_separate(self):
        rng = random.Random(7)
        gray = _render_lines(["cat", "dog"], rng)
        boxes = _components(gray)
        self.assertTrue(boxes, "no components found")
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        lines = rec.segment_components(x_boxes, y_boxes)
        # two rows -> two lines; each line holds 3 letters
        self.assertEqual(len(lines), 2, f"expected 2 lines, got {len(lines)}")
        self.assertTrue(all(len(l) == 3 for l in lines),
                        f"line sizes {[len(l) for l in lines]}")
        # top line letters all above bottom line letters
        top_max_y = max(y_boxes[i][1] for i in lines[0])
        bot_min_y = min(y_boxes[i][0] for i in lines[1])
        self.assertLess(top_max_y, bot_min_y, "lines not ordered top-to-bottom")

    def test_words_group_per_line(self):
        rng = random.Random(11)
        gray = _render_lines(["the cat", "dog"], rng)
        boxes = _components(gray)
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        lines = rec.segment_components(x_boxes, y_boxes)
        self.assertEqual(len(lines), 2, f"expected 2 lines, got {len(lines)}")
        rec_all = recommend.recommend_all(
            [{"a": 1.0} for _ in boxes], boxes, lines=lines)
        sentence = rec_all["sentence"]
        words = rec_all["words"]
        self.assertEqual(sentence.count(" "), 2,
                         f"sentence {sentence!r} should have 2 spaces")
        self.assertEqual(len(words), 3,
                         f"expected 3 words, got {[w['spelling'] for w in words]}")

    def test_merge_overlapping_letters(self):
        # two components that overlap in x and y -> same letter (multi-stroke)
        boxes = [(10, 10, 30, 40), (25, 10, 45, 40), (60, 10, 80, 40)]
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        line = list(range(3))
        merged = rec.merge_letters(line, x_boxes, y_boxes)
        # first two overlap in x -> merge into one; third separate
        self.assertEqual(len(merged), 2,
                         f"expected 2 merged boxes, got {len(merged)}")
        self.assertGreaterEqual(merged[0][2], 40,
                                "merged box should span both strokes")

    def test_no_merge_different_heights_close_x(self):
        # h (ascender) + a (x-height) at 6px gap are DIFFERENT letters.
        # Real fixture: child wrote "i am happy", app wrongly merged h+a (1.67)
        # and p+y (1.79) into single boxes read as 'w'/'y'.
        boxes = [
            (422, 69, 450, 134),   # h
            (456, 90, 503, 129),   # a
        ]
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        merged = rec.merge_letters([0, 1], x_boxes, y_boxes)
        self.assertEqual(len(merged), 2,
                         f"h+a must stay separate, got {len(merged)} box(es)")

    def test_i_am_happy_all_eight_letters(self):
        # Ground-truth regression: "i am happy" -> 8 letter boxes, not 6.
        # The 8 components (sorted by x) with real fixture boxes:
        boxes = [
            (68, 88, 123, 137),   # i
            (227, 87, 280, 141),  # a
            (288, 90, 331, 149),  # m
            (422, 69, 450, 134),  # h
            (456, 90, 503, 129),  # a
            (515, 90, 541, 154),  # p
            (555, 85, 591, 151),  # p
            (597, 84, 678, 202),  # y
        ]
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        merged = rec.merge_letters(list(range(8)), x_boxes, y_boxes)
        self.assertEqual(len(merged), 8,
                         f"expected 8 letter boxes, got {len(merged)}")

    def test_i_have_cats_strokes_stay_separate(self):
        # Ground-truth regression: "i have cats" -> i,h,a,v,e must stay separate
        # (no over-merge). The 5 "i have" components use real fixture boxes.
        boxes = [
            (34, 80, 94, 161),    # i
            (208, 50, 263, 169),  # h
            (285, 113, 321, 163), # a (x-height)
            (353, 76, 409, 161),  # v
            (422, 100, 461, 168), # e
        ]
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        merged = rec.merge_letters(list(range(5)), x_boxes, y_boxes)
        self.assertEqual(len(merged), 5,
                         f"i/h/a/v/e must stay separate, got {len(merged)}")

    def test_dot_stem_still_merges(self):
        # i-dot (h=12) + stem (h=57) overlap in x -> must STILL merge (ratio 4.75)
        # despite the height cap (cap only applies to gap>0).
        boxes = [
            (207, 123, 246, 135),  # dot
            (215, 70, 262, 127),   # stem
        ]
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        merged = rec.merge_letters([0, 1], x_boxes, y_boxes)
        self.assertEqual(len(merged), 1,
                         "dot+stem must merge into one i")

    def test_dot_attaches_to_stem_line(self):
        # Regression: an i-dot (y47-54) sits 2px above its stem (y56-114) but
        # their y-centers are ~35px apart. The old center-vs-first-box rule put
        # the dot on its own "line", which then read as a phantom 'h'. The
        # running-y-range rule must keep all three blobs on ONE line so
        # merge_letters fuses dot+stem into a single 'i'.
        boxes = [
            (27, 47, 85, 54),    # i dot
            (30, 56, 79, 114),   # i stem
            (165, 73, 196, 114), # next letter
        ]
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        lines = rec.segment_components(x_boxes, y_boxes)
        self.assertEqual(len(lines), 1,
                         f"dot must attach to stem's line, got {lines}")
        self.assertEqual(len(lines[0]), 3,
                         f"all 3 blobs on one line, got {lines}")
        # and merge_letters fuses dot+stem into one letter, leaving next separate
        merged = rec.merge_letters(lines[0], x_boxes, y_boxes)
        self.assertEqual(len(merged), 2,
                         f"dot+stem -> one letter + next letter, got {len(merged)}")
        self.assertEqual(merged[0], (27, 47, 85, 114),
                         f"i box should span dot+stem, got {merged[0]}")

    def test_group_words_large_letters_split(self):
        # Large child letters: widths ~86px push the old width-based threshold
        # (1.2*86=103px) above a real 60-70px word gap, so words merged into
        # one. Height-based threshold (0.5*90=45px) must still split them.
        x_boxes = [(20, 106), (130, 216), (280, 340)]
        y_boxes = [(30, 120), (30, 120), (30, 120)]
        words = rec.group_words(x_boxes, y_boxes)
        self.assertEqual(words, [[0, 1], [2]],
                         f"expected 2 words, got {words}")

    def test_group_words_small_letters_stay_one_word(self):
        # Small letters with an intra-word gap ~30px (below 0.5*height and the
        # 40px floor) must NOT split into separate words.
        x_boxes = [(20, 70), (95, 145), (170, 220)]
        y_boxes = [(30, 66), (30, 66), (30, 66)]
        words = rec.group_words(x_boxes, y_boxes)
        self.assertEqual(words, [[0, 1, 2]],
                         f"expected one word, got {words}")

    def test_group_words_uses_height_when_given(self):
        # Same x-gaps and widths: only the letter height differs, so it must be
        # height -- not width -- that decides the boundary. A 60px gap is a
        # word boundary for 100px-tall letters (0.5*100=50) but intra-word for
        # 200px-tall letters (0.5*200=100).
        x_boxes = [(20, 70), (140, 190)]
        tall_y = [(30, 230), (30, 230)]
        short_y = [(30, 130), (30, 130)]
        self.assertEqual(rec.group_words(x_boxes, tall_y), [[0, 1]])
        self.assertEqual(rec.group_words(x_boxes, short_y), [[0], [1]])

    def test_offsets_valid_for_lm(self):
        rng = random.Random(13)
        gray = _render_lines(["the cat", "dog"], rng)
        boxes = _components(gray)
        x_boxes = [(b[0], b[2]) for b in boxes]
        y_boxes = [(b[1], b[3]) for b in boxes]
        lines = rec.segment_components(x_boxes, y_boxes)
        rec_all = recommend.recommend_all(
            [{"a": 1.0} for _ in boxes], boxes, lines=lines)
        sentence = rec_all["sentence"]
        for w in rec_all["words"]:
            ws, we = w["offsets"]
            self.assertEqual(sentence[ws:we], w["spelling"],
                             f"offset {w['offsets']} mismatch for "
                             f"{w['spelling']!r} in {sentence!r}")


if __name__ == "__main__":
    unittest.main()
