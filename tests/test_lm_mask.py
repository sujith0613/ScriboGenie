"""Tests for the LM mask-to-end fix (context_lm.py).

When scoring a word region, every token after the region start is masked to
the end of the sequence -- including the tokens of a partially-written next
word. Without this, the trailing "c" in "i have c" leaks into the masked-LM
context and can push "hawe" above the correct "have".

Run:  .venv/bin/python -m unittest tests.test_lm_mask -v
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import context_lm

_MODEL = Path(__file__).resolve().parents[1] / "models" / "lm" / "model.onnx"


@unittest.skipUnless(_MODEL.exists(), "char-LM onnx missing")
class LMMaskToEndTest(unittest.TestCase):

    def test_partial_next_word_does_not_flip_hawe(self):
        # Child wrote "i have c" (c is the partial next word). Scoring the
        # completed word region "have" must NOT let the trailing "c" boost
        # "hawe" above "have".
        s = "i have c"
        have = context_lm.score_candidate(s, 2, 6, "have")
        hawe = context_lm.score_candidate(s, 2, 6, "hawe")
        self.assertGreater(
            have, hawe,
            f"expected 'have' > 'hawe' with mask-to-end, got {have} vs {hawe}")

    def test_word_candidates_batch_same_behavior(self):
        s = "i have c"
        scores = context_lm.score_word_candidates(s, 2, 6, ["have", "hawe"])
        self.assertGreater(scores["have"], scores["hawe"],
                           f"scores flipped: {scores}")

    def test_first_word_still_scored(self):
        # Sanity: a normal mid-sentence word still gets sensible context.
        # Both candidates must match the region length (1 char) to be scored.
        s = "i have a cat"
        scores = context_lm.score_word_candidates(s, 8, 9, ["a", "u"])
        self.assertGreater(scores["a"], scores["u"],
                           f"expected 'a' > 'u' after 'have', got {scores}")


if __name__ == "__main__":
    unittest.main()
