"""Tests for prefix-constrained current-word prediction (Fix B).

Runs without the LM: monkeypatches the char-LM scorer and next-word filler so
we only exercise the candidate-filtering + fill logic in recommend.py.

Run:  .venv/bin/python -m unittest tests.test_predictions -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import recommend


CANDIDATES = ["cat", "cake", "can", "car", "dog", "doll", "apple", "am"]


class PredictCurrentWordTest(unittest.TestCase):

    def _patch(self, scored, filled):
        rec = recommend
        original_scores = rec.context_lm.next_word_scores
        original_next = rec.next_word

        def fake_scores(context, cands):
            return [(w, scored.get(w, 0.5)) for w in cands]

        def fake_next(context, cands, k=5):
            return [{"word": w, "score": 0.0} for w in filled[:k]]

        rec.context_lm.next_word_scores = fake_scores
        rec.next_word = fake_next
        self.addCleanup(setattr, rec.context_lm, "next_word_scores",
                        original_scores)
        self.addCleanup(setattr, rec, "next_word", original_next)

    def test_prefix_filters_candidates(self):
        # child wrote "ca" -> only cat/cake/can/car survive
        self._patch({"cat": 0.9, "cake": 0.8, "can": 0.7, "car": 0.6},
                    filled=[])
        out = recommend.predict_current_word("i", "ca", CANDIDATES, k=5)
        words = [d["word"] for d in out]
        self.assertEqual(words, ["cat", "cake", "can", "car"],
                         f"expected prefix-matched order, got {words}")

    def test_scores_rank_prefix_matches(self):
        # scorer ranks cake above cat even though both match "ca"
        self._patch({"cake": 0.99, "cat": 0.1, "can": 0.5, "car": 0.4},
                    filled=[])
        out = recommend.predict_current_word("i", "ca", CANDIDATES, k=5)
        words = [d["word"] for d in out]
        self.assertEqual(words, ["cake", "can", "car", "cat"],
                         f"expected score-ranked order, got {words}")

    def test_fills_remaining_slots_with_context_words(self):
        # only one word matches "do" -> remaining slots come from filler
        self._patch({"dog": 0.9}, filled=["cat", "cake", "can", "car"])
        out = recommend.predict_current_word("i", "do", CANDIDATES, k=5)
        words = [d["word"] for d in out]
        self.assertEqual(words[0], "dog")
        self.assertEqual(len(words), 5, f"expected 5 slots, got {words}")
        self.assertNotIn("dog", words[1:], "no duplicate words allowed")

    def test_no_match_falls_back_to_context(self):
        # prefix "z" matches nothing -> pure context-based next-word list
        self._patch({}, filled=["cat", "cake", "can", "car", "dog"])
        out = recommend.predict_current_word("i", "z", CANDIDATES, k=5)
        self.assertEqual([d["word"] for d in out],
                         ["cat", "cake", "can", "car", "dog"])

    def test_empty_prefix_uses_next_word(self):
        self._patch({}, filled=["cat", "dog"])
        out = recommend.predict_current_word("i", "", CANDIDATES, k=5)
        self.assertEqual([d["word"] for d in out], ["cat", "dog"])

    def test_case_insensitive_prefix(self):
        self._patch({"cat": 0.9, "cake": 0.8, "can": 0.7, "car": 0.6},
                    filled=[])
        out = recommend.predict_current_word("i", "CA", CANDIDATES, k=5)
        self.assertEqual([d["word"] for d in out], ["cat", "cake", "can", "car"])


if __name__ == "__main__":
    unittest.main()
