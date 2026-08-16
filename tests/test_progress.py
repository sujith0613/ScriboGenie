"""Tests for the persistent progress store (badges, ratings, persistence).

Run:  .venv/bin/python -m unittest tests.test_progress -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from progress_store import ProgressStore


class ProgressStoreTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "progress.json")
        self.store = ProgressStore(self.path)
        self.store.reset()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_word_badges_1_10_50(self):
        ids = []
        for _ in range(1):
            ids.extend(b["id"] for b in self.store.record_word_accepted())
        self.assertIn("first_word", ids)
        self.assertNotIn("words_10", ids)

        ids = []
        for _ in range(9):
            ids.extend(b["id"] for b in self.store.record_word_accepted())
        self.assertIn("words_10", ids)
        self.assertNotIn("words_50", ids)

        ids = []
        for _ in range(40):
            ids.extend(b["id"] for b in self.store.record_word_accepted())
        self.assertIn("words_50", ids)

        st = self.store.state()
        self.assertEqual(st["total_words"], 50)
        self.assertEqual(st["score"], 250)

    def test_rating_badges_1_10_25(self):
        ids = []
        ids.extend(b["id"] for b in self.store.record_rating("awesome", "cat"))
        self.assertIn("rating_1", ids)

        ids = []
        for _ in range(9):
            ids.extend(b["id"] for b in self.store.record_rating("great", "cat"))
        self.assertIn("rating_10", ids)

        ids = []
        for _ in range(15):
            ids.extend(b["id"] for b in self.store.record_rating("awesome", "dog"))
        self.assertIn("rating_25", ids)

        st = self.store.state()
        self.assertEqual(len(self.store.progress["ratings"]), 25)
        self.assertEqual(len(st["ratings"]), 20)
        self.assertEqual(self.store.progress["ratings"][-1]["rating"], "awesome")

    def test_streak_3_badge(self):
        ids = []
        for _ in range(3):
            ids.extend(b["id"] for b in self.store.record_word_accepted())
        self.assertIn("streak_3", ids)
        self.assertEqual(self.store.state()["best_streak"], 3)

    def test_rejected_resets_streak(self):
        self.store.record_word_accepted()
        self.store.record_word_accepted()
        self.store.record_word_rejected()
        st = self.store.state()
        self.assertEqual(st["streak"], 0)
        self.assertEqual(st["best_streak"], 2)
        self.assertEqual(st["total_words"], 2)

    def test_no_duplicate_badges(self):
        self.store.record_word_accepted()
        badges_again = self.store.record_word_accepted()
        earned = [b["id"] for b in self.store.state()["badges"]]
        self.assertEqual(earned.count("first_word"), 1)
        self.assertEqual(badges_again, [])

    def test_persistence_roundtrip(self):
        self.store.record_word_accepted()
        self.store.record_rating("great", "cat")

        fresh = ProgressStore(self.path)
        st = fresh.state()
        self.assertEqual(st["total_words"], 1)
        self.assertEqual(st["score"], 5)
        self.assertEqual(len(st["ratings"]), 1)
        self.assertIn("first_word", [b["id"] for b in st["badges"]])
        self.assertEqual(st["active_days"], self.store.state()["active_days"])

    def test_file_content_is_valid_json(self):
        self.store.record_word_accepted()
        self.store.record_rating("awesome", "sun")
        with open(self.path) as f:
            data = json.load(f)
        self.assertEqual(data["total_words"], 1)
        self.assertEqual(data["ratings"][-1]["word"], "sun")


if __name__ == "__main__":
    unittest.main()
