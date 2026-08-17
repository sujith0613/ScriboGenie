"""Persistent progress store for ScriboGenie.

Tracks score, streak, total words, adult ratings, and badge unlocks, persisted
to a JSON file so badges/awards/recognitions survive restarts and stay
consistent between the desktop app and the mobile companion.

Independent module (no tensorflow/tkinter) so it can be unit-tested in
isolation. Run:

    .venv/bin/python -m unittest tests.test_progress -v
"""

import json
import logging
import os
import threading
import time

log = logging.getLogger("ScriboGenie.Progress")

DEFAULT_PROGRESS = {
    "score": 0,
    "streak": 0,
    "best_streak": 0,
    "total_words": 0,
    "ratings": [],
    "badges": [],
    "active_days": [],
}

# Server-authoritative badge definitions. Each entry: id, name, icon (matches
# the mobile SVG sprite), and a predicate (progress -> earned).
BADGE_DEFS = [
    {"id": "first_word", "name": "First Steps", "icon": "trophy",
     "earned": lambda p: p["total_words"] >= 1},
    {"id": "words_10", "name": "Perfect 10", "icon": "star",
     "earned": lambda p: p["total_words"] >= 10},
    {"id": "words_50", "name": "Super Writer", "icon": "star",
     "earned": lambda p: p["total_words"] >= 50},
    {"id": "rating_1", "name": "First Cheer", "icon": "trophy",
     "earned": lambda p: len(p["ratings"]) >= 1},
    {"id": "rating_10", "name": "Cheer Leader", "icon": "flame",
     "earned": lambda p: len(p["ratings"]) >= 10},
    {"id": "rating_25", "name": "Super Fan", "icon": "flame",
     "earned": lambda p: len(p["ratings"]) >= 25},
    {"id": "streak_3", "name": "3 Word Streak", "icon": "flame",
     "earned": lambda p: p["best_streak"] >= 3},
]


class ProgressStore:
    """Thread-safe, JSON-backed progress store.

    `path` may be overridden (tests) or None to disable persistence.
    """

    def __init__(self, path=None):
        self.path = path
        self.lock = threading.RLock()
        self.progress = self._fresh_progress()
        if path:
            self.load()

    @staticmethod
    def _fresh_progress():
        return {k: list(v) if isinstance(v, list) else v
                for k, v in DEFAULT_PROGRESS.items()}

    # ---- persistence ----
    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            with self.lock:
                self.progress.update(
                    {k: data.get(k, v) for k, v in DEFAULT_PROGRESS.items()})
        except Exception as e:
            log.warning("Failed to load progress file %s: %s", self.path, e)

    def save(self):
        if not self.path:
            return
        with self.lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.progress, f, indent=2)
            os.replace(tmp, self.path)

    # ---- state snapshot ----
    def state(self):
        with self.lock:
            return {
                "score": self.progress["score"],
                "streak": self.progress["streak"],
                "best_streak": self.progress["best_streak"],
                "total_words": self.progress["total_words"],
                "badges": list(self.progress["badges"]),
                "active_days": list(self.progress["active_days"]),
                "ratings": list(self.progress["ratings"][-20:]),
            }

    def _mark_active_day(self):
        day = time.strftime("%Y-%m-%d")
        if day not in self.progress["active_days"]:
            self.progress["active_days"].append(day)
            self.progress["active_days"] = self.progress["active_days"][-30:]

    def _check_badges(self):
        """Return list of newly unlocked badge dicts."""
        earned_ids = {b["id"] for b in self.progress["badges"]}
        unlocked = []
        for bdef in BADGE_DEFS:
            if bdef["id"] in earned_ids:
                continue
            if bdef["earned"](self.progress):
                self.progress["badges"].append({
                    "id": bdef["id"], "name": bdef["name"],
                    "icon": bdef["icon"], "earned_at": int(time.time()),
                })
                unlocked.append(self.progress["badges"][-1])
        return unlocked

    # ---- mutations ----
    def record_word_accepted(self):
        """Increment counters after an accepted word. Returns new badges."""
        with self.lock:
            self._mark_active_day()
            self.progress["total_words"] += 1
            self.progress["score"] += 5
            self.progress["streak"] += 1
            self.progress["best_streak"] = max(
                self.progress["best_streak"], self.progress["streak"])
            unlocked = self._check_badges()
            self.save()
            return unlocked

    def record_word_rejected(self):
        with self.lock:
            self.progress["streak"] = 0
            self.save()

    def record_rating(self, rating_id, word):
        """Store an adult's rating for a word. Returns new badges."""
        with self.lock:
            self.progress["ratings"].append({
                "ts": int(time.time()), "rating": rating_id, "word": word,
            })
            self.progress["ratings"] = self.progress["ratings"][-200:]
            unlocked = self._check_badges()
            self.save()
            return unlocked

    def reset(self):
        with self.lock:
            self.progress = self._fresh_progress()
            self.save()
