"""Minimal byte-level BPE tokenizer for GPT-Neo / TinyStories (GPT-2 style).

Runtime dependency-free: reads vocab.json + merges.txt and implements the
standard GPT-2 byte-level BPE (same algorithm as tokenizers/transformers).
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(ROOT, "models", "lm-causal")

# Byte-to-unicode mapping used by GPT-2 byte-level BPE.
_BYTE_ENCODER = {}
_n = 0
for _b in range(ord("!"), ord("~") + 1):
    _BYTE_ENCODER[_b] = chr(_b)
    _n += 1
for _b in range(ord("¡"), ord("¬") + 1):
    _BYTE_ENCODER[_b] = chr(_b)
    _n += 1
for _b in range(ord("®"), ord("ÿ") + 1):
    _BYTE_ENCODER[_b] = chr(_b)
    _n += 1
_bs = 0
for _b in range(256):
    if _b not in _BYTE_ENCODER:
        _BYTE_ENCODER[_b] = chr(256 + _bs)
        _bs += 1
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}

_WHITESPACE_RE = re.compile(r"\s+")
_NONPRINT_RE = re.compile(r"[^ !-~]")


class BPETokenizer:
    def __init__(self, model_dir=_DIR):
        with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(os.path.join(model_dir, "merges.txt"), "r", encoding="utf-8") as f:
            merges = f.read().split("\n")
        merges = [m for m in merges if m and not m.startswith("#")]
        self.bpe_ranks = {}
        for i, m in enumerate(merges):
            left, right = m.split()
            self.bpe_ranks[(left, right)] = i
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.encoder)
        self.cache = {}

    def _byte_encode(self, text):
        return "".join(_BYTE_ENCODER[b] for b in text.encode("utf-8"))

    def _byte_decode(self, text):
        return bytes(_BYTE_DECODER[c] for c in text).decode("utf-8", errors="replace")

    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token
        while True:
            min_rank = min(
                (self.bpe_ranks[p] for p in pairs if p in self.bpe_ranks),
                default=None,
            )
            if min_rank is None:
                break
            bigram = next(
                p for p in pairs if self.bpe_ranks.get(p) == min_rank
            )
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(bigram[0], i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == bigram[0] and i + 1 < len(word) and word[i + 1] == bigram[1]:
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        self.cache[token] = " ".join(word)
        return self.cache[token]

    def encode(self, text, add_special_tokens=False):
        """Encode text to token ids. GPT-2 style: space-prefix all but first word."""
        text = _WHITESPACE_RE.sub(" ", text.strip())
        bpe_tokens = []
        words = _NONPRINT_RE.sub("", text).split()
        for i, token in enumerate(words):
            if i > 0:
                token = " " + token
            token = self._byte_encode(token)
            bpe_tokens.extend(self._bpe(token).split(" "))
        ids = [self.encoder[t] for t in bpe_tokens if t in self.encoder]
        return ids

    def decode(self, ids):
        """Decode token ids back to text."""
        text = "".join(self.decoder[i] for i in ids if i in self.decoder)
        return self._byte_decode(text)
