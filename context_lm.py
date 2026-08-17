"""Char-level masked transformer (ONNX) used to score candidate spellings by
sentence context.

Given a sentence and a target word region [start, end), every character of the
word AND every character after it (the trailing, possibly partial, remainder
of the sentence) is replaced with the <mask> id, and the model is run once.
Masking to end-of-sequence (rather than only the word region) prevents tokens
of a partially-written next word from leaking into the context: without it,
the trailing ``c`` in "i have c" could push "hawe" above "have". The log-prob
the model assigns to each character of a candidate spelling at the
corresponding position is summed to give that candidate's context score.

This is the "context brain" behind the recommendation system.
"""

import json
import os
from functools import lru_cache

import numpy as np
import onnxruntime as ort

from bpe_tokenizer import BPETokenizer

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_ROOT, "models", "lm", "model.onnx")
_VOCAB_PATH = os.path.join(_ROOT, "models", "lm", "vocab.json")

_PAD_ID = 0
_MASK_ID = 1

# The char-LM transformer has fixed positional embeddings for exactly this many
# tokens. Feeding a longer sequence makes ONNX fail to broadcast position
# embeddings (e.g. "64 by 66"). Sequences are truncated to this window while
# always keeping the scored word region intact.
MAX_SEQ_LEN = 64


def _fit_window(ids: list[int], tok_lo: int, tok_hi: int
                ) -> tuple[list[int], int, int]:
    """Truncate a token list to the model's MAX_SEQ_LEN window.

    Returns (ids', tok_lo', tok_hi') such that the word region [tok_lo, tok_hi)
    is fully preserved. Drops leading context first (never cutting into the word
    region) so as much useful left-context as possible is kept; any remaining
    overflow is dropped from the trailing, fully-<mask>-ed tail (safe: those
    tokens carry no real information).
    """
    L = len(ids)
    if L <= MAX_SEQ_LEN:
        return ids, tok_lo, tok_hi
    drop_front = min(L - MAX_SEQ_LEN, tok_lo)
    new_ids = ids[drop_front:]
    lo = tok_lo - drop_front
    hi = tok_hi - drop_front
    if len(new_ids) > MAX_SEQ_LEN:
        new_ids = new_ids[:MAX_SEQ_LEN]
        hi = min(hi, MAX_SEQ_LEN)
    return new_ids, lo, hi

# Completion gate: a sentence is considered "complete" when the causal LM
# assigns at least this probability to a sentence-ending token (period/EOS)
# AND the sentence has at least SENTENCE_END_MIN_WORDS words. Strict by design:
# short/fragment sentences under-trigger rather than over-trigger. Note: this
# small model essentially never ranks a period above a continuation token, so
# an absolute P(stop) threshold is used instead of a stop-vs-word margin.
SENTENCE_END_PROB = 0.13
SENTENCE_END_MIN_WORDS = 3

_session = None
_vocab = None


def _get_session():
    global _session
    if _session is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"char-LM onnx missing: {_MODEL_PATH} (copy from pi-learn-station)")
        _session = ort.InferenceSession(
            _MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _get_vocab():
    global _vocab
    if _vocab is None:
        with open(_VOCAB_PATH) as f:
            _vocab = json.load(f)["char2id"]
    return _vocab


def _tokenize(sentence: str):
    """Map sentence -> (token ids, char_offset list).

    offsets[j] is the character index in `sentence` for token j (None for
    dropped/unknown chars).
    """
    v = _get_vocab()
    ids = []
    offsets = []
    for i, ch in enumerate(sentence):
        if ch in v:
            ids.append(v[ch])
            offsets.append(i)
    return ids, offsets


def score_candidate(sentence: str, word_start: int, word_end: int,
                    candidate: str) -> float:
    """Context log-likelihood of `candidate` replacing [word_start, word_end).

    Masks every char in the word region plus every char after it (to the end of
    the token sequence), runs the masked transformer once, and sums the
    log-probs the model assigns to each of the candidate's chars at the
    corresponding positions.
    """
    ids, offsets = _tokenize(sentence)
    if not ids:
        return -1e9

    tok_lo = tok_hi = None
    for j, off in enumerate(offsets):
        if off >= word_start and tok_lo is None:
            tok_lo = j
        if off >= word_end:
            tok_hi = j
            break
    if tok_lo is None:
        return -1e9
    if tok_hi is None:
        tok_hi = len(ids)
    if len(candidate) != (tok_hi - tok_lo):
        return 0.0  # length mismatch -> flat prior (recognizer decides)

    ids, tok_lo, tok_hi = _fit_window(ids, tok_lo, tok_hi)

    masked = ids[:]
    for j in range(tok_lo, len(ids)):
        masked[j] = _MASK_ID
    seq = np.array([masked], dtype=np.int64)

    sess = _get_session()
    logits = sess.run(None, {"token_ids": seq})[0][0]  # (L, V)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-9)

    v = _get_vocab()
    score = 0.0
    for j, ch in enumerate(candidate):
        if ch in v:
            score += float(logp[tok_lo + j, v[ch]])
    return score


def score_word_candidates(sentence: str, word_start: int, word_end: int,
                          candidates: list[str]) -> dict[str, float]:
    """Context log-likelihoods of every `candidates` replacing [ws, we).

    All candidates for a word region share the SAME masked sequence (the word
    region plus all trailing tokens are masked to end-of-sequence), so a single
    forward pass scores them all (instead of one onnx run per candidate). Returns
    {candidate: score}; length-mismatched candidates get a flat 0.0 prior.
    """
    if not candidates:
        return {}
    ids, offsets = _tokenize(sentence)
    if not ids:
        return {c: -1e9 for c in candidates}

    tok_lo = tok_hi = None
    for j, off in enumerate(offsets):
        if off >= word_start and tok_lo is None:
            tok_lo = j
        if off >= word_end:
            tok_hi = j
            break
    if tok_lo is None:
        return {c: -1e9 for c in candidates}
    if tok_hi is None:
        tok_hi = len(ids)

    ids, tok_lo, tok_hi = _fit_window(ids, tok_lo, tok_hi)

    masked = ids[:]
    for j in range(tok_lo, len(ids)):
        masked[j] = _MASK_ID
    seq = np.array([masked], dtype=np.int64)

    sess = _get_session()
    logits = sess.run(None, {"token_ids": seq})[0][0]  # (L, V)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-9)

    v = _get_vocab()
    out: dict[str, float] = {}
    for cand in candidates:
        c = cand.strip().lower()
        if len(c) != (tok_hi - tok_lo):
            out[cand] = 0.0  # length mismatch -> flat prior (recognizer decides)
            continue
        score = 0.0
        for j, ch in enumerate(c):
            if ch in v:
                score += float(logp[tok_lo + j, v[ch]])
        out[cand] = score
    return out


def next_word_scores(sentence: str, candidates: list[str]) -> list[tuple[str, float]]:
    """Score each candidate as the next word following `sentence`.

    Uses left-context-only scoring: the candidate is appended to the sentence
    and scored in place, so the model's prediction depends only on the words
    the child has already written. Returns a sorted list of (word, logp),
    best first. Candidates not present in the vocab are skipped.
    """
    if not candidates:
        return []
    sep = " " if sentence and not sentence.endswith(" ") else ""
    base = sentence + sep
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        c = cand.strip().lower()
        if not c:
            continue
        ws, we = len(base), len(base) + len(c)
        sc = score_candidate(base + c, ws, we, c)
        if sc > -1e6:
            scored.append((c, sc))
    scored.sort(key=lambda x: -x[1])
    return scored


# ---------------------------------------------------------------------------
# Causal decoder LM (TinyStories-8M, GPT-Neo) for open-vocabulary next-word
# generation. Pure CPU via onnxruntime; char-level BPE handled by
# bpe_tokenizer.py. Lazy-loaded; call get_causal() once.
# ---------------------------------------------------------------------------

_CAUSAL_DIR = os.path.join(_ROOT, "models", "lm-causal")
_causal = None


class CausalNextWord:
    """Next-word generation with a small causal (GPT-Neo) language model.

    Encodes the sentence prefix with the GPT-2 byte-level BPE, then runs a
    small beam search so the returned words are complete dictionary-shaped
    words (not mid-word pieces). Falls back to an empty list if the model or
    tokenizer files are missing.
    """

    def __init__(self, model_dir=_CAUSAL_DIR, beam_width=4, max_tokens=12):
        self._tok = BPETokenizer(model_dir)
        model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"causal-LM onnx missing: {model_path}")
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self._eos = 50256  # GPT-2/GPT-Neo EOS
        self._beam_width = beam_width
        self._max_tokens = max_tokens
        self._cache: dict = {}

    def _run(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids (B, S) -> logits (B, S, V)."""
        B, S = token_ids.shape
        mask = np.ones((B, S), dtype=np.int64)
        pos = np.tile(np.arange(S, dtype=np.int64), (B, 1))
        return self._sess.run(
            None,
            {"input_ids": token_ids,
             "attention_mask": mask,
             "position_ids": pos})[0]

    def sentence_end_score(self, sentence: str) -> tuple[float, float]:
        """Stop preference for `sentence` as-is.

        Returns (stop_logp, top_word_logp): log-probability the model assigns to
        a sentence-ending token (period or EOS) vs the single best non-ending
        continuation, for the token position after `sentence`. Cached per
        sentence.
        """
        key = ("end", sentence)
        if key in self._cache:
            stop, top = self._cache[key]
            return stop, top
        ids = self._tok.encode(sentence)
        if not ids:
            return -1e9, -1e9
        logits = self._run(np.array([ids], dtype=np.int64))[0, -1, :]
        logp = logits - np.log(np.exp(logits).sum() + 1e-9)
        end_ids = [self._eos, 13]  # EOS + period
        stop = float(np.logaddexp(*[logp[t] for t in end_ids]))
        top = float(np.max(np.delete(logp, end_ids)))
        if len(self._cache) >= 64:
            self._cache.clear()
        self._cache[key] = (stop, top)
        return stop, top

    def sentence_is_complete(self, sentence: str) -> bool:
        """Strict completion gate for `sentence`.

        Complete when the model assigns >= SENTENCE_END_PROB probability to a
        sentence-ending token (period/EOS) and the sentence has at least
        SENTENCE_END_MIN_WORDS words.
        """
        import numpy as _np
        stop, _top = self.sentence_end_score(sentence)
        n_words = len(sentence.split())
        return float(_np.exp(stop)) >= SENTENCE_END_PROB and \
            n_words >= SENTENCE_END_MIN_WORDS

    def next_word(self, sentence: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k likely next words after `sentence`, best first.

        Returns (word, beam_score) pairs. `sentence` is the text written so far
        (without the missing word). Scores are cumulative log-probs. Results are
        cached per (sentence, k) so repeated strokes on the same context reuse
        the beam search instead of re-running it.
        """
        key = (sentence, k)
        if key in self._cache:
            return [tuple(x) for x in self._cache[key]]
        if len(self._cache) >= 64:
            self._cache.clear()
        out = self._next_word_uncached(sentence, k)
        self._cache[key] = [(w, float(s)) for w, s in out]
        return out

    def _next_word_uncached(self, sentence: str, k: int = 5):
        import re
        prefix_ids = self._tok.encode(sentence)
        if not prefix_ids:
            prefix_ids = [220]  # single leading space -> predict first word
        cands: dict[str, float] = {}
        beams: list[tuple[list[int], float]] = [([], 0.0)]
        completed: list[tuple[str, float]] = []

        for _step in range(self._max_tokens):
            if not beams:
                break
            nexts: list[tuple[list[int], float]] = []
            for gen, lp in beams:
                ids = np.array([prefix_ids + gen], dtype=np.int64)
                logits = self._run(ids)[0, -1, :]
                logp = logits - np.log(np.exp(logits).sum() + 1e-9)
                top = np.argsort(-logp)[: self._beam_width * 2]
                for tid in top:
                    nexts.append((gen + [int(tid)], lp + float(logp[tid])))
            nexts.sort(key=lambda x: -x[1])
            beams = nexts[: self._beam_width]

            pr = len(self._tok.decode(prefix_ids))
            keep: list[tuple[list[int], float]] = []
            for gen, lp in beams:
                if gen and gen[-1] == self._eos:
                    completed.append(
                        (self._tok.decode(prefix_ids + gen)[pr:], lp))
                    continue
                tail = self._tok.decode(prefix_ids + gen)[pr:]
                # A full word has been emitted when the tail contains a space
                # after the first word (a new word has begun).
                m = re.match(r"^\s*([a-zA-Z']+)\s+", tail)
                if m:
                    completed.append((tail, lp))
                else:
                    keep.append((gen, lp))
            beams = keep

        pool = completed + [
            (self._tok.decode(prefix_ids + g)[
                len(self._tok.decode(prefix_ids)):], lp) for g, lp in beams]
        for tail, lp in pool:
            m = re.match(r"^\s*([a-zA-Z']+)", tail or "")
            if m:
                w = m.group(1).lower()
                if w and w not in cands:
                    cands[w] = lp
        ranked = sorted(cands.items(), key=lambda x: -x[1])
        return ranked[:k]


def get_causal() -> CausalNextWord | None:
    """Lazily build (and cache) the causal next-word model. None if missing."""
    global _causal
    if _causal is None:
        try:
            _causal = CausalNextWord()
        except (FileNotFoundError, OSError):
            _causal = None
    return _causal