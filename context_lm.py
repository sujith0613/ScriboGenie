"""Char-level masked transformer (ONNX) used to score candidate spellings by
sentence context.

Given a sentence and a target word region [start, end), every character of the
word is replaced with the <mask> id and the model is run once; the log-prob the
model assigns to each character of a candidate spelling at the corresponding
position is summed to give that candidate's context score.

This is the "context brain" behind the recommendation system.
"""

import json
import os

import numpy as np
import onnxruntime as ort

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_ROOT, "models", "lm", "model.onnx")
_VOCAB_PATH = os.path.join(_ROOT, "models", "lm", "vocab.json")

_PAD_ID = 0
_MASK_ID = 1

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

    Masks every char in the word region, runs the masked transformer once, and
    sums the log-probs the model assigns to each of the candidate's chars at
    the corresponding positions.
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

    masked = ids[:]
    for j in range(tok_lo, tok_hi):
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