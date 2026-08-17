"""Confusion pairs used to generate spelling candidates for the char-LM
recommendation layer.

Evidence-backed pairs (see pi-learn-station docs/research.md):

- Tier A — mirror/reversible letters: b<->d, p<->q (left-right), b<->p,
  d<->q (up-down). The only lowercase Latin letters differing solely by
  orientation.
- Tier B — voiceless/voiced cognates: d<->t, g<->k, f<->v, s<->z.
- Tier C — stretch (off by default): m<->n.

Kept in sync with the backend copy in pi-learn-station.
"""

TIER_A = [("b", "d"), ("p", "q"), ("b", "p"), ("d", "q")]
TIER_B = [("d", "t"), ("g", "k"), ("f", "v"), ("s", "z")]
# Visual similarity (dyslexia research): mirror/rotation + shape look-alikes.
TIER_C = [("m", "n"), ("u", "n"), ("m", "w"), ("h", "n"), ("i", "l"),
          ("r", "n"), ("v", "w"), ("c", "s")]
# Handwriting look-alikes: lowercase letters a child may write so small/messy
# that the EMNIST recognizer confuses them (most commonly e misread as r).
TIER_D = [("e", "r"), ("e", "c"), ("e", "o"), ("e", "u"),
          ("a", "o"), ("o", "u"), ("i", "j"), ("u", "v"),
          ("c", "o"), ("o", "e"), ("n", "r")]


def all_pairs(include_tier_c: bool = True) -> list[tuple[str, str]]:
    pairs = list(TIER_A) + list(TIER_B)
    if include_tier_c:
        pairs += TIER_C
    pairs += TIER_D
    return pairs


def neighbours(letter: str, include_tier_c: bool = True) -> list[str]:
    pairs = all_pairs(include_tier_c)
    out = []
    for a, b in pairs:
        if a == letter:
            out.append(b)
        elif b == letter:
            out.append(a)
    return out


CONFUSION_LETTERS = sorted({l for p in all_pairs() for l in p})
