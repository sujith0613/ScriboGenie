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
TIER_C = [("m", "n")]


def all_pairs(include_tier_c: bool = False) -> list[tuple[str, str]]:
    pairs = list(TIER_A) + list(TIER_B)
    if include_tier_c:
        pairs += TIER_C
    return pairs


def neighbours(letter: str, include_tier_c: bool = False) -> list[str]:
    pairs = all_pairs(include_tier_c)
    out = []
    for a, b in pairs:
        if a == letter:
            out.append(b)
        elif b == letter:
            out.append(a)
    return out


CONFUSION_LETTERS = sorted({l for p in all_pairs() for l in p})
