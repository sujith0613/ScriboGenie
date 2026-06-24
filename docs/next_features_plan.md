# ScriboGenie — Next Feature: Context-Aware Confusion Prediction

## Goal

When a user writes a sentence containing common dyslexic confusion letters (b/d, p/q, etc.), the system should:
1. Identify all valid word variations from the confusion map
2. Rank them by **sentence context** + **word frequency**
3. Show the top 3-5 ranked suggestions on the mobile PWA
4. Let the user tap to select the intended word

## Research: Common Confusion Patterns

### Letter Mirroring (exact shape flips)
| Pair | Type |
|------|------|
| b ↔ d | Left-right mirror |
| p ↔ q | Left-right mirror |
| u ↔ n | Upside-down |
| m ↔ w | Upside-down |
| p ↔ d | Partial rotation |

### Letter Substitutions (phonetic + visual)
| Confusion | Context |
|-----------|---------|
| i ↔ l | Visual similarity |
| f ↔ t | Visual similarity |
| a ↔ e | Vowel confusion |
| c ↔ s | Phonetic similarity |
| r ↔ n | Visual similarity |
| h ↔ n | Visual similarity |
| v ↔ w | Visual similarity |

### Common Word-Level Reversals
- was ↔ saw
- on ↔ no
- felt ↔ left
- act ↔ cat
- from ↔ form
- there ↔ three / their
- a ↔ and (misread small words)

### Digit-Letter Confusions (from EMNIST OCR)
| Confusion | Reason |
|-----------|--------|
| 0 ↔ o | Shape identical |
| 1 ↔ l | Shape identical |
| 2 ↔ z | Shape similar |
| 5 ↔ s | Shape similar |
| 6 ↔ b | Shape similar |
| 8 ↔ r | Shape similar (8 → B → b → r) |
| 9 ↔ g | Shape similar |

## Architecture

```
Normal flow (no change):
  Canvas → OCR → raw text → EMNIST correction → confusion variants → SpellChecker filter → single best word

New flow:
  Canvas → OCR → raw text → EMNIST correction → confusion variants → SpellChecker filter
        └→ ALL valid variants → Context Scorer (frequency × bigram) → top 5 ranked
        └→ WebSocket → Mobile PWA shows ranked list
        └→ User taps → selected word sent back → replaces current word
```

## Implementation Plan

### Phase 1: Backup & Baseline (session N+1)
- [ ] Create `docs/` folder in repo
- [ ] Save requirements + plan in MD files

### Phase 2: Enhanced Confusion Engine
- [ ] Extend `confusions` dict with full research list above
- [ ] Add word frequency scoring (from SpellChecker's `word_frequency` or bundled frequency list)
- [ ] Add bigram context map (~1000 common English bigrams, pre-built as JSON)
- [ ] Rewrite `dyslexia_aware_correction()` → returns ranked list instead of single word
- [ ] Score formula: `rank_score = freq(word) × bigram_cooccur(prev_word, word)`

### Phase 3: Mobile PWA Integration
- [ ] Extend WebSocket message to include `suggestions: [{word, score}, ...]`
- [ ] Update `mobile/index.html` to display tap-able suggestion chips
- [ ] Handle tap → send selected word back to server via WebSocket
- [ ] Server receives selection → updates current word in UI

### Phase 4: Edge Cases & Polish
- [ ] First word in sentence: use unigram frequency only (no previous context)
- [ ] Short words (<3 chars): limit confusion generation (too many false positives)
- [ ] Mobile suggestion timeout / dismiss
- [ ] Handle no valid variants → fall back to current behavior

## Files to Modify
- `app.py` — confusion engine, suggestion scoring, WebSocket message format
- `mobile/index.html` — suggestion chip UI, tap handler
- (New) `docs/confusion_map.json` — editable confusion patterns
- (New) `docs/bigram_freq.json` — common word-pair frequencies

## What NOT to Do
- ❌ No YOLO (object detection — unrelated)
- ❌ No BERT/GPT/transformers (100MB+ models, GPU, slow)
- ❌ No external APIs (keep offline)
- ❌ No new heavy dependencies (stick with `pyspellchecker` which is already there)
