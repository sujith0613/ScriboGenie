# Feature Requirements: Context-Aware Confusion Prediction

## Why

Learners with dyslexia/dysgraphia write letters that look or sound similar. For example:
- Writing "bog" instead of "dog" (b↔d mirror)
- Writing "saw" instead of "was" (word reversal)
- Writing "cAT" instead of "cat" (digit-letter confusion from OCR)

The current system picks ONE correction using SpellChecker. But the user might have intended any one of several valid variations. **We need to show all likely candidates** and let the mobile app help the user pick the right one.

## Core Requirements

### R1 — Confusion Map (extensible)
- Must cover: letter mirror pairs, phonetic substitutions, digit-letter confusions, common word reversals
- Must be editable without code changes (JSON file)
- Must allow weighted confusions (some are more likely than others)

### R2 — Variant Generation
- Given a word with N confusion-eligible chars, generate all valid dictionary variants
- Must use SpellChecker's dictionary (already loaded) as the filter
- Must limit explosion: skip variants for words >8 chars or with >4 confusion points

### R3 — Context Scoring
- Score each valid variant by: `word_frequency × bigram_likelihood(with_previous_word)`
- Must handle first-word-of-sentence case (unigram only)
- Must normalize scores to 0-1 range for display

### R4 — Mobile Display
- Show top 3-5 suggestions as tap-able chips below the current word
- Each chip shows: word + relevance bar (visual indicator)
- Tapping a chip replaces the current word and updates the display

### R5 — Communication
- WebSocket message `{type: "suggestions", words: [{word, score}, ...], context_word: "..."}`
- Mobile sends back: `{type: "select_suggestion", word: "..."}`
- Server updates UI and continues with the selected word

### R6 — Performance
- All computation must complete in <50ms (under the prediction debounce)
- No new ML models, no GPU, no external APIs
- Must work on Raspberry Pi 4 (1.5GHz ARM) without lag

## Non-Requirements
- Not a full language model (no grammar, no syntax trees)
- Not a speech recognizer
- Not a handwriting recognizer replacement (OCR stays as-is)
- Not an auto-complete / next-word predictor (only corrects what's written)
