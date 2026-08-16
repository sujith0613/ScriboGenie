# ScriboGenie — Key Things To Remember

Verification-backed project facts. Do not re-litigate what is recorded here
without new empirical evidence that contradicts it (and then update this file).

## Model orientation: characters are UPRIGHT, never flip input

- The glyph pool `data/processed/val62_uint8.npz` stores **upright** letters
  (ink=255 on bg=0). The EMNIST "flipped by default" storage quirk (raw rows are
  rotated ~90° + mirrored) was **already corrected when the pool was built**.
- `model.onnx` (the recognizer) reads upright crops correctly:
  `i/m/n/o/y/j` ≥0.95, and eval sentences `am→am`, `mice→mice`, `you→you`.
- **Do NOT flip/rotate crop input.** `fliplr`, `T`, `rot180`, and the training
  notebook's `fix_emnist_orientation` all make recognition *worse* in practice.
  `recognizer.normalize_component` (no flip) is correct as-is.
- YOLO was removed (2026-08-15); the app uses connected-component
  segmentation exclusively. No flip is applied to crop input.

## Root-cause of `m`→"uhnn" / `y`→"uy": over-segmentation

- Multi-hump/multi-stroke letters (`m`, `y`) previously got split into 2–3
  overlapping boxes, each reading as a wrong letter (e.g. `m`→`n n n`,
  `y`→`u y`).
- The component pipeline merges multi-stroke letters via `recognizer.merge_letters`,
  which requires an actual x-overlap (gap ≤ 0) before fusing — a positive gap
  never merges, so distinct letters written close together stay separate.
- The app must restart after any code change to load it (current run, log
  `/tmp/scribo_run.log`).

## Recognizer form-confusion (not segmentation, not orientation)

- `a`→`u` (script "ɑ": tall stem + looped bowl) and `e`→`r`/`q` are the CNN
  confusing unusual *handwritten letterforms*. They are neither over-
  segmentation nor an orientation problem. Fixing them needs retraining /
  augmentation of those forms, not a pipeline hack.

## Confusion-pair policy

- The 8 dyslexia pairs (b/d, b/p, d/q, d/t, f/v, g/k, p/q, s/z) are handled by
  the correction layer (`recommend.recommend_all`). Treat them as *expected*,
  not as recognizer bugs. Do not modify `recommend.recommend_all` to "fix"
  something already handled upstream.

## Conventions & environment

- Debug dumps write JSON and PNG with different ms suffixes — match them by
  numeric prefix (e.g. `...3977042_prediction.json` ↔ `...3977043_prediction.png`).
- espeak is NOT installed (needs sudo) — TTS is silently disabled.