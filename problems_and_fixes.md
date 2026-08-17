# Problems & Fixes — ScriboGenie Pi

## 1. Keras 3 / TensorFlow 2.15 — Model Loading Failure

**Problem**: `myCnn.h5` was saved with Keras 3 (default in TF 2.15+), which uses different layer config key names. Loading the h5 file directly via `tf.keras.models.load_model()` failed with deserialization errors for layers like `FixedInputLayer`, or threw `TypeError` / `Unknown layer` exceptions.

**Fix**: Replaced `load_model()` with an explicit Functional API model built in code matching the original architecture, then used `model.load_weights(path, by_name=True)`. This bypasses Keras 3's config deserialization while correctly loading all trained weights.

```python
# Before — failed with Keras 3 config mismatch
model = tf.keras.models.load_model(MODEL_PATH)

# After — explicit architecture + load_weights by name
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
# ... build 3-block ResNet explicitly ...
model = tf.keras.Model(inputs, outputs)
model.load_weights(MODEL_PATH, by_name=True)
```

## 2. `FixedInputLayer` — Dead Custom Layer

**Problem**: The original code referenced a custom `FixedInputLayer` class that no longer existed and wasn't needed. Caused `Unknown layer: FixedInputLayer` errors if `load_model()` was used.

**Fix**: Removed the dead class entirely. The explicit ResNet build doesn't need it.

## 3. `queue.Empty` — Prediction Thread Spam

**Problem**: The worker thread's `tq.get(timeout=0.2)` raised `queue.Empty` every 200ms when no task was available. This was caught by a bare `except: pass`, which hid real errors and filled logs.

**Fix**: Added specific `except queue.Empty: continue` before the general exception handler.

```python
try:
    task = self.tq.get(timeout=0.2)
except queue.Empty:
    continue
except Exception:
    log.error(...)
```

## 4. Slow Prediction — PIL Overhead

**Problem**: Per-character image processing used `PIL.ImageDraw`, `ImageOps.invert`, and similar PIL calls — 7 allocations per character. Prediction called `model.predict()` which added TF Dataset validation overhead per batch.

**Fix**: 
- Replaced PIL pipeline with OpenCV + NumPy (2 allocations per character)
- Changed `model.predict(batch, verbose=0)` → `model.predict_on_batch(batch)`

## 5. Hidden Buttons — Tkinter Pack Order

**Problem**: The `controls` and `result_panel` frames were packed *after* `canvas_frame`, so they appeared at the bottom of the window. The canvas expanded to fill all space, pushing the result panel and buttons off-screen or making them invisible.

**Fix**: Changed pack order so `controls` and `result_panel` pack with `side="bottom"` *before* `canvas_frame`, ensuring they are positioned at the bottom while the canvas fills remaining space.

## 6. Smooth Drawing — Ovals vs Lines

**Problem**: Drawing used `create_oval` for every mouse move event, creating thousands of overlapping oval items on the canvas. This caused sluggish performance and jagged strokes, especially on Pi.

**Fix**: Replaced with `create_line` tracking (`_last_x, _last_y → e.x, e.y`). Creates a single line segment per mouse-move event instead of dozens of overlapping ovals.

## 7. Wacom Coordinate Offset

**Problem**: Wacom tablet coordinates (raw 0–15200 x, 0–9500 y) were mapped directly to canvas coordinates, causing the drawn stroke to appear far from the pen position (offset by screen/canvas position).

**Fix**: Mapped device coords → screen coords → canvas-local coords:
```python
local_x = (raw_x / 15200) * scr_w - cvs_x
local_y = (raw_y / 9500) * scr_h - cvs_y
```
Where `cvs_x/y` = `canvas.winfo_rootx/y()` and `scr_w/h` = `root.winfo_screenwidth/height()`.

## 8. Wacom Writing in Other Apps (Focus Gating)

**Problem**: When using the Wacom tablet, pen strokes appeared in other windows (browser, terminal) even when they weren't focused, because the tablet sent absolute coordinates globally.

**Fix**: Added `FocusIn`/`FocusOut` bindings with a `self._focused` flag. Drawing methods check `if not self._focused: return` before processing pen events.

## 9. Mid-Stroke Junk Predictions

**Problem**: The prediction timer triggered while the user was still drawing (within 1 second of last stroke), causing the model to predict on partial/incomplete handwriting.

**Fix**: Added `_last_draw_time` tracking. `_check_results` debounces: if `now - self._last_draw_time < 1.0`, it skips queueing a new prediction.

## 10. Stale Results on Clear / Next Word

**Problem**: Pressing "Clear" or "Next Word" left stale recognition results in the result queues. The old prediction would pop up on the new word's canvas.

**Fix**: Both `clear()` and `next_word()` drain `self.rq` and `self.tq` queues, reset `_last_pred_timer`, and clear the display labels.

## 11. Main Thread Blocking — PIL Image Copy

**Problem**: `_check_results` passed `self.image` to the prediction queue, which was modified on the main thread while the worker thread read it, causing race conditions.

**Fix**: Pass `self.image.copy()` (PIL copy) to the queue. Worker does `np.array(pil_img)` in its own thread — main thread is free to draw simultaneously.

## 12. Character Ordering — Unstable Predictions

**Problem**: ConnectedComponents returned components in arbitrary order, so predicted characters appeared in wrong sequence.

**Fix**: Sort components by X position (`stats[i][0]`) before processing left-to-right.

## 13. Debian 13 Trixie — Hotspot Configuration Broken

**Problem**: Debian 13 Trixie uses NetworkManager as the primary network manager and doesn't have `/etc/dhcpcd.conf`. The original `setup_hotspot.sh` wrote to `dhcpcd.conf` and relied on it for static IP configuration, which caused the hotspot to never appear.

**Fix**: Rewrote hotspot setup to:
- Disable NM management of wlan0: `nmcli dev set wlan0 managed no`
- Use `systemd-networkd` for static IP: `/etc/systemd/network/12-wlan0.network`
- Keep hostapd + dnsmasq for AP + DHCP (unchanged)

Also added `./setup_hotspot.sh restore` mode and menu Option 6 in `install_and_run.sh`.

## 14. Wacom Pad Buttons Not Working in App

**Problem**: Wacom pad buttons (mapped at driver level to Ctrl+Z, Ctrl+Y, Ctrl+E) did nothing in the app. No undo/redo/eraser functionality existed at all in the code.

**Fix**: Added full stroke history tracking, `undo()`, `redo()`, `toggle_eraser()` methods, and keyboard bindings for `Ctrl+Z/Y/E`. Wacom driver was already sending these keystrokes — the app just wasn't listening.

## 15. No Auto-Progression on Correct Word

**Problem**: When the predicted word matched the target, the app showed the result but never advanced to the next word. User had to manually press "Next Word" or tap "Finish" on mobile.

**Fix**: In `_check_results`, the corrected word is compared to the target word. If they match, the app auto-advances after 1.5 seconds with TTS "Correct! Well done!" and sends `correct: true` to mobile to auto-show the reward.

## 16. No Character-Level Error Feedback

**Problem**: When the predicted word was wrong, the app just showed the raw/corrected text with no indication of which characters were incorrect.

**Fix**: Added per-character comparison in `_check_results`. For each wrong position, TTS speaks: "Character 3 should be T, not R" (first 3 errors). Mobile shows wrong letter cards in red and a banner with feedback text.

## 17. Mobile "Finish" Button Stuck / No Auto-Response

**Problem**: Tapping "Finish" on mobile sent an `attempt` message, but the Pi's response only appeared in the mobile UI after a manual cycle. No auto-reward, no auto-next-word.

**Fix**: Mobile `handleMessage` now handles `correct: true` in recognition messages directly — flips letter cards green, shows reward sheet + confetti. The Continue button clears wrong/flipped states before requesting the next lesson.

## 18. Poll-Based `_check_results` — Idle CPU Waste

**Problem**: `root.after(500, self._check_results)` fired every 500ms even when no strokes were drawn, wasting CPU on the Pi.

**Fix**: Removed all `root.after(500, ...)` self-scheduling. Now purely event-driven: pen-up → 2s debounce → `_check_results` → queue → worker → `result_callback` → `root.after(0, _check_results)`. Zero idle timers. Also removed `root.after(100, ...)` at startup.

## 19. `_check_results` Called on Empty Canvas

**Problem**: `_check_results` called `self.image.copy()` on an all-white canvas when no strokes existed, wasting worker time.

**Fix**: Added `self._stroke_groups` guard before copying. Worker only receives tasks when strokes exist on canvas.

## 20. Initial `_last_pred_timer = time.time()` Missing

**Problem**: First prediction triggered immediately at startup (no pen strokes yet) because `_last_pred_timer` was 0.

**Fix**: Set `_last_pred_timer = time.time()` in `__init__` to prevent premature first prediction.

## 21. Duplicate Prediction Results Processed

**Problem**: Worker re-predicts same canvas state when multiple `_check_results` calls queue identical images, producing duplicate results that double-feedback to user.

**Fix**: `_check_results` caches `self._last_result`; skips processing when `corrected_text == self._last_result`.

## 22. Wacom Express Keys — Driver Keystroke Mapping Unreliable

**Problem**: Wacom pad buttons mapped to keystrokes (Ctrl+Z, etc.) via driver config were unreliable across reboots and OS updates.

**Fix**: Replaced keystroke approach with direct evdev `EV_KEY` handling in `_wacom_loop`: `BTN_0`→undo, `BTN_1`→redo, `BTN_2`→clear, `BTN_3`→speak, `BTN_STYLUS`→eraser toggle, `BTN_STYLUS2`→speak word. No driver config needed. Removed keyboard bindings for `Shift_L`/`Shift_R`/`Alt_L`/`Alt_R`.

## 23. Eraser Show-Through / Thick Undo Strokes

**Problem**: Erased area showed remaining artifacts; undo restored wrong stroke width.

**Fix**: Stored `"width": w` in each stroke group. `_rebuild_from_strokes` reads `group.get("width", 4)`. First-point oval scaled by width. `_eraser_size = 20` shown in status bar. `_pen_width = 4` for normal drawing.

## 24. Auto Level-Up on 3-Streak Not Broadcasting to Mobile

**Problem**: Level advanced on Pi but mobile dropdown stayed on old level. No notification sent.

**Fix**: Added `send_to_mobile_sync({"type": "level_up", "new_level": ...})` in `_check_results` correct branch. Mobile `handleMessage` updates `els.levelSelect.value` on `level_up` message.

## 25. NLTK Dependency for Word Bank

**Problem**: `setup_pi.sh` installed NLTK + corpus (~300MB) solely for word list.

**Fix**: Replaced with `pyspellchecker` top 3000 common words sorted by frequency. No corpus download, ~3MB pip package already in `requirements_pi.txt`. Fallback to hardcoded 15-word-per-level lists on failure. Setup script reverted — no nltk dependency.

## 26. Mobile "Pi sees:" Shows Stale Results From Previous Word

**Problem**: Mobile displayed recognition results from a previous word after user had already advanced. "Pi sees: CAME" showed while current word was "BETTER", confusing the user.

**Root cause**: Worker thread processes predictions asynchronously. When predictions are slow (common on Pi with TF 2.15), result for word N arrives after `next_word()` advanced to word N+1. `_check_results` compared result against `lesson_state["word"]` (now for word N+1), producing wrong feedback and sending stale data to mobile.

**Fix**: Added `self._word_gen` counter incremented in `clear()`. Each prediction task includes the current gen value. The worker passes it through into the result dict. `_check_results` compares result gen against `self._word_gen`; skips stale results (gen mismatch) with `log.debug(...)`. Fall-through still queues fresh predictions for current strokes. Key changes:
- `app_pi.py:378` — `self._word_gen = 0` in `__init__`
- `app_pi.py:591` — `self._word_gen += 1` in `clear()`
- `app_pi.py:265` — worker extracts `gen` from task tuple
- `app_pi.py:288` — worker passes `gen` in result dict
- `app_pi.py:495-497` — `_check_results` gen check skips stale
- `app_pi.py:568` — queue put includes `self._word_gen`

## 27. `send_to_mobile_sync` and `broadcast` Functions Missing

**Problem**: App crashed silently when attempting to send any message to mobile (`NameError`). Mobile showed no "Pi sees:" updates, no reward sheets, no lesson sync. Auto-progression stalled because `_check_results` threw before it could schedule `after(1500, next_word)`.

**Root cause**: Event-driven refactor (Fix #18) accidentally deleted the two module-level functions `broadcast()` and `send_to_mobile_sync()` from `app_pi.py`. Called at 6 call sites — none worked.

**Fix**: Re-added both functions between `start_http_server()` and `apply_emnist_context_correction()`. `broadcast()` accepts dict, serializes to JSON via `json.dumps()`, sends to all connected WebSocket clients via `asyncio.gather()` with `return_exceptions=True`. `send_to_mobile_sync()` wraps `broadcast()` for synchronous callers via `asyncio.run_coroutine_threadsafe()` with `mobile_loop.is_running()` guard.

```python
async def broadcast(data):
    if not mobile_clients: return
    msg = json.dumps(data) if isinstance(data, dict) else data
    await asyncio.gather(*(s.send(msg) for s in mobile_clients.copy()), return_exceptions=True)

def send_to_mobile_sync(data):
    if not mobile_loop or not mobile_loop.is_running(): return
    asyncio.run_coroutine_threadsafe(broadcast(data), mobile_loop)
```

## 28. Mobile Level Dropdown Invisible on Safari

**Problem**: `<select id="levelSelect">` with levels 1–6 didn't appear on mobile device Safari.

**Root cause**: Custom CSS (`border-radius`, `background: var(--navy-light)`, custom font) triggered implicit `-webkit-appearance: none` on iOS Safari, completely hiding the native dropdown widget.

**Fix**: Added `-webkit-appearance: menulist-button; appearance: auto; min-width: 90px; font-size: 14px` to force native dropdown rendering while preserving theme colors.

## 29. Auto-Progress + Mobile "Finish" Button Double-Score Same Word

**Problem**: Pi's auto-prediction processes correct word (scores +10), then user taps mobile "Finish" button which sends `attempt` message → `analyze_attempt` scores the same word again (+10 more). Word gets double-scored.

**Root cause**: Two independent paths (auto-prediction in `_check_results` and manual submission in `ws_handler` → `analyze_attempt`) both score the same recognition result.

**Fix**: Not yet fixed. Requires either: (a) making mobile view-only by removing the Finish button's `attempt` handler, or (b) adding a completion flag to prevent `analyze_attempt` from scoring an already-correct word.

## 30. `merge_letters` Over-Merges Close Different-Height Letters ("w"/"m" confusion)

**Problem**: Child wrote "i am happy"; app predicted "iawwpy". Debug geometry (logs/debug/run_1786774742/1786776102355_prediction.json) showed 8 components for 8 letters, but `merge_letters` merged two DIFFERENT-letter pairs into single boxes:
- h (ascender, h=65) + a (x-height, h=39) at a 6px gap → ratio 1.67 → read as "w"
- p (h=66) + y (deep descender, h=118) at a 6px gap → ratio 1.79 → read as "y"

The old rule merged any gap'd pair with y-overlap, so near-touching letters of different heights collapsed into one box, inflating "w"/"m"-looking shapes.

**Root cause**: merge condition (`gap <= merge_gap and (gap <= 0 or y_overlap)`) ignored height similarity. A 6px inter-letter gap is far below merge_gap (~7px at median letter width 47), so distinct letters merged.

**Fix**: Added a height-similarity cap applied ONLY when components do not overlap in x (gap > 0): a gap'd pair now merges only when its height ratio ≤ 1.5. Overlapping x (gap <= 0) still always merges, so dot+stem i's (ratio ~4.75) and two-stroke w/y (ratio ~1.05) are preserved. Verified across all 89 debug fixtures: the only remaining near-touch merges are legit same-x-height strokes (ratio ≤ 1.33, e.g. 'thassa' family); "i am happy" → 8 boxes → words ["i","am","happy"]; "i have cats" h/a/v/e stay separate. Eval gate unchanged/pass (overall 91.7%).

**Tests**: `test_no_merge_different_heights_close_x`, `test_i_am_happy_all_eight_letters`, `test_i_have_cats_strokes_stay_separate`, `test_dot_stem_still_merges` in tests/test_segmentation.py.
