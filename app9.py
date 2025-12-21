# handwriting_app_modeA.py
# Mode A: fixed logical canvas 800x400, visually scalable display + background
import tkinter as tk
from tkinter import Canvas, Frame, Button, Label, Scale, Scrollbar, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFilter
import numpy as np
import cv2
import os
import threading
import queue
import time
from tensorflow.keras.models import load_model
from spellchecker import SpellChecker
from itertools import product
import hashlib

# --------------------------- CONFIG ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "myCnn.h5")

DEBUG_SAVE = False
LIVE_POLL_INTERVAL_MS = 400
MAX_FPS = 3
LOGICAL_W, LOGICAL_H = 800, 400   # fixed internal pad size (Mode A)
DOWNSAMPLE_MAX_WIDTH = 800
DEBUG_DIR = "debug_pro"
os.makedirs(DEBUG_DIR, exist_ok=True)

# --------------------------- LOAD MODEL ---------------------------
print("Loading model...", end=" ", flush=True)
model = load_model(model_path)
print("Done ✅")

char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# --------------------------- SPELLCHECK / DYSLEXIA ---------------------------
confusions = {'b':'d','d':'b','p':'q','q':'p','i':'l','l':'i','1':'l','l':'1','0':'o','o':'0','r':'8','8':'r'}
spell = SpellChecker()

def correct_word(word):
    if len(word) <= 1:
        return word
    corrected = spell.correction(word)
    return corrected if corrected else word

def generate_dyslexic_variants(word):
    chars_options = [[c, confusions[c]] if c in confusions else [c] for c in word]
    return [''.join(p) for p in product(*chars_options)]

def dyslexia_aware_correction(word):
    variants = generate_dyslexic_variants(word)
    valid_candidates = [w for w in variants if spell.correction(w) == w]
    return valid_candidates[0] if valid_candidates else correct_word(word)

# --------------------------- IMAGE PREPROCESS (for model) ---------------------------
def preprocess_char_pil(pil_img, name=None):
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    img.thumbnail((20,20), Image.Resampling.LANCZOS)
    new_img = Image.new("L", (28,28), 0)
    w,h = img.size
    new_img.paste(img, ((28-w)//2, (28-h)//2))
    arr = np.array(new_img).astype("float32") / 255.0
    arr = arr.reshape(28,28,1)
    if DEBUG_SAVE and name:
        new_img.save(os.path.join(DEBUG_DIR, f"{name}.png"))
    return arr

# --------------------------- SEGMENTATION HELPERS ---------------------------
def downsample_gray(gray, max_width=DOWNSAMPLE_MAX_WIDTH):
    h, w = gray.shape
    if w <= max_width:
        return gray, 1.0
    factor = max_width / w
    new_w = int(w * factor)
    new_h = int(h * factor)
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, factor

def binarize(gray):
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thr = cv2.dilate(thr, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    return thr

def connected_components_boxes(thr):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x,y,w,h,area = stats[i]
        if area < 12:
            continue
        boxes.append((x,y,w,h,area))
    return [(x,y,w,h) for (x,y,w,h,area) in boxes]

def merge_dot_stems(boxes, max_dot_size=14, max_vertical_gap=20):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    dots = []
    stems = []
    for b in boxes:
        x,y,w,h = b
        if w <= max_dot_size and h <= max_dot_size:
            dots.append(b)
        else:
            stems.append(b)
    merged = []
    used_dots = set()
    for stem in stems:
        x_s,y_s,w_s,h_s = stem
        chosen_dot_idx = None
        best_score = None
        for i,dot in enumerate(dots):
            if i in used_dots: continue
            x_d,y_d,w_d,h_d = dot
            if y_d + h_d < y_s and (y_s - (y_d + h_d)) <= max_vertical_gap:
                overlap = min(x_d + w_d, x_s + w_s) - max(x_d, x_s)
                if overlap <= 0:
                    continue
                vert_gap = y_s - (y_d + h_d)
                score = (vert_gap, -overlap)
                if best_score is None or score < best_score:
                    best_score = score
                    chosen_dot_idx = i
                    chosen_dot = dot
        if chosen_dot_idx is not None:
            used_dots.add(chosen_dot_idx)
            x_d,y_d,w_d,h_d = chosen_dot
            x_new = min(x_s, x_d)
            y_new = min(y_s, y_d)
            w_new = max(x_s + w_s, x_d + w_d) - x_new
            h_new = max(y_s + h_s, y_d + h_d) - y_new
            merged.append((x_new, y_new, w_new, h_new))
        else:
            merged.append(stem)
    for i,dot in enumerate(dots):
        if i not in used_dots:
            merged.append(dot)
    merged.sort(key=lambda b: b[0])
    return merged

def group_boxes_to_lines(boxes, min_gap=12):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []
    for box in boxes:
        x,y,w,h = box
        placed = False
        for line in lines:
            tops = [b[1] for b in line]
            bottoms = [b[1] + b[3] for b in line]
            median_top = np.median(tops)
            median_bottom = np.median(bottoms)
            if (y <= median_bottom + min_gap) and (y + h >= median_top - min_gap):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    for line in lines:
        line.sort(key=lambda b: b[0])
    return lines

def adaptive_word_split(line_boxes):
    if not line_boxes:
        return []
    widths = [b[2] for b in line_boxes]
    avg_w = max(np.mean(widths), 1.0)
    gap_thresh = avg_w * 1.5
    words = []
    current = [line_boxes[0]]
    prev_end = line_boxes[0][0] + line_boxes[0][2]
    for b in line_boxes[1:]:
        x,y,w,h = b
        if x - prev_end > gap_thresh:
            words.append(current)
            current = [b]
        else:
            current.append(b)
        prev_end = x + w
    if current:
        words.append(current)
    return words

# --------------------------- UTILS ---------------------------
def image_checksum(np_img):
    small = cv2.resize(np_img, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()

# --------------------------- BACKGROUND WORKER (predictor) ---------------------------
class PredictorWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, stop_event):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            img_rgb, canvas_w, canvas_h, ts = task
            try:
                result = self.process(img_rgb, canvas_w, canvas_h)
                self.result_queue.put((ts, result))
            except Exception as e:
                print("Prediction error:", e)
            self.task_queue.task_done()

    def process(self, img_rgb, canvas_w, canvas_h):
        gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_small, factor = downsample_gray(gray_full, max_width=DOWNSAMPLE_MAX_WIDTH)
        thr = binarize(gray_small)
        boxes = connected_components_boxes(thr)
        boxes_merged = merge_dot_stems(boxes)
        lines = group_boxes_to_lines(boxes_merged)
        scaled_lines = []
        for line in lines:
            scaled_line = []
            for (x,y,w,h) in line:
                x_full = int(round(x / factor))
                y_full = int(round(y / factor))
                w_full = int(round(w / factor))
                h_full = int(round(h / factor))
                x_full = max(0, min(x_full, canvas_w-1))
                y_full = max(0, min(y_full, canvas_h-1))
                w_full = max(1, min(w_full, canvas_w - x_full))
                h_full = max(1, min(h_full, canvas_h - y_full))
                scaled_line.append((x_full, y_full, w_full, h_full))
            scaled_lines.append(scaled_line)

        char_images = []
        char_meta = []
        final_lines_words = []

        for line_idx, line in enumerate(scaled_lines):
            if not line:
                final_lines_words.append([])
                continue
            line2 = merge_dot_stems(line, max_dot_size=16, max_vertical_gap=25)
            line2 = sorted(line2, key=lambda b: b[0])
            words_boxes = adaptive_word_split(line2)
            line_words = []
            for widx, word_boxes in enumerate(words_boxes):
                word_chars = []
                for cidx, (x,y,w,h) in enumerate(word_boxes):
                    crop = gray_full[y:y+h, x:x+w]
                    pil_char = Image.fromarray(crop)
                    arr = preprocess_char_pil(pil_char, name=None)
                    char_images.append(arr)
                    char_meta.append((line_idx, widx, cidx, (x,y,w,h)))
                    word_chars.append(None)
                line_words.append(word_chars)
            final_lines_words.append(line_words)

        recognized = {}
        if char_images:
            batch = np.stack(char_images, axis=0)
            preds = model.predict(batch, verbose=0)
            for i, p in enumerate(preds):
                cls = char_list[np.argmax(p)]
                line_idx, widx, cidx, box = char_meta[i]
                recognized.setdefault((line_idx, widx), []).append((cidx, cls, box))

        final_output_lines = []
        for line_idx, line_words in enumerate(final_lines_words):
            out_words = []
            for widx, chars in enumerate(line_words):
                key = (line_idx, widx)
                if key not in recognized:
                    out_words.append("")
                    continue
                items = recognized[key]
                items_sorted = sorted(items, key=lambda x: x[0])
                chars_str = "".join([c for (_, c, _) in items_sorted])
                out_words.append(chars_str)
            corrected = [dyslexia_aware_correction(w) if w else "" for w in out_words]
            final_output_lines.append((out_words, corrected, line_words))

        overlay_boxes = []
        for (line_idx,widx), items in recognized.items():
            for (cidx, cls, box) in items:
                overlay_boxes.append((line_idx, widx, cidx, cls, box))

        return {"lines": final_output_lines, "overlay": overlay_boxes}

# --------------------------- MAIN APP (Mode A: fixed logical size; scaled display) ---------------------------
class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✍️ Handwriting (Mode A) — Scalable Display (800x400 logical)")
        self.root.geometry("1100x700")
        self.root.minsize(900,600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # logical (fixed) image (internal)
        self.image = Image.new("RGB", (LOGICAL_W, LOGICAL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.history = []
        self.redo_stack = []
        self.last_x = None
        self.last_y = None
        self.brush_width = 5
        self.eraser_width = 20
        self.mode = "draw"

        # background image (beautiful) - generated gradient + subtle noise
        self.bg_image = self._make_background(LOGICAL_W, LOGICAL_H)

        # debounce/checksum/predict worker
        self._last_change_ts = 0
        self._last_pred_ts = 0
        self._last_checksum = None
        self.task_q = queue.Queue()
        self.res_q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = PredictorWorker(self.task_q, self.res_q, self.stop_event)
        self.worker.start()

        # layout
        top = Frame(root)
        top.pack(side=tk.TOP, fill=tk.X)
        main = Frame(root)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side = Frame(root, width=320)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        # toolbar
        Button(top, text="✏️ Draw", command=self.set_draw_mode).pack(side=tk.LEFT, padx=4, pady=4)
        Button(top, text="🩹 Erase", command=self.set_erase_mode).pack(side=tk.LEFT, padx=4, pady=4)
        Button(top, text="🧹 Clear", command=self.clear).pack(side=tk.LEFT, padx=4, pady=4)
        Button(top, text="↩️ Undo", command=self.undo).pack(side=tk.LEFT, padx=4, pady=4)
        Button(top, text="↪️ Redo", command=self.redo).pack(side=tk.LEFT, padx=4, pady=4)
        Button(top, text="💾 Save", command=self.save_canvas).pack(side=tk.LEFT, padx=4, pady=4)
        self.live_btn = Button(top, text="Start Live Predict", bg="lightgray", command=self.toggle_live)
        self.live_btn.pack(side=tk.LEFT, padx=6)
        Button(top, text="🔍 Predict", command=self.trigger_predict_once).pack(side=tk.LEFT, padx=4, pady=4)

        Label(top, text="Brush").pack(side=tk.LEFT, padx=6)
        self.brush_slider = Scale(top, from_=1, to=50, orient=tk.HORIZONTAL, command=self.update_brush, length=120)
        self.brush_slider.set(self.brush_width)
        self.brush_slider.pack(side=tk.LEFT)
        Label(top, text="Eraser").pack(side=tk.LEFT, padx=6)
        self.eraser_slider = Scale(top, from_=10, to=100, orient=tk.HORIZONTAL, command=self.update_eraser, length=120)
        self.eraser_slider.set(self.eraser_width)
        self.eraser_slider.pack(side=tk.LEFT)

        # canvas (display-scaled)
        self.canvas = Canvas(main, bg="white", cursor="pencil")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # keep PhotoImage reference
        self._last_photoimage = None

        # eraser preview (single transient projection rectangle)
        self.eraser_preview = None

        # prediction display
        Label(side, text="Live Prediction (recognized → corrected)", font=("Arial", 11, "bold")).pack(pady=6)
        self.pred_text = tk.Text(side, wrap=tk.WORD, width=40, height=30, font=("Consolas", 11))
        self.pred_text.pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.pred_scroll = Scrollbar(side, command=self.pred_text.yview)
        self.pred_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pred_text.config(yscrollcommand=self.pred_scroll.set)

        # schedule result checks
        self.root.after(100, self._check_results)

        # initial render
        self._redraw_canvas_from_image()

    # ---- background creation ----
    def _make_background(self, w, h):
        # nice gradient + subtle texture
        bg = Image.new("RGB", (w,h), "#f7f3ee")
        grad = Image.new("L", (1,h))
        for y in range(h):
            grad.putpixel((0,y), int(255 * (0.95 - 0.15 * (y/h))))
        grad = grad.resize((w,h))
        bg = Image.composite(Image.new("RGB",(w,h), "#fef6e7"), Image.new("RGB",(w,h), "#e8f0ff"), grad)
        # add subtle noise
        noise = Image.effect_noise((w,h), 20).convert("L").filter(ImageFilter.GaussianBlur(1))
        noise_rgb = ImageOps.colorize(noise, black="#000000", white="#ffffff")
        bg = Image.blend(bg, noise_rgb, 0.02)
        return bg

    # ---- drawing handlers (map display coords -> logical coords) ----
    def _display_to_logical(self, x_disp, y_disp):
        c_w = max(1, self.canvas.winfo_width())
        c_h = max(1, self.canvas.winfo_height())
        scale_x = LOGICAL_W / c_w
        scale_y = LOGICAL_H / c_h
        x_log = int(round(x_disp * scale_x))
        y_log = int(round(y_disp * scale_y))
        # clamp
        x_log = max(0, min(LOGICAL_W-1, x_log))
        y_log = max(0, min(LOGICAL_H-1, y_log))
        return x_log, y_log

    def on_press(self, event):
        self.last_x, self.last_y = event.x, event.y
        # create eraser preview if eraser mode
        if self.mode == "erase":
            if self.eraser_preview is None:
                self.eraser_preview = self.canvas.create_rectangle(0,0,0,0, outline="gray", dash=(4,4), tags=('eraser_preview',))
            self._move_eraser_preview(event.x, event.y)

    def on_drag(self, event):
        if self.last_x is None:
            self.last_x, self.last_y = event.x, event.y
        color = "black" if self.mode == "draw" else "white"
        width = self.brush_width if self.mode == "draw" else self.eraser_width
        # draw on display canvas for immediate feedback
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill=color, width=width, capstyle=tk.ROUND, smooth=True, tags=('stroke',))
        # map to logical coords and draw on PIL logical image
        x1_log, y1_log = self._display_to_logical(self.last_x, self.last_y)
        x2_log, y2_log = self._display_to_logical(event.x, event.y)
        draw_width = max(1, int(round(width * (LOGICAL_W / max(1, self.canvas.winfo_width())))))
        self.draw.line([x1_log, y1_log, x2_log, y2_log], fill=color, width=draw_width)
        self.history.append((x1_log, y1_log, x2_log, y2_log, color, draw_width))
        self.last_x, self.last_y = event.x, event.y
        # update eraser preview if in erase mode
        if self.mode == "erase":
            self._move_eraser_preview(event.x, event.y)
        self._mark_changed()

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        # remove eraser preview if present (keep only until pointer up)
        if self.eraser_preview is not None:
            self.canvas.delete(self.eraser_preview)
            self.eraser_preview = None
        self._mark_changed(immediate=True)
        # after release, re-render display from logical PIL (clear 'stroke' tags then draw full scaled)
        self._redraw_canvas_from_image()

    def _move_eraser_preview(self, x_disp, y_disp):
        size_disp_x = int(round(self.eraser_width * (self.canvas.winfo_width() / LOGICAL_W)))
        size_disp_y = int(round(self.eraser_width * (self.canvas.winfo_height() / LOGICAL_H)))
        x1, y1 = x_disp - size_disp_x, y_disp - size_disp_y
        x2, y2 = x_disp + size_disp_x, y_disp + size_disp_y
        self.canvas.coords(self.eraser_preview, x1, y1, x2, y2)
        self.canvas.tag_raise(self.eraser_preview)

    def set_draw_mode(self):
        self.mode = "draw"
        self.canvas.config(cursor="pencil")

    def set_erase_mode(self):
        self.mode = "erase"
        self.canvas.config(cursor="dotbox")

    def update_brush(self, v):
        self.brush_width = int(v)

    def update_eraser(self, v):
        self.eraser_width = int(v)

    def undo(self):
        if not self.history:
            return
        self.redo_stack.append(self.history.pop())
        self._rebuild_image_from_history()
        self._mark_changed()
        self._redraw_canvas_from_image()

    def redo(self):
        if not self.redo_stack:
            return
        self.history.append(self.redo_stack.pop())
        self._rebuild_image_from_history()
        self._mark_changed()
        self._redraw_canvas_from_image()

    def _rebuild_image_from_history(self):
        self.image = Image.new("RGB", (LOGICAL_W, LOGICAL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        for (x1,y1,x2,y2,color,width) in self.history:
            self.draw.line([x1,y1,x2,y2], fill=color, width=width)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (LOGICAL_W, LOGICAL_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.history.clear()
        self.redo_stack.clear()
        self.pred_text.delete("1.0", tk.END)
        self._mark_changed()

    def save_canvas(self):
        # composite background + drawing logical -> upscale to save as actual image
        combo = Image.alpha_composite(self.bg_image.convert("RGBA"), self.image.convert("RGBA"))
        file = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files","*.png")])
        if file:
            combo.save(file)
            messagebox.showinfo("Saved", f"Canvas saved as {file}")

    def on_canvas_resize(self, event):
        # display changed; simply redraw scaled version (do NOT resize logical image)
        self._redraw_canvas_from_image()

    def _pil_display_image(self):
        """Return PIL image for display: composite background + current drawing."""
        combo = Image.alpha_composite(self.bg_image.convert("RGBA"), self.image.convert("RGBA"))
        return combo

    def _redraw_canvas_from_image(self):
        # Clear temporary strokes (tag 'stroke') and overlays (tag 'overlay') before drawing
        try:
            self.canvas.delete('stroke')
        except Exception:
            pass
        # Render composite image scaled to current canvas size
        display_pil = self._pil_display_image()
        c_w = max(1, self.canvas.winfo_width())
        c_h = max(1, self.canvas.winfo_height())
        disp = display_pil.resize((c_w, c_h), Image.Resampling.LANCZOS)
        tkimg = ImageTk.PhotoImage(disp)
        self._last_photoimage = tkimg
        # clear background (keep overlays separately with tag 'overlay')
        self.canvas.delete('bgimg')
        self.canvas.create_image(0, 0, anchor="nw", image=tkimg, tags=('bgimg',))
        # ensure overlay drawn on top
        self.canvas.tag_raise('overlay')

    # ----- live prediction control -----
    def toggle_live(self):
        self.live_btn.config(state=tk.DISABLED)
        if getattr(self, "live_on", False):
            self.live_on = False
            self.live_btn.config(text="Start Live Predict", bg="lightgray", state=tk.NORMAL)
        else:
            self.live_on = True
            self.live_btn.config(text="Stop Live Predict", bg="#6bd36b", state=tk.NORMAL)
            self._mark_changed(immediate=True)

    def trigger_predict_once(self):
        self._enqueue_current_image()

    def _mark_changed(self, immediate=False):
        self._last_change_ts = time.time()
        np_rgb = np.array(self.image)
        checksum = image_checksum(np_rgb)
        if checksum == self._last_checksum and not immediate:
            return
        self._last_checksum = checksum
        if immediate:
            self._enqueue_current_image()
            return
        try:
            if hasattr(self, "_debounce_after_id"):
                self.root.after_cancel(self._debounce_after_id)
        except Exception:
            pass
        self._debounce_after_id = self.root.after(LIVE_POLL_INTERVAL_MS, self._debounced_enqueue)

    def _debounced_enqueue(self):
        now = time.time()
        if now - self._last_pred_ts < (1.0 / MAX_FPS):
            self._debounce_after_id = self.root.after(int(1000.0 / MAX_FPS), self._debounced_enqueue)
            return
        if getattr(self, "live_on", False) or True:
            self._enqueue_current_image()
            self._last_pred_ts = time.time()

    def _enqueue_current_image(self):
        # send logical image (800x400) as numpy rgb to worker
        np_rgb = np.array(self.image.copy())
        try:
            self.task_q.put_nowait((np_rgb, LOGICAL_W, LOGICAL_H, time.time()))
        except queue.Full:
            pass

    # ----- check results and update overlay/UI -----
    def _check_results(self):
        while True:
            try:
                ts, result = self.res_q.get_nowait()
            except queue.Empty:
                break
            self._apply_result_to_ui(result)
        self.root.after(100, self._check_results)

    def _apply_result_to_ui(self, result):
        # clear old overlay (tag 'overlay')
        try:
            self.canvas.delete('overlay')
        except Exception:
            pass
        overlay = result.get("overlay", [])
        colors = ["#FF0000", "#00AA00", "#0000FF", "#AA00AA", "#FF8800"]
        c_w = max(1, self.canvas.winfo_width())
        c_h = max(1, self.canvas.winfo_height())
        scale_x = c_w / LOGICAL_W
        scale_y = c_h / LOGICAL_H
        for (line_idx,widx,cidx,cls,box) in overlay:
            x,y,w,h = box
            x1 = int(round(x * scale_x))
            y1 = int(round(y * scale_y))
            x2 = int(round((x + w) * scale_x))
            y2 = int(round((y + h) * scale_y))
            color = colors[line_idx % len(colors)]
            # tag = 'overlay' to allow clearing without affecting user strokes
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=('overlay',))
            self.canvas.create_text(x1+3, y1+8, anchor="nw", text=cls, font=("Arial",9,"bold"), fill=color, tags=('overlay',))
        # update prediction text area
        lines = result.get("lines", [])
        text_lines = []
        for (out_words, corrected, _) in lines:
            recognized = " ".join(out_words)
            corrected_line = " ".join(corrected)
            text_lines.append(f"Recognized: {recognized}")
            text_lines.append(f"Corrected : {corrected_line}")
            text_lines.append("")
        full_text = "\n".join(text_lines)
        self.pred_text.delete("1.0", tk.END)
        self.pred_text.insert(tk.END, full_text)

    def on_close(self):
        self.stop_event.set()
        timeout = time.time() + 1.0
        while self.worker.is_alive() and time.time() < timeout:
            time.sleep(0.05)
        try:
            self.root.destroy()
        except Exception:
            pass

# --------------------------- run ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()

