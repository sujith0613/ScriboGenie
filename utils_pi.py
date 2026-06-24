import numpy as np
from PIL import Image, ImageGrab
import os
import subprocess

def preprocess_canvas(canvas, temp_ps="temp.ps"):
    """
    Preprocesses the Tkinter canvas content for CNN inference.
    Laptop-compatible version.
    """
    try:
        if os.name == 'nt':
            # On Windows Laptop: Use ImageGrab for a clean screenshot of the canvas
            x = canvas.winfo_rootx()
            y = canvas.winfo_rooty()
            x1 = x + canvas.winfo_width()
            y1 = y + canvas.winfo_height()
            img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
        else:
            # On Raspberry Pi: Use Postscript + Ghostscript (more reliable on Linux)
            canvas.postscript(file=temp_ps, colormode="color")
            img = Image.open(temp_ps).convert("L")
        
        # Standard Resize to 28x28
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        
        # Invert (Model expects 1.0 for ink)
        img_array = 1.0 - img_array
        return img_array.reshape(1, 28, 28, 1)
        
    except Exception as e:
        # print(f"⚠️ Canvas preprocessing error: {e}")
        return None
    finally:
        if os.path.exists(temp_ps):
            try: os.remove(temp_ps)
            except: pass
