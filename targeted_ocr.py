"""Targeted OCR for missing labels — render high-resolution crop around expected
position and try multiple OCR configurations.
"""
import fitz
import numpy as np
from analyze import _render_green_for_ocr
import easyocr
import re

PARCEL_RE = re.compile(r"^\d{2,3}(/\d{1,2})?$")

doc = fitz.open("Mapy/PZT Grochowska-Model.pdf")
page = doc[0]
proc = _render_green_for_ocr(page, 12)  # higher scale 12
H, W = proc.shape
print(f"render shape: {proc.shape}")

# Find region around 421 expected position
# In PDF coords: ~(1050-1180, 250-400)
# At scale 12 → (12600-14160, 3000-4800)
# Crop with margin
x0_pt, y0_pt = 1000, 200
x1_pt, y1_pt = 1250, 450
x0, y0 = x0_pt * 12, y0_pt * 12
x1, y1 = x1_pt * 12, y1_pt * 12
crop = proc[y0:y1, x0:x1]
print(f"crop shape: {crop.shape}")

# Save crop
from PIL import Image
Image.fromarray(crop).save("/tmp/grochowska_421_crop.png")
print(f"saved crop")

# OCR with very low thresholds
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
rgb = np.stack([crop, crop, crop], axis=-1)

print("\nOCR results (text_threshold=0.05, low_text=0.02):")
hits = reader.readtext(rgb, allowlist="0123456789/", min_size=2,
                        paragraph=False, text_threshold=0.05, low_text=0.02,
                        link_threshold=0.4)
for bbox, txt, conf in hits:
    px = (bbox[0][0] + x0) / 12
    py = (bbox[0][1] + y0) / 12
    if conf >= 0.1:
        print(f"  {txt!r:>10s}  conf={conf:.3f}  pdf_pos=({px:.0f}, {py:.0f})")
