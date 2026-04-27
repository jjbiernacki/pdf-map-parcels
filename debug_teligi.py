"""Render Teligi."""
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from analyze_cv import load_ocr_cache, extract_paths
from analyze_hybrid import detect_road_corridor_polygon

PDF = "Mapy/PZT Teligi-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Teligi-Model.pkl"
doc = fitz.open(PDF); page = doc[0]
print(f"Page: {page.rect}")
scale = 2
mat = fitz.Matrix(scale, scale)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img)

greens, reds_trace, red_pins = extract_paths(page)
corridor = detect_road_corridor_polygon(reds_trace, greens)
if corridor:
    coords = list(corridor.exterior.coords)
    for p1, p2 in zip(coords[:-1], coords[1:]):
        draw.line([(p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale)],
                  fill="magenta", width=4)

labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
for l in labels:
    if l.text == "428":
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-10, y-10, x+10, y+10], outline="blue", width=3)
        draw.text((x+12, y), f"GT:{l.text}", fill="blue")
    elif l.text in ("427", "426"):
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-10, y-10, x+10, y+10], outline="red", width=3)
        draw.text((x+12, y), f"{l.text}", fill="red")

img.save("/tmp/teligi.png")
print(f"saved /tmp/teligi.png size={img.size}")
