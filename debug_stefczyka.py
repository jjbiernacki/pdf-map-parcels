"""Render Stefczyka with GT + detected corridor."""
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from shapely.geometry import Point
from analyze_cv import load_ocr_cache
from analyze_hybrid import detect_road_corridor_polygon
from analyze_cv import extract_paths

PDF = "Mapy/PZT Stefczyka-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Stefczyka-Model.pkl"
doc = fitz.open(PDF); page = doc[0]

scale = 2
mat = fitz.Matrix(scale, scale)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img)

greens, reds_trace, red_pins = extract_paths(page)
corridor = detect_road_corridor_polygon(reds_trace, greens,
                                         min_line_length_pt=30.0,
                                         max_line_dist_to_route_pt=120.0)
if corridor:
    coords = list(corridor.exterior.coords)
    for p1, p2 in zip(coords[:-1], coords[1:]):
        draw.line([(p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale)],
                  fill="magenta", width=5)

labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
for l in labels:
    if l.text == "98/86":
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-12, y-12, x+12, y+12], outline="blue", width=4)
        draw.text((x+14, y), f"GT:{l.text}", fill="blue")
    elif l.text == "170":
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-12, y-12, x+12, y+12], outline="red", width=4)
        draw.text((x+14, y), f"FP:{l.text}", fill="red")

img.save("/tmp/stefczyka.png")
print(f"saved /tmp/stefczyka.png size={img.size}")
