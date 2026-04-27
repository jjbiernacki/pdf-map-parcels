"""Debug Laurowa."""
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from shapely.geometry import Point
from analyze_cv import load_ocr_cache, extract_paths
from analyze_hybrid import detect_road_corridor_polygon

PDF = "Mapy/PZT Laurowa-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Laurowa-Model.pkl"
doc = fitz.open(PDF); page = doc[0]
scale = 2
pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img)

greens, reds, _ = extract_paths(doc[0])
corridor = detect_road_corridor_polygon(reds, greens)
if corridor:
    print(f"corridor bounds: {corridor.bounds}")
    coords = list(corridor.exterior.coords)
    for p1, p2 in zip(coords[:-1], coords[1:]):
        draw.line([(p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale)],
                  fill="magenta", width=4)

labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
for l in labels:
    if l.text in ('75', '260'):
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-10, y-10, x+10, y+10], outline="blue", width=3)
        draw.text((x+12, y), f"GT:{l.text}", fill="blue")
        if corridor:
            print(f"{l.text} at ({l.x:.1f}, {l.y:.1f}): inside={corridor.contains(Point(l.x, l.y))}")
    elif l.text == '15':
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-10, y-10, x+10, y+10], outline="red", width=3)
        draw.text((x+12, y), f"FP:{l.text}", fill="red")
        if corridor:
            print(f"{l.text} at ({l.x:.1f}, {l.y:.1f}): inside={corridor.contains(Point(l.x, l.y))}")

img.save("/tmp/laurowa.png")
print(f"saved /tmp/laurowa.png")
