"""Render Agatowa with GT labels and detected corridor."""
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from analyze_cv import load_ocr_cache, extract_paths
from analyze_hybrid import detect_road_corridor_polygon

PDF = "Mapy/PZT Agatowa-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Agatowa-Model.pkl"
doc = fitz.open(PDF); page = doc[0]
print(f"Page: {page.rect}")
scale = 2
mat = fitz.Matrix(scale, scale)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img)

greens, reds_trace, red_pins = extract_paths(page)

# Route bounds
from analyze_cv import iter_segments
red_segs = [(p1,p2) for d in reds_trace for p1,p2 in iter_segments(d) if p1!=p2]
xs = [x for (p1,p2) in red_segs for x in [p1[0],p2[0]]]
ys = [y for (p1,p2) in red_segs for y in [p1[1],p2[1]]]
print(f"Route bounds: x=[{min(xs):.0f}, {max(xs):.0f}], y=[{min(ys):.0f}, {max(ys):.0f}]")

corridor = detect_road_corridor_polygon(reds_trace, greens,
                                         min_line_length_pt=30.0,
                                         max_line_dist_to_route_pt=120.0)
if corridor:
    print(f"Corridor: bounds={corridor.bounds} area={corridor.area:.0f}")
    coords = list(corridor.exterior.coords)
    for p1, p2 in zip(coords[:-1], coords[1:]):
        draw.line([(p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale)],
                  fill="magenta", width=4)

labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
gt = {'180/7', '180/24', '236/43', '180/18'}
fp = {'261/3'}
for l in labels:
    if l.text in gt:
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-10, y-10, x+10, y+10], outline="blue", width=3)
        draw.text((x+12, y), f"{l.text}", fill="blue")
    elif l.text in fp:
        x, y = l.x*scale, l.y*scale
        draw.ellipse([x-10, y-10, x+10, y+10], outline="red", width=3)
        draw.text((x+12, y), f"FP:{l.text}", fill="red")

img.save("/tmp/agatowa.png")
print(f"saved /tmp/agatowa.png size={img.size}")
