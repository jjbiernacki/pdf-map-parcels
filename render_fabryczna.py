"""Render Fabryczna with GT labels + parallel road strip boundaries annotated."""
import math
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from shapely.geometry import LineString, Point

from analyze_cv import extract_paths, iter_segments, load_ocr_cache

PDF = "Mapy/PZT Fabryczna-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Fabryczna-Model.pkl"

doc = fitz.open(PDF)
page = doc[0]
greens, reds_trace, red_pins = extract_paths(page)

# Get green segs near route (angle ~160°, dist<100pt)
red_segs = [(p1,p2) for d in reds_trace for p1,p2 in iter_segments(d) if p1!=p2]
route_cx = sum(x for (p1,p2) in red_segs for x in [p1[0],p2[0]]) / (2*len(red_segs))
route_cy = sum(y for (p1,p2) in red_segs for y in [p1[1],p2[1]]) / (2*len(red_segs))
print(f"route centroid: ({route_cx:.1f}, {route_cy:.1f})")

scale = 3
mat = fitz.Matrix(scale, scale)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img)

# Render long green segments near route, by angle
green_segs = [(p1,p2) for d in greens for p1,p2 in iter_segments(d) if p1!=p2]
for (p1, p2) in green_segs:
    ls = LineString([p1,p2])
    if ls.length < 30:
        continue
    d = ls.distance(Point(route_cx, route_cy))
    if d > 150:
        continue
    a = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    if a < 0: a += 180
    # draw with color by angle bucket
    if 155 <= a <= 175:
        color = "cyan"  # parallel road candidates
    else:
        color = "orange"
    draw.line([(p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale)],
              fill=color, width=3)

# GT labels
labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
lbl_by_text = {l.text: l for l in labels}
for name in ['151/35', '470', '423']:
    l = lbl_by_text[name]
    x, y = l.x * scale, l.y * scale
    r = 12
    draw.ellipse([x-r, y-r, x+r, y+r], outline="blue", width=3)
    draw.text((x+r+2, y-r-2), f"{name}", fill="blue")

# draw red centroid
draw.ellipse([route_cx*scale-8, route_cy*scale-8, route_cx*scale+8, route_cy*scale+8],
             outline="red", width=3)

# Crop
pad = 250
crop = (max(0, int((route_cx-pad)*scale)),
        max(0, int((route_cy-pad)*scale)),
        min(pix.width, int((route_cx+pad)*scale)),
        min(pix.height, int((route_cy+pad)*scale)))
img_c = img.crop(crop)
img_c.save("/tmp/fabryczna_parallels.png")
print(f"Saved /tmp/fabryczna_parallels.png size={img_c.size}")
