"""Render each new map with GT, FP, FN highlights and corridor overlay."""
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from shapely.geometry import Point
from analyze_cv import load_ocr_cache, extract_paths
from analyze_hybrid import detect_road_corridor_polygon, analyze


TARGETS = {
    "PZT Grochowska-Model": {
        "gt": {"421", "430", "440", "439/3", "448", "465", "479/4", "479/2",
               "429/2", "502", "523", "522", "568", "581", "592", "594/1",
               "605/2", "614/1"},
    },
    "PZT Kurka-Model": {
        "gt": {"443", "465", "456/12", "456/11", "456/1", "456/2",
               "456/4", "456/5"},
    },
    "PZT Polna-Model": {
        "gt": {"536", "550/6", "547/5", "547/10", "550/3", "550/4",
               "550/5", "558/2", "557"},
    },
}

import sys
name = sys.argv[1] if len(sys.argv) > 1 else "PZT Polna-Model"
spec = TARGETS[name]
gt = spec["gt"]

pdf_path = f"Mapy/{name}.pdf"
cache = f"/tmp/mapy_ocr/{name}.pkl"
doc = fitz.open(pdf_path); page = doc[0]
print(f"Page: {page.rect}")

# Run algorithm
res = analyze(pdf_path, ocr_cache=cache)
got = set(res.crossed)
fp = got - gt
fn = gt - got
print(f"GT={sorted(gt)}")
print(f"Crossed={sorted(got)}")
print(f"FP={sorted(fp)}  FN={sorted(fn)}")

# Render
scale = 2
pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img)

greens, reds, _ = extract_paths(page)
corridor = detect_road_corridor_polygon(reds, greens)
if corridor:
    coords = list(corridor.exterior.coords)
    for p1, p2 in zip(coords[:-1], coords[1:]):
        draw.line([(p1[0]*scale, p1[1]*scale), (p2[0]*scale, p2[1]*scale)],
                  fill="magenta", width=4)

labels = load_ocr_cache(Path(cache), ocr_scale=8)
for l in labels:
    if l.text in gt:
        color = "blue" if l.text in got else "darkblue"
        tag = "GT" if l.text in got else "FN"
    elif l.text in fp:
        color = "red"
        tag = "FP"
    else:
        continue
    x, y = l.x*scale, l.y*scale
    draw.ellipse([x-10, y-10, x+10, y+10], outline=color, width=3)
    draw.text((x+12, y), f"{tag}:{l.text}", fill=color)

out_path = f"/tmp/{name.lower().replace(' ', '_').replace('/','_')}.png"
img.save(out_path)
print(f"saved {out_path} size={img.size}")
