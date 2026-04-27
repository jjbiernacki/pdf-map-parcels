"""Render Kurka with polygons containing 456/11 and 456/3 highlighted."""
import fitz
from PIL import Image, ImageDraw
from pathlib import Path
from shapely.geometry import Point
from analyze_cv import extract_paths, load_ocr_cache
from analyze_hybrid import (build_polygons, build_red_union, green_segments_vec,
    find_endpoint_closures, find_endpoint_extensions,
    find_parallel_endpoint_closures, find_tjunction_extensions,
    pin_crossline_segments, route_end_closures, route_buffer_frame,
    frame_segments, RasterCtx)

doc = fitz.open('Mapy/PZT Kurka-Model.pdf'); page = doc[0]
greens, reds, _ = extract_paths(page)
labels = load_ocr_cache(Path('/tmp/mapy_ocr/PZT Kurka-Model.pkl'), ocr_scale=8)
scale = 4.0
ctx = RasterCtx(scale=scale, width_px=int(round(page.rect.width * scale)),
                height_px=int(round(page.rect.height * scale)))
segs = green_segments_vec(greens)
segs.extend(find_endpoint_closures(greens, ctx, line_thickness=2, max_gap_pt=5.0))
segs.extend(find_endpoint_extensions(greens, ctx, line_thickness=2, max_extend_pt=80.0))
par = find_parallel_endpoint_closures(greens, ctx, line_thickness=2, max_gap_pt=120.0, cluster_r_pt=15.0)
segs.extend(par)
segs.extend(find_endpoint_extensions(greens, ctx, line_thickness=2, max_extend_pt=80.0, extra_segments=par))
segs.extend(find_tjunction_extensions(greens, ctx, line_thickness=2, max_extend_pt=40.0))
segs.extend(pin_crossline_segments([], reds, half_length_pt=25.0))
segs.extend(route_end_closures(reds, half_length_pt=250.0))
segs.extend(route_buffer_frame(reds, buffer_pt=100.0))
segs.extend(frame_segments(page.rect))
polys = build_polygons(segs, snap_tol=3.0)
polys = [p for p in polys if p.area <= 1e6]

# Render
SCALE = 5
mat = fitz.Matrix(SCALE, SCALE)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
draw = ImageDraw.Draw(img, "RGBA")

# Find polygon for 456/11 and 456/3
target_labels = {
    '456/11': ('blue', (0, 0, 255, 60)),  # GT
    '456/3': ('red', (255, 0, 0, 60)),    # FP
    '456/4': ('green', (0, 200, 0, 60)),  # GT
    '456/5': ('purple', (160, 0, 200, 60)),  # GT
}
for n, (outline_color, fill_color) in target_labels.items():
    for l in labels:
        if l.text == n:
            pt = Point(l.x, l.y)
            for p in polys:
                if p.contains(pt):
                    coords = list(p.exterior.coords)
                    pts = [(c[0]*SCALE, c[1]*SCALE) for c in coords]
                    draw.polygon(pts, fill=fill_color, outline=outline_color, width=4)
                    break
            x, y = l.x*SCALE, l.y*SCALE
            r = 15
            draw.ellipse([x-r, y-r, x+r, y+r], outline=outline_color, width=4)
            draw.text((x+r+3, y), n, fill=outline_color)
            break

# crop to relevant area
crop = (450*SCALE, 600*SCALE, 850*SCALE, 1100*SCALE)
img_c = img.crop(crop)
img_c.save("/tmp/kurka_polys.png")
print(f"saved /tmp/kurka_polys.png size={img_c.size}")
