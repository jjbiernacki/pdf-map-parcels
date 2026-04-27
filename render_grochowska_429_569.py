"""Render zooms of Grochowska 429/2 (TP wanted) vs 569 (FP)."""
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

doc = fitz.open('Mapy/PZT Grochowska-Model.pdf'); page = doc[0]
greens, reds, _ = extract_paths(page)
labels = load_ocr_cache(Path('/tmp/mapy_ocr/PZT Grochowska-Model.pkl'), ocr_scale=8)
scale_p = 4.0
ctx = RasterCtx(scale=scale_p, width_px=int(round(page.rect.width * scale_p)),
                height_px=int(round(page.rect.height * scale_p)))
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

SCALE = 6
mat = fitz.Matrix(SCALE, SCALE)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

for n, color, fill in [('429/2', 'blue', (0,0,255,80)), ('569', 'red', (255,0,0,80)),
                         ('501', 'orange', (255,165,0,80)), ('420', 'purple', (160,0,200,80))]:
    draw = ImageDraw.Draw(img, "RGBA")
    for l in labels:
        if l.text == n:
            for p in polys:
                if p.contains(Point(l.x, l.y)):
                    coords = list(p.exterior.coords)
                    pts = [(c[0]*SCALE, c[1]*SCALE) for c in coords]
                    draw.polygon(pts, fill=fill, outline=color, width=4)
                    break
            x, y = l.x*SCALE, l.y*SCALE
            r = 12
            draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=4)
            draw.text((x+r+3, y), n, fill=color)
            break

# Two crops: 429/2 area (2701, 282) and 569 area (4534, 1143)
for tag, cx, cy in [('429', 2750, 350), ('569', 4500, 1150), ('501', 3200, 700), ('420', 700, 320)]:
    pad = 200
    crop = (max(0, int((cx-pad)*SCALE)), max(0, int((cy-pad)*SCALE)),
            min(pix.width, int((cx+pad)*SCALE)), min(pix.height, int((cy+pad)*SCALE)))
    img_c = img.crop(crop)
    img_c.save(f"/tmp/grochowska_{tag}.png")
    print(f"saved /tmp/grochowska_{tag}.png size={img_c.size}")
