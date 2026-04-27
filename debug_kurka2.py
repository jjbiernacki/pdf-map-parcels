"""Detailed debug of Kurka FN — why 456/11, 456/4, 456/5 not classified?"""
import fitz
from pathlib import Path
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

from analyze_cv import extract_paths, iter_segments, load_ocr_cache
from analyze_hybrid import (
    build_polygons, build_red_union, green_segments_vec,
    find_endpoint_closures, find_endpoint_extensions,
    find_parallel_endpoint_closures, find_tjunction_extensions,
    pin_crossline_segments, route_end_closures, route_buffer_frame,
    frame_segments, RasterCtx,
)
from analyze_ray import (
    build_green_segments, build_red_union as rb_red,
    compute_crossed_greens, label_is_crossed,
)

PDF = "Mapy/PZT Kurka-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Kurka-Model.pkl"
doc = fitz.open(PDF); page = doc[0]
greens, reds, _ = extract_paths(page)
labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
lbl_by_text = {l.text: l for l in labels}

scale = 4.0
ctx = RasterCtx(scale=scale, width_px=int(round(page.rect.width * scale)),
                height_px=int(round(page.rect.height * scale)))
segs = green_segments_vec(greens)
segs.extend(find_endpoint_closures(greens, ctx, line_thickness=2, max_gap_pt=5.0))
segs.extend(find_endpoint_extensions(greens, ctx, line_thickness=2, max_extend_pt=80.0))
par_segs = find_parallel_endpoint_closures(greens, ctx, line_thickness=2,
                                           max_gap_pt=120.0, cluster_r_pt=15.0)
segs.extend(par_segs)
segs.extend(find_endpoint_extensions(greens, ctx, line_thickness=2,
                                      max_extend_pt=80.0, extra_segments=par_segs))
segs.extend(find_tjunction_extensions(greens, ctx, line_thickness=2, max_extend_pt=40.0))
segs.extend(pin_crossline_segments([], reds, half_length_pt=25.0))
segs.extend(route_end_closures(reds, half_length_pt=250.0))
segs.extend(route_buffer_frame(reds, buffer_pt=100.0))
segs.extend(frame_segments(page.rect))

polys = build_polygons(segs, snap_tol=3.0)
polys = [p for p in polys if p.area <= 1e6]
print(f"Total polygons: {len(polys)}")

red_union = build_red_union(reds)
red_buf2 = red_union.buffer(2, cap_style=2, join_style=2)
red_buf5 = red_union.buffer(5, cap_style=2, join_style=2)

# For each FN label, find its polygon and check classification
for name in ["456/11", "456/4", "456/5", "456/1", "456/2", "456/3", "456/12", "456/8", "456/9", "456/10"]:
    if name not in lbl_by_text:
        print(f"\n{name} NOT IN OCR")
        continue
    l = lbl_by_text[name]
    pt = Point(l.x, l.y)
    print(f"\n=== {name} at ({l.x:.1f}, {l.y:.1f}) ===")
    found_i = None
    for i, p in enumerate(polys):
        if p.contains(pt):
            found_i = i
            break
    if found_i is None:
        d = red_union.distance(pt)
        print(f"  NOT IN ANY poly. d_route={d:.1f}")
        continue
    p = polys[found_i]
    print(f"  poly#{found_i} area={p.area:.1f} bounds={p.bounds}")
    # Boundary intersection with red
    bint = p.boundary.intersection(red_union)
    bint_len = bint.length if hasattr(bint, "length") else 0
    print(f"  boundary∩red len={bint_len:.2f}")
    # Boundary intersection with red_buf
    bint2 = p.boundary.intersection(red_buf2)
    bint2_len = bint2.length if hasattr(bint2, "length") else 0
    bint5 = p.boundary.intersection(red_buf5)
    bint5_len = bint5.length if hasattr(bint5, "length") else 0
    print(f"  boundary∩red_buf2 len={bint2_len:.2f}, ∩red_buf5 len={bint5_len:.2f}")
    # Distance from poly boundary to route
    d_b = p.boundary.distance(red_union)
    print(f"  d(boundary, route) = {d_b:.2f}pt")
    # Distance from poly to route
    d_p = p.distance(red_union)
    print(f"  d(poly, route) = {d_p:.2f}pt")
