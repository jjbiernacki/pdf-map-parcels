"""Debug Fabryczna — po dodaniu road_corridor_closures, sprawdź polygony
zawierające GT labels."""
import fitz
from pathlib import Path
from shapely.geometry import Point

from analyze_cv import extract_paths, iter_segments, load_ocr_cache
from analyze_hybrid import (
    build_polygons, build_red_union, green_segments_vec,
    find_endpoint_closures, find_endpoint_extensions,
    find_parallel_endpoint_closures, find_tjunction_extensions,
    pin_crossline_segments, route_end_closures, route_buffer_frame,
    road_corridor_closures,
    frame_segments, RasterCtx,
)

PDF = "Mapy/PZT Fabryczna-Model.pdf"
CACHE = "/tmp/mapy_ocr/PZT Fabryczna-Model.pkl"
doc = fitz.open(PDF); page = doc[0]
greens, reds_trace, red_pins = extract_paths(page)
labels = load_ocr_cache(Path(CACHE), ocr_scale=8)
lbl_by_text = {l.text: l for l in labels}

scale = 4.0
ctx = RasterCtx(scale=scale,
                width_px=int(round(page.rect.width * scale)),
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
segs.extend(pin_crossline_segments(red_pins, reds_trace, half_length_pt=25.0))
segs.extend(route_end_closures(reds_trace, half_length_pt=250.0))

corr = road_corridor_closures(greens, reds_trace, page.rect,
                              min_segment_pt=30.0, max_perp_dist_pt=150.0,
                              angle_tol_deg=35.0)
print(f"road corridor closures: {len(corr)}")
for c in corr:
    print(f"  {c}")
segs.extend(corr)

segs.extend(route_buffer_frame(reds_trace, buffer_pt=100.0))
segs.extend(frame_segments(page.rect))
polys = build_polygons(segs, snap_tol=3.0)
polys = [p for p in polys if p.area <= 1e6]
print(f"\nTotal polygons: {len(polys)}")

red_union = build_red_union(reds_trace)

for name in ["151/35", "470", "423"]:
    l = lbl_by_text[name]
    pt = Point(l.x, l.y)
    print(f"\n=== {name} at ({l.x:.1f}, {l.y:.1f}) ===")
    for i, p in enumerate(polys):
        if p.contains(pt):
            binter = p.boundary.intersection(red_union)
            # bint length
            bint_len = 0
            if binter.is_empty:
                pass
            elif binter.geom_type == "LineString":
                bint_len = binter.length
            elif hasattr(binter, "geoms"):
                for g in binter.geoms:
                    if hasattr(g, "length"):
                        bint_len += g.length
            interior = p.buffer(-0.5)
            inter_len = 0
            if not interior.is_empty:
                inter = interior.intersection(red_union)
                inter_len = inter.length if hasattr(inter, "length") else 0
            n_components = len(list(binter.geoms)) if hasattr(binter, "geoms") else (1 if not binter.is_empty else 0)
            print(f"  poly#{i} area={p.area:.1f} bounds={p.bounds}")
            print(f"    boundary∩red type={binter.geom_type} n_comp={n_components} len={bint_len:.2f}")
            print(f"    interior(-0.5)∩red len={inter_len:.2f}")
            break
    else:
        print(f"  NOT IN ANY poly after area filter")
print(f"\nred_union bounds: {red_union.bounds}, length={red_union.length:.1f}")
rc = red_union.centroid
print(f"\nred centroid: ({rc.x:.1f}, {rc.y:.1f})")

# Polygons containing the route centroid or with route.length inside them
print("\nPolygons containing route centroid:")
for i, p in enumerate(polys):
    if p.contains(rc):
        print(f"  poly#{i} area={p.area:.1f} bounds={p.bounds}")
print("\nPolygons where route's interior-length >= 0.5:")
for i, p in enumerate(polys):
    interior = p.buffer(-0.5)
    if interior.is_empty:
        continue
    inter = interior.intersection(red_union)
    il = inter.length if hasattr(inter, "length") else 0
    if il >= 0.5:
        print(f"  poly#{i} area={p.area:.1f} bounds={p.bounds} red∩interior={il:.2f}")

# Polygons whose boundary has shared line with red
print("\nPolygons with boundary∩red LENGTH > 1pt:")
for i, p in enumerate(polys):
    binter = p.boundary.intersection(red_union)
    bint_len = 0
    if binter.is_empty:
        continue
    if binter.geom_type == "LineString":
        bint_len = binter.length
    elif hasattr(binter, "geoms"):
        for g in binter.geoms:
            if hasattr(g, "length"):
                bint_len += g.length
    if bint_len > 1.0:
        print(f"  poly#{i} area={p.area:.1f} bnd∩red len={bint_len:.2f} bounds={p.bounds}")
