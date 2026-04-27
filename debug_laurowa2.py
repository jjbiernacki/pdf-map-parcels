"""Debug Laurowa 15 (FP) and 260 (FN)."""
import fitz
from pathlib import Path
from shapely.geometry import Point
from analyze_cv import extract_paths, load_ocr_cache
from analyze_hybrid import (build_polygons, build_red_union, green_segments_vec,
    find_endpoint_closures, find_endpoint_extensions,
    find_parallel_endpoint_closures, find_tjunction_extensions,
    pin_crossline_segments, route_end_closures, route_buffer_frame,
    detect_road_corridor_polygon, frame_segments, RasterCtx)

doc = fitz.open('Mapy/PZT Laurowa-Model.pdf'); page = doc[0]
greens, reds, _ = extract_paths(page)
labels = load_ocr_cache(Path('/tmp/mapy_ocr/PZT Laurowa-Model.pkl'), ocr_scale=8)
lbl_by_text = {l.text: l for l in labels}

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
ru = build_red_union(reds)
wide5 = ru.buffer(5.0, cap_style=2, join_style=2)
corridor = detect_road_corridor_polygon(reds, greens)
print(f"corridor: {corridor.bounds if corridor else None}")
for name in ['15', '260', '75']:
    l = lbl_by_text[name]
    pt = Point(l.x, l.y)
    print(f"\n=== {name} at ({l.x:.1f}, {l.y:.1f}) ===")
    print(f"  in corridor: {corridor.contains(pt) if corridor else None}")
    found = False
    for i, p in enumerate(polys):
        if p.contains(pt):
            wb = p.boundary.intersection(wide5)
            wb_len = wb.length if hasattr(wb, 'length') else 0
            perim = p.boundary.length
            inside = p.buffer(-0.5)
            inter = inside.intersection(ru) if not inside.is_empty else None
            inter_len = inter.length if inter is not None and hasattr(inter, 'length') else 0
            d_route = p.distance(ru)
            print(f"  poly#{i} area={p.area:.0f} perim={perim:.1f} bounds={p.bounds}")
            print(f"  wb_len={wb_len:.2f} ({wb_len/perim*100:.1f}%) interior_red_len={inter_len:.2f} d_route={d_route:.2f}")
            bint = p.boundary.intersection(ru)
            bint_len = bint.length if hasattr(bint, 'length') else 0
            n_comp = 1 if not bint.is_empty and not hasattr(bint, 'geoms') else (0 if bint.is_empty else len(list(bint.geoms)))
            print(f"  boundary∩red type={bint.geom_type} len={bint_len:.2f} n_comp={n_comp}")
            found = True
            break
    if not found:
        print(f"  NOT IN ANY POLY (large filter)")
        d = ru.distance(pt)
        print(f"  d_label_route={d:.1f}")
