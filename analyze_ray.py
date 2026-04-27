"""Ray-casting approach: działka CROSSED ⇔ którakolwiek z jej granic jest
przecinana przez trasę.

Algorytm:
  1.  Zbierz wszystkie zielone odcinki (granice) i czerwone (trasa).
  2.  Dla każdej zielonej granicy sprawdź czy trasa ją przecina
      (Shapely intersects) — nazwij taką granicę `crossed`.
  3.  Dla każdej etykiety OCR wyznacz "granice działki L" przez
      ray casting: z punktu etykiety puścić N promieni co 360°/N,
      w każdym kierunku znaleźć PIERWSZY (najbliższy) zielony
      odcinek trafiony przez promień.
  4.  Jeśli CHOĆ JEDNA z tych granic jest `crossed` → L jest crossed.

Zalety tego podejścia (sugestia użytkownika):
  - Nie wymaga zamkniętego polygonize — działa nawet gdy działka
    ma otwarte granice.
  - Trasa może się rozgałęziać — algorytm traktuje ją jako zbiór
    odcinków i sprawdza przecięcia z dowolnym z nich.
  - Filtruje stacje pomiarowe naturalnie: stacja w pasie drogowym
    "widzi" tylko RÓWNOLEGŁE do trasy granice pasa (nieprzecinane),
    więc nie jest klasyfikowana jako crossed.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from analyze_cv import extract_paths, iter_segments, Label, load_ocr_cache


@dataclass
class Result:
    crossed: list[str] = field(default_factory=list)
    borderline: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def build_green_segments(greens) -> list[LineString]:
    segs = []
    for d in greens:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                segs.append(LineString([p1, p2]))
    return segs


def build_red_union(reds_trace, *, buffer_pt: float = 0.0):
    """Unia odcinków trasy, opcjonalnie z buforem (zamyka luki między
    kreseczkami kreskowanej trasy — długie dziury 1-3pt między sąsiednimi
    kreskami powodują że zielone granice je "omijają" bez przecięcia)."""
    segs = []
    for d in reds_trace:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                segs.append(LineString([p1, p2]))
    if not segs:
        return LineString()
    u = unary_union(MultiLineString(segs))
    if buffer_pt > 0:
        u = u.buffer(buffer_pt, cap_style=2, join_style=2)
    return u


def compute_crossed_greens(green_segs: list[LineString], red_union,
                           min_cross_len: float = 0.0) -> set[int]:
    """Zwraca indeksy zielonych segmentów przeciętych przez trasę."""
    if not green_segs or red_union.is_empty:
        return set()
    out = set()
    for i, seg in enumerate(green_segs):
        if not seg.intersects(red_union):
            continue
        ip = seg.intersection(red_union)
        # Point, MultiPoint — na pewno przecięcie transwersalne
        if ip.geom_type in ("Point", "MultiPoint"):
            out.add(i)
            continue
        # tangencja długością — sprawdź że intersection > min_cross_len
        if hasattr(ip, "length") and ip.length >= min_cross_len:
            out.add(i)
    return out


def label_is_crossed(origin: Point, green_segs: list[LineString],
                     tree: STRtree, crossed_greens: set[int], *,
                     n_rays: int, max_ray_pt: float,
                     all_hits: bool = False) -> tuple[bool, list[int]]:
    """Ray casting z punktu etykiety.

    `all_hits=False` (klasyczny): bierze tylko PIERWSZĄ (najbliższą) granicę
    na promieniu. Problem: bliższe granice mogą "zasłaniać" dalsze
    (przeciętą przez trasę granicę).

    `all_hits=True`: zbiera WSZYSTKIE granice trafione na promieniu
    w zasięgu max_ray_pt. Łapie dalsze granice za zasłoną najbliższej.
    """
    ox, oy = origin.x, origin.y
    hit_borders: list[int] = []
    crossed_hit = False
    for k in range(n_rays):
        angle = 2 * math.pi * k / n_rays
        dx, dy = math.cos(angle), math.sin(angle)
        ex, ey = ox + dx * max_ray_pt, oy + dy * max_ray_pt
        ray = LineString([(ox, oy), (ex, ey)])
        try:
            cands = tree.query(ray)
        except Exception:
            cands = range(len(green_segs))
        if all_hits:
            for idx in cands:
                i = int(idx)
                seg = green_segs[i]
                if seg.intersects(ray):
                    hit_borders.append(i)
                    if i in crossed_greens:
                        crossed_hit = True
        else:
            best_dist = None
            best_i = None
            for idx in cands:
                i = int(idx)
                seg = green_segs[i]
                if not seg.intersects(ray):
                    continue
                ip = seg.intersection(ray)
                if ip.geom_type == "Point":
                    d = origin.distance(ip)
                elif ip.geom_type == "MultiPoint":
                    d = min(origin.distance(p) for p in ip.geoms)
                elif hasattr(ip, "coords"):
                    d = min(origin.distance(Point(c)) for c in ip.coords)
                else:
                    continue
                if d < 1e-6:
                    continue
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_i = i
            if best_i is not None:
                hit_borders.append(best_i)
                if best_i in crossed_greens:
                    crossed_hit = True
    return crossed_hit, hit_borders


def analyze(
    pdf_path: str | Path,
    *,
    ocr_cache: str | Path | None = None,
    ocr_scale: int = 8,
    n_rays: int = 24,
    max_ray_pt: float = 400.0,
    red_buffer_pt: float = 2.0,
    all_hits: bool = False,
) -> Result:
    log = logging.getLogger("analyze_ray")
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    greens, reds_trace, red_pins = extract_paths(page)

    green_segs = build_green_segments(greens)
    red_union = build_red_union(reds_trace, buffer_pt=red_buffer_pt)
    log.info("paths: green_segs=%d red_union_type=%s",
             len(green_segs), red_union.geom_type)

    crossed_greens = compute_crossed_greens(green_segs, red_union)
    log.info("crossed green borders: %d / %d", len(crossed_greens), len(green_segs))

    tree = STRtree(green_segs)

    if not ocr_cache:
        raise RuntimeError("Brak cache OCR.")
    labels = load_ocr_cache(Path(ocr_cache), ocr_scale=ocr_scale)
    log.info("OCR labels (valid): %d", len(labels))

    crossed_texts: dict[str, Label] = {}
    for lbl in labels:
        origin = Point(lbl.x, lbl.y)
        is_crossed, hit_borders = label_is_crossed(
            origin, green_segs, tree, crossed_greens,
            n_rays=n_rays, max_ray_pt=max_ray_pt, all_hits=all_hits,
        )
        if is_crossed:
            prev = crossed_texts.get(lbl.text)
            if prev is None or lbl.conf > prev.conf:
                crossed_texts[lbl.text] = lbl

    crossed = sorted(crossed_texts.keys())
    log.info("crossed=%d (%.1fs)", len(crossed), time.time() - t0)
    return Result(crossed=crossed, debug={"n_green_segs": len(green_segs),
                                          "n_crossed_greens": len(crossed_greens)})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--ocr-cache", default="/tmp/ocr_cache_v2.pkl")
    p.add_argument("--n-rays", type=int, default=24)
    p.add_argument("--max-ray", type=float, default=400.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    res = analyze(args.pdf, ocr_cache=args.ocr_cache,
                  n_rays=args.n_rays, max_ray_pt=args.max_ray)
    print(f"DZIAŁKI PRZECIĘTE ({len(res.crossed)}):")
    print("  " + (", ".join(res.crossed) if res.crossed else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
