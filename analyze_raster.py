"""Custom polygonizacja oparta na rastrze + flood-fill + boundary check.

Motywacja: Shapely polygonize tworzy polygony "ze wszystkich domkniętych
cykli grafu zielonych linii" — ale graph jest często zbyt fragmentowany
(pikselowe artefakty, T-junctions) i powstają polygony które NIE odpowiadają
działkom na mapie.

Prostsze podejście:
  1. Rasteryzuj zielone granice + wszystkie closures (T-j, pins, parallel, ramka).
  2. Flood-fill na NEGATYWIE → connected components (każde CC = jedna działka
     lub "tło zewnętrzne").
  3. Dla każdej etykiety OCR: w której CC siedzi?
  4. Dla każdej CC: sprawdź czy jej GRANICA jest przecięta przez trasę
     (dilated red mask dotyka granicy CC).
  5. Etykieta w przeciętej CC → crossed.

Fallback: jeśli trasa jest w CAŁOŚCI wewnątrz jednej CC (nie przecina
żadnej granicy) — ta CC jest crossed (trasa siedzi w obrębie jednej działki).
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
import numpy as np
from skimage.morphology import skeletonize

from analyze_cv import (
    RasterCtx, extract_paths, rasterize_paths, morph_close,
    LABEL_RE, OCR_CONF_MIN, Label,
)
import analyze_hybrid as _ah


@dataclass
class Result:
    crossed: list[str] = field(default_factory=list)
    borderline: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def build_closed_green_mask(page, *, scale: float = 4.0,
                            line_thickness: int = 2,
                            parallel_close_pt: float = 120.0,
                            parallel_cluster_r_pt: float = 15.0,
                            tj_extend_pt: float = 40.0,
                            pin_crossline_half_pt: float = 25.0,
                            endpoint_close_pt: float = 5.0,
                            endpoint_extend_pt: float = 80.0,
                            route_end_half_pt: float = 250.0,
                            route_buffer_pt: float = 100.0,
                            frame: bool = True) -> tuple[np.ndarray, RasterCtx, list, list]:
    """Zwraca (mask, ctx, reds_trace, red_pins) — domkniętą maskę zielonych
    granic gotową do flood-fill."""
    greens, reds_trace, red_pins = extract_paths(page)
    W_pt, H_pt = page.rect.width, page.rect.height
    ctx = RasterCtx(scale=scale,
                    width_px=int(round(W_pt * scale)),
                    height_px=int(round(H_pt * scale)))

    # 1. rasteryzuj zielone + wszystkie closures jako VECTOR, potem raster
    segs = _ah.green_segments_vec(greens)
    segs.extend(_ah.find_endpoint_closures(greens, ctx,
                                           line_thickness=line_thickness,
                                           max_gap_pt=endpoint_close_pt))
    par_segs = _ah.find_parallel_endpoint_closures(
        greens, ctx, line_thickness=line_thickness,
        max_gap_pt=parallel_close_pt,
        cluster_r_pt=parallel_cluster_r_pt,
    )
    segs.extend(par_segs)
    segs.extend(_ah.find_endpoint_extensions(
        greens, ctx, line_thickness=line_thickness,
        max_extend_pt=endpoint_extend_pt, extra_segments=par_segs))
    segs.extend(_ah.find_tjunction_extensions(
        greens, ctx, line_thickness=line_thickness,
        max_extend_pt=tj_extend_pt))
    segs.extend(_ah.pin_crossline_segments(
        red_pins, reds_trace, half_length_pt=pin_crossline_half_pt))
    segs.extend(_ah.route_end_closures(reds_trace, half_length_pt=route_end_half_pt))
    if route_buffer_pt > 0:
        segs.extend(_ah.route_buffer_frame(reds_trace, buffer_pt=route_buffer_pt))
    if frame:
        segs.extend(_ah.frame_segments(page.rect))

    mask = np.zeros((ctx.height_px, ctx.width_px), dtype=np.uint8)
    for p1, p2 in segs:
        x1, y1 = ctx.pt2px(*p1)
        x2, y2 = ctx.pt2px(*p2)
        cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness, cv2.LINE_8)

    # drobne szczeliny pikselowe (1-2px)
    mask = morph_close(mask, 1)
    return mask, ctx, reds_trace, red_pins


def parcel_components(green_mask: np.ndarray, *,
                      red_mask: np.ndarray | None = None,
                      min_area_px: int = 200,
                      max_area_ratio: float = 0.5,
                      big_cc_trace_ratio: float = 0.95,
                      ) -> tuple[np.ndarray, int, np.ndarray]:
    """Flood-fill na NEGATYWIE maski — każda CC = kandydat na działkę.

    Wykluczamy z CC:
      - dotykające brzegu obrazu (zewnętrzne tło),
      - za małe (<min_area_px, szum),
      - za duże (>max_area_ratio * total) JEŻELI nie zawierają ≥`big_cc_trace_ratio`
        frakcji trasy. To chroni przed „wyciekami" gdy polygonize nie domknął
        obszaru i tło mapy jest jednym wielkim komponentem — a jednocześnie
        pozwala zachować dużą działkę gdy TO ONA zawiera trasę w całości
        (przypadek Teligi).
    """
    inv = (green_mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    H, W = inv.shape
    total = H * W
    max_area = int(max_area_ratio * total)
    edge_labels = set()
    edge_labels.update(labels[0, :].tolist())
    edge_labels.update(labels[H - 1, :].tolist())
    edge_labels.update(labels[:, 0].tolist())
    edge_labels.update(labels[:, W - 1].tolist())
    edge_labels.discard(0)

    # policz frakcję trasy w każdym CC (dla reguły big_cc_trace_ratio)
    red_frac = np.zeros(n, dtype=float)
    if red_mask is not None and red_mask.max() > 0:
        red_labels_per = labels[red_mask > 0]
        red_counts = np.bincount(red_labels_per.ravel(), minlength=n)
        total_red = red_counts.sum()
        if total_red > 0:
            red_frac = red_counts.astype(float) / total_red

    remap = np.zeros(n, dtype=np.int32)
    new_stats = [None]
    nxt = 1
    for lab in range(n):
        area = stats[lab, cv2.CC_STAT_AREA]
        is_bg = (lab == 0
                 or lab in edge_labels
                 or area < min_area_px)
        if not is_bg and area > max_area:
            # duży CC — zachowaj tylko gdy zawiera większość trasy
            is_bg = red_frac[lab] < big_cc_trace_ratio
        if is_bg:
            remap[lab] = 0
        else:
            remap[lab] = nxt
            new_stats.append(area)
            nxt += 1
    new_labels = remap[labels]
    return new_labels, nxt - 1, np.array(new_stats[1:] or [0])


def find_crossed_components(labels_img: np.ndarray,
                            red_mask: np.ndarray, *,
                            min_coverage_px: int = 10) -> set[int]:
    """Komponent jest CROSSED gdy NIEDILATOWANA trasa pokrywa co najmniej
    `min_coverage_px` jego wnętrza.

    Bez dilation: jeśli trasa tylko "muska" granicę CC, jej pixele są na
    granicy (label=0 w labels_img, bo granica to zielony pixel) — nie łapie
    sąsiednich CC przez tangent. Jeśli trasa faktycznie wchodzi w CC,
    ma piksele wewnątrz (label=cid) w ilości > min_coverage_px.
    """
    if red_mask.max() == 0:
        return set()
    red_labels = labels_img[red_mask > 0]
    counts = np.bincount(red_labels.ravel(),
                         minlength=int(labels_img.max()) + 1)
    crossed = set()
    for cid, cnt in enumerate(counts):
        if cid == 0:
            continue
        if cnt >= min_coverage_px:
            crossed.add(int(cid))
    return crossed


def label_component(labels_img: np.ndarray, x_px: int, y_px: int,
                    *, search_radius_px: int = 30) -> int:
    """Znajdź id komponentu pod punktem (x,y). Jeśli tam 0 (granica/tło),
    szukaj spiralnie w promieniu."""
    H, W = labels_img.shape
    if not (0 <= x_px < W and 0 <= y_px < H):
        return 0
    v = labels_img[y_px, x_px]
    if v != 0:
        return int(v)
    y0 = max(0, y_px - search_radius_px)
    y1 = min(H, y_px + search_radius_px + 1)
    x0 = max(0, x_px - search_radius_px)
    x1 = min(W, x_px + search_radius_px + 1)
    window = labels_img[y0:y1, x0:x1]
    ys, xs = np.where(window != 0)
    if len(ys) == 0:
        return 0
    dy = ys - (y_px - y0)
    dx = xs - (x_px - x0)
    d2 = dy * dy + dx * dx
    i = int(np.argmin(d2))
    return int(window[ys[i], xs[i]])


def load_ocr_cache(path: Path, ocr_scale: int = 8) -> list[Label]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    out = []
    for r in raw:
        t = r["text"].strip()
        if not LABEL_RE.match(t):
            continue
        if r.get("conf", 0) < OCR_CONF_MIN:
            continue
        if "x" in r and "y" in r:
            x, y = r["x"], r["y"]
        elif "cx" in r and "cy" in r:
            x, y = r["cx"] / ocr_scale, r["cy"] / ocr_scale
        else:
            continue
        out.append(Label(text=t, conf=float(r["conf"]),
                         x=float(x), y=float(y)))
    return out


def analyze(pdf_path: str | Path, *,
            ocr_cache: str | Path,
            ocr_scale: int = 8,
            scale: float = 4.0,
            parallel_close_pt: float = 120.0,
            tj_extend_pt: float = 40.0,
            pin_crossline_half_pt: float = 25.0,
            endpoint_extend_pt: float = 80.0,
            route_buffer_pt: float = 100.0,
            min_component_area_px: int = 200,
            max_component_area_ratio: float = 0.5,
            red_thickness: int = 3,
            green_thickness: int = 2,
            min_coverage_px: int = 10) -> Result:
    log = logging.getLogger("analyze_raster")
    doc = fitz.open(str(pdf_path))
    page = doc[0]

    # Adaptacja: route_buffer_frame dzieli działki pasa drogowego po obu
    # stronach trasy na sliverów — ale dla map gdzie trasa jest krótka
    # i siedzi W OBRĘBIE jednej działki, route_buffer_frame wprowadza
    # ARTEFAKTY GRANICY wewnątrz działki. Heurystyka: route_buffer tylko
    # gdy trasa zajmuje >=15% rozmiaru strony.
    greens_tmp, reds_tmp, _ = extract_paths(page)
    page_extent = max(page.rect.width, page.rect.height)
    route_extent = 0.0
    if reds_tmp:
        xs_, ys_ = [], []
        from analyze_cv import iter_segments as _iter
        for d in reds_tmp:
            for p1, p2 in _iter(d):
                xs_.append(p1[0]); xs_.append(p2[0])
                ys_.append(p1[1]); ys_.append(p2[1])
        if xs_:
            route_extent = max(max(xs_) - min(xs_), max(ys_) - min(ys_))
    ratio = route_extent / page_extent if page_extent > 0 else 0.0
    use_buffer = ratio >= 0.15
    log.info("route_extent=%.0fpt page_extent=%.0fpt ratio=%.2f use_buffer=%s",
             route_extent, page_extent, ratio, use_buffer)

    mask, ctx, reds_trace, red_pins = build_closed_green_mask(
        page, scale=scale, line_thickness=green_thickness,
        parallel_close_pt=parallel_close_pt,
        tj_extend_pt=tj_extend_pt,
        pin_crossline_half_pt=pin_crossline_half_pt,
        endpoint_extend_pt=endpoint_extend_pt,
        route_buffer_pt=route_buffer_pt if use_buffer else 0.0,
    )

    # rasteryzuj trasę (grubiej — dashed mostkujemy przez dilation)
    red_mask = rasterize_paths(reds_trace, ctx, thickness=red_thickness)

    # komponenty (działki z etykietami); używamy red_mask do heurystyki
    # "duży CC z całą trasą = zachować (Teligi), bez trasy = tło"
    labels_img, n_components, areas = parcel_components(
        mask, red_mask=red_mask,
        min_area_px=min_component_area_px,
        max_area_ratio=max_component_area_ratio,
    )
    log.info("components: %d (min_area=%d px)", n_components, min_component_area_px)

    crossed_cids = find_crossed_components(
        labels_img, red_mask,
        min_coverage_px=min_coverage_px,
    )
    log.info("crossed components: %d", len(crossed_cids))

    # OCR
    labels = load_ocr_cache(Path(ocr_cache), ocr_scale=ocr_scale)
    log.info("OCR labels: %d", len(labels))

    # dopasowanie etykieta → komponent
    crossed_texts: dict[str, Label] = {}
    all_label_cids: dict[str, int] = {}
    search_r_px = max(5, int(round(10 * scale)))
    for lbl in labels:
        x_px, y_px = ctx.pt2px(lbl.x, lbl.y)
        cid = label_component(labels_img, x_px, y_px, search_radius_px=search_r_px)
        all_label_cids.setdefault(lbl.text, cid)
        if cid > 0 and cid in crossed_cids:
            prev = crossed_texts.get(lbl.text)
            if prev is None or lbl.conf > prev.conf:
                crossed_texts[lbl.text] = lbl

    # FALLBACK A: trasa w CAŁOŚCI wewnątrz jednej działki — komponent z
    # większością trasy (Teligi).
    if not crossed_cids and red_mask.max() > 0:
        bincounts = np.bincount(
            labels_img[red_mask > 0].ravel(),
            minlength=n_components + 1,
        )
        bincounts[0] = 0
        if bincounts.max() > 10:
            best_cid = int(np.argmax(bincounts))
            crossed_cids = {best_cid}
            log.info("FALLBACK A: route inside component %d (area=%d)",
                     best_cid, areas[best_cid - 1] if best_cid > 0 else 0)
            for lbl in labels:
                x_px, y_px = ctx.pt2px(lbl.x, lbl.y)
                cid = label_component(labels_img, x_px, y_px,
                                      search_radius_px=search_r_px)
                if cid == best_cid:
                    prev = crossed_texts.get(lbl.text)
                    if prev is None or lbl.conf > prev.conf:
                        crossed_texts[lbl.text] = lbl

    # FALLBACK B: jeśli wciąż nic — etykieta najbliższa trasie (trasa w
    # obrębie działki której polygonize w ogóle nie wydzielił).
    if not crossed_texts and red_mask.max() > 0 and labels:
        from shapely.geometry import Point
        from analyze_ray import build_red_union as _ray_red
        ru = _ray_red(reds_trace, buffer_pt=0)
        if not ru.is_empty:
            scored = [(ru.distance(Point(lbl.x, lbl.y)), lbl) for lbl in labels]
            scored.sort()
            d_best, lbl = scored[0]
            if d_best < 200.0:
                crossed_texts[lbl.text] = lbl
                log.info("FALLBACK B: nearest-to-route %s at d=%.1fpt",
                         lbl.text, d_best)

    crossed = sorted(crossed_texts.keys())
    return Result(
        crossed=crossed,
        debug={
            "n_components": n_components,
            "n_crossed_components": len(crossed_cids),
        },
    )
