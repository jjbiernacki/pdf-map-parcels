"""Hybrydowy pipeline: OpenCV (do wykrycia T-junction + pinów) + Shapely
polygonize (do dekompozycji na działki).

Motywacja: raster-based komponenty „wyciekają" (98% pikseli w jednym komponencie).
Vector polygonize precyzyjnie zamyka wielokąty jeśli geometria jest domknięta.

Pipeline:
  1. Wyciągnij zielone odcinki z PDF-a (vector).
  2. Rasteryzuj zielone → skeletonize → znajdź T-junctions.
  3. Dla każdego T-j wyznacz kierunek gałęzi bocznej (antykolinearne 2 gł.).
  4. Ekstrapoluj z T-j w kierunku gałęzi o długość `tj_extend_pt`
     — dodaj jako NOWE odcinki vector.
  5. Dla każdego pina wyznacz prostopadłą do osi trasy, narysuj crossline
     jako NOWE odcinki vector.
  6. Dodaj ramkę mapy jako 4 odcinki vector (bbox + mały margines).
  7. unary_union + snap(tol) + polygonize → polygons.
  8. Klasyfikacja:
     - polygon przecinany przez red_union (buffer -0.5) → crossed
     - etykieta w polygon crossed → działka CROSSED
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize, snap, unary_union
from skimage.morphology import skeletonize

from analyze_cv import (
    RasterCtx, rasterize_paths, morph_close, find_endpoints,
    find_tjunctions, tjunction_branch_direction, _walk_from,
    _direction_from_path, red_segments, dedup_pins, pin_centers_pt,
    trace_tangent_at, extract_paths, iter_segments,
)
from analyze_cv import LABEL_RE, OCR_CONF_MIN, Label, load_ocr_cache


@dataclass
class Result:
    crossed: list[str] = field(default_factory=list)
    borderline: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def green_segments_vec(greens) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segs = []
    for d in greens:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                segs.append((p1, p2))
    return segs


def find_tjunction_extensions(greens, ctx: RasterCtx, *,
                              line_thickness: int, max_extend_pt: float,
                              min_branch_len: int = 4):
    """Ekstrapoluj gałęzie T-junctions jako odcinki vector."""
    mask = rasterize_paths(greens, ctx, thickness=line_thickness)
    mask = morph_close(mask, 2)
    sk = skeletonize(mask > 0)
    tj = find_tjunctions(sk)
    ys, xs = np.where(tj)
    out = []
    max_extend_px = max_extend_pt * ctx.scale
    for y, x in zip(ys, xs):
        d = tjunction_branch_direction(sk, int(y), int(x), steps=min_branch_len)
        if d is None:
            continue
        end_y, end_x = None, None
        reached = 0
        for t in range(1, int(max_extend_px) + 1):
            ny = int(round(y + d[0] * t))
            nx = int(round(x + d[1] * t))
            if ny < 0 or ny >= sk.shape[0] or nx < 0 or nx >= sk.shape[1]:
                break
            if t >= 2 and sk[ny, nx]:
                end_y, end_x = ny, nx
                break
            end_y, end_x = ny, nx
            reached = t
        if end_y is None or reached < 2:
            continue
        p1 = ctx.px2pt(int(x), int(y))
        p2 = ctx.px2pt(int(end_x), int(end_y))
        out.append((p1, p2))
    return out


def find_endpoint_closures(greens, ctx: RasterCtx, *,
                           line_thickness: int, max_gap_pt: float):
    """Parowanie wolnych końcówek szkieletu w promieniu `max_gap_pt`.
    Zwraca segmenty vector łączące pary."""
    from scipy.spatial import cKDTree
    mask = rasterize_paths(greens, ctx, thickness=line_thickness)
    mask = morph_close(mask, 2)
    sk = skeletonize(mask > 0)
    ep = find_endpoints(sk)
    ys, xs = np.where(ep)
    if len(ys) == 0:
        return []
    coords = np.column_stack([ys, xs])
    tree = cKDTree(coords)
    max_gap_px = max_gap_pt * ctx.scale
    pairs = tree.query_pairs(r=max_gap_px)
    pairs_sorted = []
    for i, j in pairs:
        dy = coords[i, 0] - coords[j, 0]
        dx = coords[i, 1] - coords[j, 1]
        dist = (dy * dy + dx * dx) ** 0.5
        pairs_sorted.append((dist, i, j))
    pairs_sorted.sort()
    used = np.zeros(len(coords), dtype=bool)
    out = []
    for dist, i, j in pairs_sorted:
        if used[i] or used[j]:
            continue
        y1, x1 = int(coords[i, 0]), int(coords[i, 1])
        y2, x2 = int(coords[j, 0]), int(coords[j, 1])
        p1 = ctx.px2pt(x1, y1)
        p2 = ctx.px2pt(x2, y2)
        out.append((p1, p2))
        used[i] = used[j] = True
    return out


def _cluster_endpoints(coords: np.ndarray, dirs: list, *,
                       cluster_r_px: float):
    """Klastruj endpointy w promieniu `cluster_r_px` do 1 reprezentanta.

    Skeletonize produkuje skupiska endpointów wokół T/Y-junctions — to
    artefakty pikselowe, a nie prawdziwe wolne końce. Klastrowanie usuwa
    szum i pozwala greedy pairing łapać właściwe pary.
    """
    from scipy.spatial import cKDTree
    if len(coords) == 0:
        return coords, dirs
    tree = cKDTree(coords)
    visited = np.zeros(len(coords), dtype=bool)
    out_coords = []
    out_dirs = []
    for i in range(len(coords)):
        if visited[i]:
            continue
        idx = tree.query_ball_point(coords[i], r=cluster_r_px)
        idx = [j for j in idx if not visited[j]]
        visited[idx] = True
        pts = coords[idx]
        centroid = pts.mean(axis=0)
        # średni tangens (unik kierunków odwrotnych przez flip)
        valid_dirs = [dirs[j] for j in idx if dirs[j] is not None]
        if not valid_dirs:
            d_avg = None
        else:
            ref = valid_dirs[0]
            flipped = [d if float(np.dot(d, ref)) >= 0 else -d for d in valid_dirs]
            d_avg = np.mean(flipped, axis=0)
            norm = float(np.linalg.norm(d_avg))
            d_avg = d_avg / norm if norm > 1e-6 else None
        out_coords.append(centroid)
        out_dirs.append(d_avg)
    return np.array(out_coords), out_dirs


def find_parallel_endpoint_closures(greens, ctx: RasterCtx, *,
                                    line_thickness: int,
                                    max_gap_pt: float,
                                    parallel_cos_thresh: float = 0.85,
                                    perpendicular_cos_thresh: float = 0.3,
                                    tangent_steps: int = 10,
                                    cluster_r_pt: float = 8.0):
    """Łączenie końcówek RÓWNOLEGŁYCH odcinków które "patrzą w tę samą stronę".

    Scenariusz: dwie pionowe linie kończą się na tej samej wysokości (np.
    „wystają" w górę z poziomej linii na dole, a nie dochodzą do równoległej
    poziomej linii powyżej). Ich tangensy są RÓWNOLEGŁE (oba W GÓRĘ), a
    wektor A→B jest PROSTOPADŁY do ich tangensów. Rysujemy odcinek A-B —
    zamyka polygon w miejscach gdzie brakuje poprzecznej granicy.

    Kryteria:
      - |cos(t_A, t_B)| >= parallel_cos_thresh (równoległe lub antyrównoległe)
      - |cos(t_A, A→B)| <= perpendicular_cos_thresh (wektor prostopadły)
      - odległość |A-B| <= max_gap_pt

    Greedy: każdy endpoint łączymy co najwyżej raz, sortując pary po odległości.
    """
    from scipy.spatial import cKDTree
    from analyze_cv import estimate_direction
    mask = rasterize_paths(greens, ctx, thickness=line_thickness)
    mask = morph_close(mask, 2)
    sk = skeletonize(mask > 0)
    ep = find_endpoints(sk)
    ys, xs = np.where(ep)
    if len(ys) == 0:
        return []
    coords = np.column_stack([ys, xs])
    dirs = [estimate_direction(sk, (int(y), int(x)), steps=tangent_steps)
            for y, x in coords]
    # klastrowanie eliminuje szumowe skupiska endpointów wokół T-junctions
    coords, dirs = _cluster_endpoints(coords, dirs,
                                      cluster_r_px=cluster_r_pt * ctx.scale)
    tree = cKDTree(coords)
    max_gap_px = max_gap_pt * ctx.scale
    pairs = tree.query_pairs(r=max_gap_px)
    candidates = []
    for i, j in pairs:
        di, dj = dirs[i], dirs[j]
        if di is None or dj is None:
            continue
        v = coords[j] - coords[i]  # (dy, dx)
        norm = float(np.linalg.norm(v))
        if norm < 1e-6:
            continue
        v_unit = v / norm
        # równoległość tangensów
        cos_par = abs(float(np.dot(di, dj)))
        if cos_par < parallel_cos_thresh:
            continue
        # prostopadłość wektora A→B do tangensu A (i przez to do B)
        cos_perp = abs(float(np.dot(di, v_unit)))
        if cos_perp > perpendicular_cos_thresh:
            continue
        candidates.append((norm, i, j))
    candidates.sort()
    # NIE greedy — jedna linia na parę (NIE max-1-per-endpoint), żeby kolejne
    # endpointy na tej samej "równoległej sekwencji" były wszystkie połączone.
    # Później shapely polygonize i snap rozstrzygną które kombinacje tworzą
    # zamknięte polygony.
    out = []
    for dist, i, j in candidates:
        y1, x1 = int(coords[i, 0]), int(coords[i, 1])
        y2, x2 = int(coords[j, 0]), int(coords[j, 1])
        p1 = ctx.px2pt(x1, y1)
        p2 = ctx.px2pt(x2, y2)
        out.append((p1, p2))
    return out


def find_endpoint_extensions(greens, ctx: RasterCtx, *,
                             line_thickness: int, max_extend_pt: float,
                             tangent_steps: int = 8,
                             extra_segments=None):
    """Wyprowadzanie wolnego endpointu w kierunku jego tangensu aż trafi
    w inny piksel szkieletu.

    Scenariusz: pionowa odnoga "wystaje" z poziomej linii ale nie dochodzi
    do równoległej linii powyżej. Kierunek tangensu endpointu = OD SZKIELETU
    W KIERUNKU dalej po niezakończonej kontynuacji. Ekstrapolacja wyprowadza
    odcinek aż trafi w inną zieloną linię — zamyka niedomknięty polygon.

    `extra_segments` — dodatkowe odcinki (np. parallel closures z
    poprzedniego passu) rasteryzowane do maski, żeby ekstrapolacja
    mogła w nie trafić.
    """
    from analyze_cv import estimate_direction
    mask = rasterize_paths(greens, ctx, thickness=line_thickness)
    # dorasteryzuj dodatkowe segmenty (np. parallel closures z pass 1)
    if extra_segments:
        for p1, p2 in extra_segments:
            x1, y1 = ctx.pt2px(*p1)
            x2, y2 = ctx.pt2px(*p2)
            cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness, cv2.LINE_8)
    mask = morph_close(mask, 2)
    sk = skeletonize(mask > 0)
    ep = find_endpoints(sk)
    ys, xs = np.where(ep)
    out = []
    max_extend_px = max_extend_pt * ctx.scale
    for y, x in zip(ys, xs):
        # tangens OD szkieletu NA ZEWNĄTRZ (_direction_from_path zwraca
        # path[0]-path[-1]) — idąc w tę stronę przedłużamy linię dalej
        d = estimate_direction(sk, (int(y), int(x)), steps=tangent_steps)
        if d is None:
            continue
        # Interpretacja: d wskazuje OD SZKIELETU NA ZEWNĄTRZ,
        # czyli = kierunek w którym linia "chciałaby" iść dalej.
        end_y, end_x = None, None
        reached = 0
        for t in range(1, int(max_extend_px) + 1):
            ny = int(round(y + d[0] * t))
            nx = int(round(x + d[1] * t))
            if ny < 0 or ny >= sk.shape[0] or nx < 0 or nx >= sk.shape[1]:
                break
            if t >= 3 and sk[ny, nx]:
                end_y, end_x = ny, nx
                break
            end_y, end_x = ny, nx
            reached = t
        if end_y is None or reached < 3:
            continue
        # rysuj tylko jeśli TRAFILIŚMY w szkielet (nie jeśli przeszliśmy
        # cały zasięg w pustce — to bez sensu, bo nie ma czego zamykać)
        if reached >= int(max_extend_px) - 1:
            continue
        p1 = ctx.px2pt(int(x), int(y))
        p2 = ctx.px2pt(int(end_x), int(end_y))
        out.append((p1, p2))
    return out


def pin_crossline_segments(red_pins_draw, red_trace_draw, *,
                           half_length_pt: float,
                           tangent_radius_pt: float = 20.0):
    """Zwraca sztuczne poprzeczne granice przez każdy pin jako odcinki vector."""
    red_segs = red_segments(red_trace_draw)
    pins = dedup_pins(pin_centers_pt(red_pins_draw), tol=4.0)
    out = []
    for (x, y) in pins:
        tan = trace_tangent_at(red_segs, x, y, r=tangent_radius_pt)
        if tan is None:
            continue
        nx, ny = -tan[1], tan[0]
        p1 = (x + nx * half_length_pt, y + ny * half_length_pt)
        p2 = (x - nx * half_length_pt, y - ny * half_length_pt)
        out.append((p1, p2))
    return out


def frame_segments(page_rect) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """4 odcinki ramki = bbox strony."""
    r = page_rect
    return [
        ((r.x0, r.y0), (r.x1, r.y0)),
        ((r.x1, r.y0), (r.x1, r.y1)),
        ((r.x1, r.y1), (r.x0, r.y1)),
        ((r.x0, r.y1), (r.x0, r.y0)),
    ]


def map_hull_segments(greens, ctx: RasterCtx, *,
                      dilate_pt: float = 40.0,
                      thickness_px: int = 2,
                      simplify_tol_pt: float = 2.0):
    """Zwraca "obwódkę mapy" — concave hull wszystkich zielonych linii
    jako listę odcinków vector.

    Algorytm:
      1. Rasteryzuj zielone granice.
      2. Dilatuj morfologicznie o `dilate_pt` — wszystkie zielone linie
         zlewają się w jedną spójną plamę.
      3. cv2.findContours(RETR_EXTERNAL) na tej plamie — zewnętrzny
         kontur = obwódka mapy.
      4. Simplify + uncast do pt + lista odcinków.

    Motywacja (sugestia użytkownika): luźne końce zielonych linii przy
    brzegu mapy powodują, że polygonize zostawia te działki jako otwarte
    wycieki. Obwódka łączy wszystkie wolne końce w jeden domknięty kontur,
    który zamyka brzegowe działki.
    """
    mask = rasterize_paths(greens, ctx, thickness=thickness_px)
    dil_px = max(1, int(round(dilate_pt * ctx.scale)))
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dil_px + 1, 2 * dil_px + 1)
    )
    dilated = cv2.dilate(mask, k)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    eps_px = max(1.0, simplify_tol_pt * ctx.scale)
    for c in contours:
        # prosty smoother: approxPolyDP żeby zredukować szum pikselowy
        approx = cv2.approxPolyDP(c, eps_px, closed=True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            p1 = ctx.px2pt(int(a[0]), int(a[1]))
            p2 = ctx.px2pt(int(b[0]), int(b[1]))
            if p1 != p2:
                out.append((p1, p2))
    return out


def _extract_long_green_polylines(greens, *, min_length_pt: float = 30.0):
    """Zwraca listę długich ciągłych polylines zielonych (nie pojedynczych
    segmentów) wraz z ich endpointami i kierunkiem.

    Shapely `unary_union(MultiLineString(segments))` scala kolineacyjne
    segmenty w ciągłe LineStrings, co pozwala znaleźć prawdziwe
    endpointy (a nie internal wierzchołki kinked-lines).
    """
    import math
    raw = []
    for d in greens:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                raw.append(LineString([p1, p2]))
    if not raw:
        return []
    merged = unary_union(MultiLineString(raw))
    out = []
    geoms = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    for g in geoms:
        if g.length < min_length_pt:
            continue
        coords = list(g.coords)
        if len(coords) < 2:
            continue
        p_start, p_end = coords[0], coords[-1]
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        direction = (dx / norm, dy / norm)
        out.append({
            "line": g,
            "start": p_start,
            "end": p_end,
            "dir": direction,
            "length": g.length,
        })
    return out


def parallel_line_pair_closures(greens, red_trace_draw, *,
                                 min_length_pt: float = 30.0,
                                 max_perp_dist_pt: float = 150.0,
                                 max_perp_between_pt: float = 120.0,
                                 angle_tol_deg: float = 15.0,
                                 max_endpoint_gap_pt: float = 200.0,
                                 min_route_extent_pt: float = 10.0):
    """Znajdź pary równoległych długich zielonych linii obok trasy i
    połącz ich odpowiadające sobie endpointy poprzecznymi odcinkami,
    zamykając „road strip polygon".

    Algorytm:
      1. Wyciągnij długie ciągłe polylines zielone (z merger scalającym
         kolineacyjne segmenty).
      2. Filtruj te w pobliżu trasy (≤ max_perp_dist_pt) i z kątem pasującym
         do kierunku trasy (±angle_tol_deg) LUB dominującego kierunku
         pozostałych kandydatów.
      3. Dla każdej pary (A, B) sprawdź: czy są RÓWNOLEGŁE (|cos|≥...),
         odległość prostopadła A↔B <= max_perp_between_pt (tworzą wąski pas).
      4. Dla takiej pary: znajdź „najbliższe odpowiadające sobie endpointy"
         (lewy A↔lewy B, prawy A↔prawy B — gdzie „lewy/prawy" wyznaczone
         kierunkiem równoległym). Połącz odcinkiem.
    """
    import math
    import numpy as np
    segs_red = red_segments(red_trace_draw)
    if not segs_red:
        return []
    # wektor kierunku trasy — uśredniony
    vsum = np.zeros(2)
    for (x1, y1), (x2, y2) in segs_red:
        v = np.array([x2 - x1, y2 - y1])
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        v = v / n
        if vsum.dot(v) < 0:
            v = -v
        vsum += v
    route_extent = max(
        max(x for (p1, p2) in segs_red for x in [p1[0], p2[0]])
        - min(x for (p1, p2) in segs_red for x in [p1[0], p2[0]]),
        max(y for (p1, p2) in segs_red for y in [p1[1], p2[1]])
        - min(y for (p1, p2) in segs_red for y in [p1[1], p2[1]]),
    )
    if route_extent < min_route_extent_pt:
        return []
    mls_red = unary_union(MultiLineString(
        [LineString([p1, p2]) for p1, p2 in segs_red if p1 != p2]
    ))
    # polylines
    pls = _extract_long_green_polylines(greens, min_length_pt=min_length_pt)
    # filtr po odległości od trasy
    near = []
    for pl in pls:
        d = pl["line"].distance(mls_red.centroid) if mls_red.centroid else 1e9
        if d <= max_perp_dist_pt:
            near.append(pl)
    if len(near) < 2:
        return []
    # Dla każdej pary sprawdź równoległość i odległość międzylinia
    out = []
    def _angle_deg(dx, dy):
        a = math.degrees(math.atan2(dy, dx)) % 180.0
        return a
    def _angle_diff(a, b):
        d = abs(a - b)
        return min(d, 180 - d)
    pairs_seen = set()
    for i, pl_a in enumerate(near):
        for j, pl_b in enumerate(near):
            if j <= i:
                continue
            if (i, j) in pairs_seen:
                continue
            pairs_seen.add((i, j))
            aA = _angle_deg(pl_a["dir"][0], pl_a["dir"][1])
            aB = _angle_deg(pl_b["dir"][0], pl_b["dir"][1])
            if _angle_diff(aA, aB) > angle_tol_deg:
                continue
            # Prostopadła odległość między liniami — dystans A.centroid do B
            d_ab = pl_a["line"].distance(pl_b["line"])
            if d_ab > max_perp_between_pt:
                continue
            # Orientuj kierunki tak żeby pl_a.dir i pl_b.dir wskazywały w tę
            # samą stronę (dot >= 0).
            dA = np.array(pl_a["dir"])
            dB = np.array(pl_b["dir"])
            if dA.dot(dB) < 0:
                dB = -dB
                bstart, bend = pl_b["end"], pl_b["start"]
            else:
                bstart, bend = pl_b["start"], pl_b["end"]
            # "Lewy"/"prawy" endpoint — po projekcji na wspólny kierunek dA
            ax1 = float(np.array(pl_a["start"]).dot(dA))
            ax2 = float(np.array(pl_a["end"]).dot(dA))
            bx1 = float(np.array(bstart).dot(dA))
            bx2 = float(np.array(bend).dot(dA))
            a_left = pl_a["start"] if ax1 < ax2 else pl_a["end"]
            a_right = pl_a["end"] if ax1 < ax2 else pl_a["start"]
            b_left = bstart if bx1 < bx2 else bend
            b_right = bend if bx1 < bx2 else bstart
            # Gap musi być ograniczony — żebyśmy nie rysowali absurdalnych
            # ukośnych poprzecznek między liniami które mają inne zakresy x
            gap_left = math.hypot(a_left[0] - b_left[0], a_left[1] - b_left[1])
            gap_right = math.hypot(a_right[0] - b_right[0], a_right[1] - b_right[1])
            if gap_left > max_endpoint_gap_pt and gap_right > max_endpoint_gap_pt:
                continue
            # Dodaj poprzeczki — łączące lewy A z lewym B i prawy A z prawym B
            if gap_left <= max_endpoint_gap_pt:
                out.append((tuple(a_left), tuple(b_left)))
            if gap_right <= max_endpoint_gap_pt:
                out.append((tuple(a_right), tuple(b_right)))
    return out


def road_corridor_closures(greens, red_trace_draw, page_rect, *,
                            min_segment_pt: float = 30.0,
                            max_perp_dist_pt: float = 120.0,
                            angle_tol_deg: float = 12.0,
                            extend_beyond_pt: float = 2000.0,
                            min_route_extent_pt: float = 10.0):
    """Wykrywa długie zielone odcinki równoległe do kierunku trasy i
    PRZEDŁUŻA JE do krawędzi strony (lub daleko poza bieżące endpointy).

    Motywacja: pas drogowy (road strip) to długi, wąski region ograniczony
    dwiema równoległymi zielonymi liniami biegnącymi wzdłuż trasy. Te linie
    często NIE DOCHODZĄ do krawędzi mapy — kończą się w powietrzu — więc
    polygonize nie może zamknąć road-strip polygonu i działki wewnątrz pasa
    (np. numery działek drogowych 470, 423 w Fabryczna) trafiają do wielkiego
    zewnętrznego polygonu-"wycieku".

    Algorytm:
      1. Wyznacz kierunek trasy (unit vector średniego kierunku czerwonych
         odcinków).
      2. Znajdź długie zielone odcinki (len > min_segment_pt) w pobliżu trasy
         (prostopadła odległość <= max_perp_dist_pt) z kątem mieszczącym się
         w angle_tol_deg od kierunku trasy.
      3. Dla każdego takiego odcinka: PRZEDŁUŻ oba endpointy w kierunku
         tangensu odcinka aż trafi w krawędź strony (+ mały margines poza).
      4. Zwróć nowe (p1→p2) odcinki które zastępują lub uzupełniają oryginalne.
    """
    import math
    segs_red = red_segments(red_trace_draw)
    if not segs_red:
        return []

    # Kierunek trasy: uśredniony wektor wszystkich red-segmentów
    # (normalized per segment, żeby nie faworyzować długich).
    vsum = np.zeros(2)
    for (x1, y1), (x2, y2) in segs_red:
        v = np.array([x2 - x1, y2 - y1])
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        v = v / n
        # Orientuj wszystkie wektory w tę samą stronę (dot z vsum >= 0)
        if vsum.dot(v) < 0:
            v = -v
        vsum += v
    route_extent = max(
        max(x for (p1, p2) in segs_red for x in [p1[0], p2[0]])
        - min(x for (p1, p2) in segs_red for x in [p1[0], p2[0]]),
        max(y for (p1, p2) in segs_red for y in [p1[1], p2[1]])
        - min(y for (p1, p2) in segs_red for y in [p1[1], p2[1]]),
    )
    if route_extent < min_route_extent_pt:
        return []
    if np.linalg.norm(vsum) < 1e-6:
        return []
    route_dir = vsum / np.linalg.norm(vsum)
    route_angle = math.degrees(math.atan2(route_dir[1], route_dir[0])) % 180.0

    # Centroid trasy dla dystansu prostopadłego
    mls_red = unary_union(MultiLineString(
        [LineString([p1, p2]) for p1, p2 in segs_red if p1 != p2]
    ))

    out = []
    for d in greens:
        for p1, p2 in iter_segments(d):
            if p1 == p2:
                continue
            x1, y1 = p1
            x2, y2 = p2
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_segment_pt:
                continue
            # kąt odcinka
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            # różnica kątów (cyklicznie w [0, 90])
            da = min(abs(ang - route_angle), 180.0 - abs(ang - route_angle))
            if da > angle_tol_deg:
                continue
            # odległość prostopadła do trasy (do najbliższego punktu trasy)
            mid = Point((x1 + x2) / 2, (y1 + y2) / 2)
            d_perp = mls_red.distance(mid)
            if d_perp > max_perp_dist_pt:
                continue
            # Przedłuż segment wzdłuż jego kierunku aż do brzegu strony (+ margines)
            vx = (x2 - x1) / length
            vy = (y2 - y1) / length
            # przedłuż w obie strony o bardzo dużą odległość, potem zaklipuj do
            # bounding boxa strony
            L = extend_beyond_pt
            nx1 = x1 - vx * L
            ny1 = y1 - vy * L
            nx2 = x2 + vx * L
            ny2 = y2 + vy * L
            # klipuj do page_rect
            def _clip(px, py, x1r, y1r, x2r, y2r):
                # prosty clip (bez Liang-Barsky, wystarczy że zmieścimy do bbox)
                # Tutaj tylko ograniczamy do bbox — nie musi być dokładnie na
                # brzegu.
                return (max(x1r, min(x2r, px)), max(y1r, min(y2r, py)))
            r = page_rect
            m = 5.0  # margines
            np1 = _clip(nx1, ny1, r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m)
            np2 = _clip(nx2, ny2, r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m)
            out.append((np1, np2))
    return out


def route_buffer_frame(red_trace_draw, *, buffer_pt: float = 200.0,
                       simplify_tol: float = 2.0):
    """Zamknięta ramka wokół osi trasy — kontur bufora o promieniu
    `buffer_pt` (w obu kierunkach od osi).

    Efekt: wszystkie działki dotykające trasy (w pasie drogowym i obok)
    mają od zewnątrz "ścianę" która zamyka ich nieDomknięte granice.
    Buffer jest zakrzywiony razem z trasą, więc dostosowuje się do jej
    kształtu (w tym PDF-ie trasa jest L-kształtna).
    """
    segs = red_segments(red_trace_draw)
    if not segs:
        return []
    # złóż wszystkie odcinki w jeden MultiLineString i zbuforuj
    lines = [LineString([p1, p2]) for p1, p2 in segs if p1 != p2]
    mls = unary_union(MultiLineString(lines))
    buf = mls.buffer(buffer_pt, cap_style=2, join_style=2)  # flat/mitre
    # pobierz zewnętrzny kontur jako linie
    out = []
    geoms = [buf] if buf.geom_type == "Polygon" else list(buf.geoms)
    for g in geoms:
        ring = g.exterior.simplify(simplify_tol)
        coords = list(ring.coords)
        for p1, p2 in zip(coords[:-1], coords[1:]):
            if p1 != p2:
                out.append((p1, p2))
    return out


def route_end_closures(red_trace_draw, *,
                       half_length_pt: float = 200.0,
                       tangent_radius_pt: float = 30.0):
    """Dla LEWEGO i PRAWEGO końca trasy: prostopadła linia o długości
    2×half_length_pt, przechodząca przez końcowy punkt trasy.

    Zamyka działki brzegowe na samych końcach trasy — kluczowe dla działek
    takich jak 391 (lewy początek) czy 283 (prawy koniec), których granice
    pionowe się nie spotykają przez BRAK POPRZECZNEJ LINII ZAMYKAJĄCEJ.
    """
    segs = red_segments(red_trace_draw)
    if not segs:
        return []
    # znajdź skrajnie lewy i prawy punkt trasy
    all_pts = []
    for (p1, p2) in segs:
        all_pts.append(p1)
        all_pts.append(p2)
    all_pts.sort()
    left = all_pts[0]
    right = all_pts[-1]
    out = []
    for (x, y) in (left, right):
        tan = trace_tangent_at(segs, x, y, r=tangent_radius_pt)
        if tan is None:
            continue
        nx, ny = -tan[1], tan[0]  # prostopadły
        p1 = (x + nx * half_length_pt, y + ny * half_length_pt)
        p2 = (x - nx * half_length_pt, y - ny * half_length_pt)
        out.append((p1, p2))
    return out


def build_polygons(all_segments, *, snap_tol: float) -> list[Polygon]:
    lines = [LineString([p1, p2]) for p1, p2 in all_segments if p1 != p2]
    if not lines:
        return []
    mls = MultiLineString(lines)
    if snap_tol > 0:
        mls = snap(mls, mls, snap_tol)
    noded = unary_union(mls)
    polys = [p for p in polygonize(noded) if p.is_valid and p.area > 0]
    return polys


def build_red_union(red_trace_draw, *, buffer_pt: float = 0.0):
    """Unia odcinków trasy; opcjonalnie z buforem jeśli buffer_pt>0."""
    segs = []
    for d in red_trace_draw:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                segs.append(LineString([p1, p2]))
    if not segs:
        return LineString()
    u = unary_union(MultiLineString(segs))
    if buffer_pt > 0:
        u = u.buffer(buffer_pt, cap_style=2, join_style=2)
    return u


def detect_road_corridor_polygon(red_trace_draw, greens, *,
                                   min_line_length_pt: float = 30.0,
                                   max_line_dist_to_route_pt: float = 120.0,
                                   corridor_buffer_pad_pt: float = 4.0,
                                   angle_tol_deg: float = 15.0):
    """Wykryj polygon "pasa drogowego" jako obszar pomiędzy zewnętrznymi
    równoległymi zielonymi liniami biegnącymi blisko trasy.

    Zwraca: Polygon (Shapely) lub None jeśli korytarz niewykryty.

    Algorytm:
      1. Zbierz długie zielone polylines blisko trasy.
      2. Zgrupuj je po kierunku (angle bucket) — korytarz tworzą linie o
         PODOBNYM kierunku.
      3. Dla dominującej grupy, znajdź line KRAŃCOWE po obu stronach trasy
         (max projection na vektor PROSTOPADŁY do kierunku grupy).
      4. Zbuduj polygon: outer_upper ∪ connector_right ∪ reverse(outer_lower) ∪ connector_left.
    """
    import math
    import numpy as np
    segs_red = red_segments(red_trace_draw)
    if not segs_red:
        return None
    mls_red = unary_union(MultiLineString(
        [LineString([p1, p2]) for p1, p2 in segs_red if p1 != p2]
    ))
    cent = mls_red.centroid
    pls = _extract_long_green_polylines(greens, min_length_pt=min_line_length_pt)
    near = [pl for pl in pls if pl["line"].distance(cent) <= max_line_dist_to_route_pt]
    if len(near) < 2:
        return None
    # Grupuj po kierunku
    def _ang(d):
        return math.degrees(math.atan2(d[1], d[0])) % 180.0
    # Dominujący kierunek — ważony długością
    # Weź największą grupę w bucketach 10°
    buckets = {}
    for pl in near:
        a = _ang(pl["dir"])
        b = int(a // 10)
        buckets.setdefault(b, []).append(pl)
    # Znajdź bucket z największą sumą długości (z uwzględnieniem sąsiadów ±1)
    def _group_length(b):
        return sum(p["length"] for bb in (b - 1, b, b + 1) for p in buckets.get(bb % 18, []))
    if not buckets:
        return None
    best_b = max(buckets.keys(), key=_group_length)
    group = [p for bb in (best_b - 1, best_b, best_b + 1) for p in buckets.get(bb % 18, [])]
    if len(group) < 2:
        return None
    # Kierunek grupy — średni, zorientowany
    vsum = np.zeros(2)
    for pl in group:
        v = np.array(pl["dir"]) * pl["length"]
        if vsum.dot(v) < 0:
            v = -v
        vsum += v
    if np.linalg.norm(vsum) < 1e-6:
        return None
    d_main = vsum / np.linalg.norm(vsum)
    d_perp = np.array([-d_main[1], d_main[0]])
    # Filtruj do tylko DŁUGICH linii (>= 50pt) — krótkie slivery nie są
    # pewnymi granicami korytarza.
    group_long = [pl for pl in group if pl["length"] >= 50.0]
    if len(group_long) < 2:
        return None
    # Dla każdej LONG linii policz projekcję centroidu na d_perp.
    perp_values = []
    for pl in group_long:
        mc = pl["line"].centroid
        perp_values.append((float(np.array([mc.x, mc.y]).dot(d_perp)), pl))
    perp_values.sort(key=lambda t: t[0])
    perp_min = perp_values[0][0]
    perp_max = perp_values[-1][0]
    perp_span = perp_max - perp_min
    if perp_span < 20.0 or perp_span > 250.0:
        return None
    # Klastruj linie po perp (1D clustering): grupy w odległości <= 15pt są
    # "tą samą granicą" (np. dwie kolineracjne krótsze linie tworzące długą
    # przerywaną granicę korytarza).
    clusters = []
    for pv, pl in perp_values:
        if clusters and abs(pv - clusters[-1][-1][0]) <= 15.0:
            clusters[-1].append((pv, pl))
        else:
            clusters.append([(pv, pl)])
    # Weź krańcowe klastry (min perp i max perp)
    neg_cluster = clusters[0]
    pos_cluster = clusters[-1]
    # Weź od KAŻDEGO klastra wszystkie linie, zbierz wszystkie punkty
    # i posortuj po projekcji na d_main → czyni envelope wzdłuż kierunku
    # korytarza.
    def _envelope(cluster):
        pts = []
        for _, pl in cluster:
            pts.extend(list(pl["line"].coords))
        if not pts:
            return []
        # Projekcja punktów na d_main (oś korytarza)
        sorted_pts = sorted(pts, key=lambda p: p[0] * d_main[0] + p[1] * d_main[1])
        # Usuń bliskie duplikaty
        dedup = []
        for p in sorted_pts:
            if not dedup or math.hypot(p[0]-dedup[-1][0], p[1]-dedup[-1][1]) > 2.0:
                dedup.append(p)
        return dedup
    up_coords = _envelope(pos_cluster)
    lo_coords = _envelope(neg_cluster)
    if len(up_coords) < 2 or len(lo_coords) < 2:
        return None
    # PRZEDŁUŻ oba envelopy wzdłuż d_main aby pokryły cały zakres osi
    # trasy (route bbox rzutowany na d_main). Bez tego korytarz jest
    # ograniczony do zakresu X long-linie, co nie obejmuje końców trasy
    # które wykraczają poza te linie (Agatowa).
    segs_red_for_ext = red_segments(red_trace_draw)
    route_pts = [p for (p1, p2) in segs_red_for_ext for p in (p1, p2)]
    if route_pts:
        route_projs = [p[0] * d_main[0] + p[1] * d_main[1] for p in route_pts]
        # Margines: korytarz przedłużony ~50pt poza zakres trasy aby
        # uchwycić działki drogowe położone TUŻ ZA fizycznym końcem trasy
        # (np. Agatowa 180/18 — 47pt za prawym końcem).
        margin = 50.0
        route_proj_min = min(route_projs) - margin
        route_proj_max = max(route_projs) + margin

        def _extend_envelope(coords, route_min, route_max):
            """Przedłuż envelope na osiach d_main do pokrycia route_min..route_max."""
            if len(coords) < 2:
                return coords
            # sort is already done in _envelope
            first, last = coords[0], coords[-1]
            # projekcja pierwszego i ostatniego punktu na d_main
            first_proj = first[0]*d_main[0] + first[1]*d_main[1]
            last_proj = last[0]*d_main[0] + last[1]*d_main[1]
            # Wyznacz kierunek envelope (od first do last)
            extended = list(coords)
            # Przedłuż przed pierwszym punktem jeśli trzeba
            if first_proj > route_min:
                dt = first_proj - route_min
                new_pt = (first[0] - d_main[0] * dt, first[1] - d_main[1] * dt)
                extended = [new_pt] + extended
            # Przedłuż po ostatnim punkcie
            if last_proj < route_max:
                dt = route_max - last_proj
                new_pt = (last[0] + d_main[0] * dt, last[1] + d_main[1] * dt)
                extended = extended + [new_pt]
            return extended
        up_coords = _extend_envelope(up_coords, route_proj_min, route_proj_max)
        lo_coords = _extend_envelope(lo_coords, route_proj_min, route_proj_max)
    # Ring: upper (od lewej do prawej) + lower (od prawej do lewej)
    ring = list(up_coords) + list(reversed(lo_coords))
    if len(ring) < 3:
        return None
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid:
            return None
        if corridor_buffer_pad_pt > 0:
            poly = poly.buffer(corridor_buffer_pad_pt)
        return poly
    except Exception:
        return None


def extend_red_along_corridor(red_trace_draw, greens, *,
                               corridor_min_length_pt: float = 30.0,
                               corridor_max_dist_pt: float = 120.0,
                               extend_pt: float = 1500.0):
    """Przedłuż oś trasy wzdłuż kierunku korytarza drogowego, jeśli taki
    został wykryty (para równoległych zielonych linii blisko trasy).

    Motywacja: na małych mapach (Fabryczna, Agatowa) czerwona linia pokazuje
    tylko mały fragment pasa drogowego, a działki drogowe rozmieszczone są
    wzdłuż całej długości korytarza. Dla klasyfikacji musimy "widzieć" pełny
    zakres korytarza.

    Algorytm:
      1. Zbierz długie ciągłe zielone polylines (≥ corridor_min_length_pt)
         blisko trasy (≤ corridor_max_dist_pt od centroidu trasy).
      2. Ich dominujący kierunek = kierunek korytarza.
      3. Centroid trasy ± kierunek × extend_pt = przedłużona oś.
      4. Zwraca MultiLineString / LineString z przedłużoną osią, lub
         oryginalną unię jeśli korytarz niewykryty.
    """
    import math
    import numpy as np
    segs_red = red_segments(red_trace_draw)
    if not segs_red:
        return None, None
    mls_red = unary_union(MultiLineString(
        [LineString([p1, p2]) for p1, p2 in segs_red if p1 != p2]
    ))
    pls = _extract_long_green_polylines(greens, min_length_pt=corridor_min_length_pt)
    # tylko te blisko trasy
    cent = mls_red.centroid
    near = []
    for pl in pls:
        if pl["line"].distance(cent) <= corridor_max_dist_pt:
            near.append(pl)
    if len(near) < 2:
        return mls_red, cent  # bez korytarza — zwracamy oryginalną trasę
    # Dominujący kierunek: średnia ważona długością (oriented to common direction)
    vsum = np.zeros(2)
    for pl in near:
        v = np.array(pl["dir"]) * pl["length"]
        if vsum.dot(v) < 0:
            v = -v
        vsum += v
    if np.linalg.norm(vsum) < 1e-6:
        return mls_red, cent
    direction = vsum / np.linalg.norm(vsum)
    # Extended axis through centroid
    cx, cy = cent.x, cent.y
    p1 = (cx - direction[0] * extend_pt, cy - direction[1] * extend_pt)
    p2 = (cx + direction[0] * extend_pt, cy + direction[1] * extend_pt)
    # Clip to the union of near polylines' bounds (żebyśmy nie leveled się
    # poza rzeczywisty zasięg korytarza widoczny na mapie)
    xs = [c for pl in near for c in (pl["start"][0], pl["end"][0])]
    ys = [c for pl in near for c in (pl["start"][1], pl["end"][1])]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = 20.0
    def _clip_line(p1, p2, xmin, ymin, xmax, ymax):
        # Liang-Barsky clipping do bboxu
        x1, y1 = p1; x2, y2 = p2
        dx = x2 - x1; dy = y2 - y1
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x1 - xmin), (dx, xmax - x1),
                     (-dy, y1 - ymin), (dy, ymax - y1)):
            if abs(p) < 1e-9:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                if t > t1: return None
                if t > t0: t0 = t
            else:
                if t < t0: return None
                if t < t1: t1 = t
        return ((x1 + t0 * dx, y1 + t0 * dy),
                (x1 + t1 * dx, y1 + t1 * dy))
    clipped = _clip_line(p1, p2,
                          xmin - margin, ymin - margin,
                          xmax + margin, ymax + margin)
    if clipped is None:
        return mls_red, cent
    # Złóż oryginalną trasę z przedłużoną osią w jedno MultiLineString
    extended = LineString([clipped[0], clipped[1]])
    union = unary_union([mls_red, extended])
    return union, cent


def analyze(
    pdf_path: str | Path,
    *,
    scale: float = 4.0,
    line_thickness: int = 2,
    ocr_cache: str | Path | None = None,
    ocr_scale: int = 8,
    adaptive: bool = False,
    tj_extend_pt: float = 40.0,
    pin_crossline_half_pt: float = 25.0,
    endpoint_close_pt: float = 5.0,
    endpoint_extend_pt: float = 80.0,
    parallel_close_pt: float = 120.0,
    parallel_cluster_r_pt: float = 15.0,
    route_end_half_pt: float = 250.0,
    route_buffer_pt: float = 100.0,
    map_hull_dilate_pt: float = 40.0,
    enable_map_hull: bool = False,
    snap_tol: float = 3.0,
    interior_buffer_pt: float = 0.5,
    interior_len_min_pt: float = 0.5,
    max_poly_area_pt2: float = 1e6,
    ray_backup: bool = True,
    ray_backup_n_rays: int = 72,
    ray_backup_max_pt: float = 110.0,
    ray_backup_min_poly_area: float = 500_000.0,
    ray_backup_min_d_trace: float = 85.0,
    ray_backup_max_d_trace: float = 200.0,
    red_buffer_pt: float = 2.0,
) -> Result:
    log = logging.getLogger("analyze_hybrid")
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    greens, reds_trace, red_pins = extract_paths(page)
    log.info("paths: green=%d red_trace=%d red_pins=%d",
             len(greens), len(reds_trace), len(red_pins))

    W_pt, H_pt = page.rect.width, page.rect.height

    # Adaptive parameters — domyślnie strojone na 03 PZT (6707×825). Dla
    # mniejszych map (np. 1191×842) skalujemy proporcjonalnie do rozmiaru
    # bbox trasy (jeśli znany) lub strony.
    if adaptive:
        from analyze_ray import build_red_union as _ray_red
        ru_tmp = _ray_red(reds_trace, buffer_pt=0)
        if not ru_tmp.is_empty:
            rb = ru_tmp.bounds
            route_extent = max(rb[2] - rb[0], rb[3] - rb[1])
        else:
            route_extent = max(W_pt, H_pt)
        # 03 PZT trasa ma route_extent ~5650pt — to nasz "referencyjny" case
        scale_f = max(0.15, min(2.0, route_extent / 5000.0))
        snap_tol = snap_tol * scale_f
        tj_extend_pt = tj_extend_pt * scale_f
        pin_crossline_half_pt = pin_crossline_half_pt * scale_f
        endpoint_close_pt = endpoint_close_pt * scale_f
        endpoint_extend_pt = endpoint_extend_pt * scale_f
        parallel_close_pt = parallel_close_pt * scale_f
        parallel_cluster_r_pt = parallel_cluster_r_pt * scale_f
        route_end_half_pt = route_end_half_pt * scale_f
        route_buffer_pt = route_buffer_pt * scale_f
        red_buffer_pt = max(0.5, red_buffer_pt * scale_f)
        log.info("adaptive: route_extent=%.0fpt scale_f=%.2f", route_extent, scale_f)

    ctx = RasterCtx(scale=scale,
                    width_px=int(round(W_pt * scale)),
                    height_px=int(round(H_pt * scale)))

    # 1. Vector zielonych
    segs = green_segments_vec(greens)
    log.info("green segments: %d", len(segs))

    # 2. Endpoint closures (pair free skeleton endpoints)
    ep_segs = find_endpoint_closures(
        greens, ctx,
        line_thickness=line_thickness,
        max_gap_pt=endpoint_close_pt,
    )
    log.info("endpoint closures: %d", len(ep_segs))
    segs.extend(ep_segs)

    # 3. Endpoint extensions (extend free endpoints in tangent direction
    # until hitting another line — zamyka jednostronnie otwarte odnogi
    # typu "pionowa linia wystająca z dolnej granicy ale nie dochodząca
    # do górnej równoległej linii").
    ext_segs = find_endpoint_extensions(
        greens, ctx,
        line_thickness=line_thickness,
        max_extend_pt=endpoint_extend_pt,
    )
    log.info("endpoint extensions: %d", len(ext_segs))
    segs.extend(ext_segs)

    # 3b. PARALLEL endpoint closures: łącz końcówki równoległych odcinków
    # które mają wolne końce naprzeciw siebie (np. dwie pionowe linie
    # kończące się w powietrzu, których połączenie poprzecznym odcinkiem
    # zamyka polygon).
    par_segs = find_parallel_endpoint_closures(
        greens, ctx,
        line_thickness=line_thickness,
        max_gap_pt=parallel_close_pt,
        cluster_r_pt=parallel_cluster_r_pt,
    )
    log.info("parallel endpoint closures: %d", len(par_segs))
    segs.extend(par_segs)

    # 3c. Drugi pass endpoint extensions — po dodaniu parallel closures,
    # wolne końce pionowych linii które nie dochodziły do górnej linii
    # mogą teraz trafić w nowo-dodaną poprzeczkę.
    ext2_segs = find_endpoint_extensions(
        greens, ctx,
        line_thickness=line_thickness,
        max_extend_pt=endpoint_extend_pt,
        extra_segments=par_segs,
    )
    log.info("endpoint extensions (pass 2): %d", len(ext2_segs))
    segs.extend(ext2_segs)


    # 4. T-junction extensions
    tj_segs = find_tjunction_extensions(
        greens, ctx,
        line_thickness=line_thickness,
        max_extend_pt=tj_extend_pt,
    )
    log.info("T-junction extensions: %d", len(tj_segs))
    segs.extend(tj_segs)

    # 5. Pin crosslines
    pin_segs = pin_crossline_segments(
        red_pins, reds_trace,
        half_length_pt=pin_crossline_half_pt,
    )
    log.info("pin crosslines: %d", len(pin_segs))
    segs.extend(pin_segs)

    # 6. Linie zamykające działki brzegowe na KOŃCACH trasy (prostopadłe
    # do lokalnego kierunku trasy w lewym/prawym punkcie końcowym).
    end_segs = route_end_closures(reds_trace, half_length_pt=route_end_half_pt)
    log.info("route-end closures: %d", len(end_segs))
    segs.extend(end_segs)

    # 6b. ROAD CORRIDOR closures — przedłużenie długich zielonych linii
    # biegnących równolegle do trasy aż do krawędzi strony. Zamyka
    # nieDomknięty "pas drogowy" (road strip) na jego lewym/prawym końcu,
    # który normalnie wycieka do tła bo linie flankujące trasę kończą
    # się w powietrzu (nie dochodzą do brzegu mapy). Kluczowe dla map
    # typu Fabryczna gdzie etykiety działek drogowych (470, 423) siedzą
    # w leaked-background polygonie.
    # wyłączone — zastąpione przez extended_red_union poniżej
    # (zachowane w kodzie na wypadek późniejszych eksperymentów)
    corr_segs = []
    log.info("parallel line pair closures: %d (disabled)", len(corr_segs))

    # 7. Ramka DOOKOŁA TRASY — buffer osi trasy o `route_buffer_pt` tworzy
    # zamkniętą pętlę zewnętrzną, która zamyka nieDomknięte działki.
    if route_buffer_pt > 0:
        buf_segs = route_buffer_frame(reds_trace, buffer_pt=route_buffer_pt)
        log.info("route buffer frame: %d", len(buf_segs))
        segs.extend(buf_segs)

    # 8. OBWÓDKA MAPY — concave hull wokół wszystkich zielonych linii.
    # Łączy wszystkie brzegowe "luźne końce" w jeden domknięty kontur,
    # tak żeby polygonize mógł poprawnie domknąć działki brzegowe.
    # Dilate skalowany proporcjonalnie do rozmiaru mapy: większe mapy
    # mogą sobie pozwolić na większy bufor, małe wymagają ciaśniejszego.
    if enable_map_hull:
        page_extent = max(page.rect.width, page.rect.height)
        adaptive_dilate = max(10.0, min(map_hull_dilate_pt,
                                        page_extent / 150.0))
        hull_segs = map_hull_segments(greens, ctx,
                                      dilate_pt=adaptive_dilate)
        log.info("map hull segments: %d (dilate=%.1fpt adaptive)",
                 len(hull_segs), adaptive_dilate)
        segs.extend(hull_segs)

    # 9. Ramka strony (ostatni fallback)
    segs.extend(frame_segments(page.rect))

    # 5. Polygonize
    polys = build_polygons(segs, snap_tol=snap_tol)
    log.info("polygons: %d", len(polys))

    # odrzuć wielkie (wycieki zewnętrzne)
    polys = [p for p in polys if p.area <= max_poly_area_pt2]
    log.info("polygons after area filter: %d", len(polys))

    # 6. Red union — niebuforowane (linia) i buforowane (polygon, do łączenia
    # dziur między kreseczkami dashed-trasy).
    red_union = build_red_union(reds_trace)
    red_buf = build_red_union(reds_trace, buffer_pt=red_buffer_pt)

    # 6c. ROAD CORRIDOR POLYGON — polygon pomiędzy zewnętrznymi
    # równoległymi zielonymi liniami blisko trasy. Dla małych map
    # (Fabryczna, Agatowa) gdzie czerwona linia jest krótka fragmentem
    # a pas drogowy długi, ten polygon obejmuje wszystkie sub-działki
    # korytarza (np. 470, 423 w Fabryczna).
    corridor_poly = detect_road_corridor_polygon(
        reds_trace, greens,
        min_line_length_pt=30.0,
        max_line_dist_to_route_pt=120.0,
    )
    # GUARDY — korytarz używamy TYLKO gdy:
    #  (i) trasa jest znacznie krótsza niż korytarz (inaczej trasa sama
    #      pokrywa wszystko co trzeba, a korytarz dodaje FP — Stefczyka)
    #  (ii) trasa RZECZYWIŚCIE wchodzi w korytarz (część trasy leży
    #       wewnątrz korytarza — inaczej to NIE JEST korytarz tej trasy
    #       tylko przypadkowy road strip obok — Teligi)
    if corridor_poly is not None:
        import math
        cx0, cy0, cx1, cy1 = corridor_poly.bounds
        corridor_extent = math.hypot(cx1 - cx0, cy1 - cy0)
        rbx0, rby0, rbx1, rby1 = red_union.bounds if not red_union.is_empty else (0,0,0,0)
        route_extent = math.hypot(rbx1 - rbx0, rby1 - rby0)
        corridor_ratio = route_extent / corridor_extent if corridor_extent > 0 else 1.0
        # frakcja trasy wewnątrz korytarza (długościowo)
        route_inside_frac = 0.0
        if not red_union.is_empty and red_union.length > 0:
            inside = red_union.intersection(corridor_poly)
            inside_len = inside.length if hasattr(inside, "length") else 0
            route_inside_frac = inside_len / red_union.length
        log.info("road corridor polygon: area=%.1f route/corr=%.2f inside_frac=%.2f",
                 corridor_poly.area, corridor_ratio, route_inside_frac)
        # Disable if route ALMOST FULLY OVERLAPS corridor (Stefczyka case:
        # route enters from residential parcel and runs along road; user's
        # GT is the start parcel, not road sub-parcels).
        if route_inside_frac > 0.80:
            log.info("corridor DISABLED: route inside_frac > 0.8 (Stefczyka pattern)")
            corridor_poly = None
        elif route_inside_frac < 0.15:
            log.info("corridor DISABLED: route barely inside corridor (<15%%)")
            corridor_poly = None
    red_union_ext = red_union  # zachowaj oryginalny red do standardowej klasyfikacji

    # 7. crossed polygons:
    #   (a) TRASA PRZECINA granicę W CO NAJMNIEJ 2 miejscach — trasa wchodzi
    #       i wychodzi (lub ma wiele odrębnych "dotknięć" granicy).
    #       Pojedyncze dotknięcie = tangent, nie crossing.
    #   (b) TRASA BIEGNIE WEWNĄTRZ polygonu — interior ∩ red.length >= min.
    def _count_components(geom):
        if geom.is_empty:
            return 0
        if hasattr(geom, "geoms"):
            return len(list(geom.geoms))
        return 1

    # Wyznacz zbiór CROSSED green SEGMENTÓW (bezpośrednio wg definicji usera:
    # "każda działka ma swoje odcinki-granice, jak chociaż jeden jest
    # przecięty przez trasę to działka ma być wpisana").
    from analyze_ray import build_green_segments as _bgs, compute_crossed_greens as _ccg
    _green_segs_list = _bgs(greens)
    _red_buf_cross = red_union.buffer(red_buffer_pt, cap_style=2, join_style=2) if not red_union.is_empty else red_union
    _crossed_ids = _ccg(_green_segs_list, _red_buf_cross)
    _crossed_segments = [_green_segs_list[i] for i in _crossed_ids]
    log.info("crossed green border segments: %d / %d",
             len(_crossed_ids), len(_green_segs_list))

    # Wider route buffer for case (e) — łapie polygony których granice biegną
    # ALONG trasy (np. instalacja wzdłuż boundary między dwiema działkami).
    # Granica może być nieco offsetnięta od osi trasy (do ~5pt).
    wide_route_buf = red_union.buffer(5.0, cap_style=2, join_style=2) \
        if not red_union.is_empty else red_union

    # Endpointy trasy — TYLKO 2 GŁOBALNE skrajności (najbardziej-lewy i
    # najbardziej-prawy punkt na osi maksymalnej rozpiętości trasy).
    # Dla case (f) — działki które SCHODZĄ SIĘ Z TRASĄ NA JEJ KOŃCU.
    # NOT każdy endpoint każdego dasha — to byłyby setki punktów.
    route_endpoints = []
    if not red_union.is_empty:
        all_pts = []
        geoms = [red_union] if red_union.geom_type == "LineString" else list(red_union.geoms)
        for ls in geoms:
            for c in ls.coords:
                all_pts.append(c)
        if all_pts:
            # Wyznacz oś największej rozciągłości (PCA approx — różnica
            # między bbox dx i dy)
            xs_p = [p[0] for p in all_pts]
            ys_p = [p[1] for p in all_pts]
            dx_range = max(xs_p) - min(xs_p)
            dy_range = max(ys_p) - min(ys_p)
            if dx_range >= dy_range:
                # leftmost / rightmost
                left = min(all_pts, key=lambda p: p[0])
                right = max(all_pts, key=lambda p: p[0])
            else:
                # topmost / bottommost
                left = min(all_pts, key=lambda p: p[1])
                right = max(all_pts, key=lambda p: p[1])
            route_endpoints = [Point(left), Point(right)]

    crossed_idx = set()
    for i, p in enumerate(polys):
        if not p.is_valid or p.area < 5:
            continue
        # (a) klasyczny transwersalny cross: trasa wchodzi i wychodzi
        binter = p.boundary.intersection(red_union_ext)
        if _count_components(binter) >= 2:
            crossed_idx.add(i)
            continue
        # (b) trasa biegnie WEWNĄTRZ polygonu (interior ∩ red.length > min)
        interior = p.buffer(-interior_buffer_pt)
        if not interior.is_empty:
            inter = interior.intersection(red_union_ext)
            if hasattr(inter, "length") and inter.length >= interior_len_min_pt:
                crossed_idx.add(i)
                continue
        # (c) odcinki granicy polygonu są CROSSED przez trasę.
        # Wymaganie: SUMA długości crossed-segmentów w obrębie boundary
        # buffer ≥ 100pt — to eliminuje pojedyncze krótkie tangent-touches
        # (np. Grochowska 569: 80pt — tylko jedno krótkie dotknięcie).
        bnd_buf = p.boundary.buffer(max(0.5, red_buffer_pt))
        cum_crossed_len = 0.0
        for seg in _crossed_segments:
            if bnd_buf.covers(seg) or bnd_buf.contains(seg):
                cum_crossed_len += seg.length
        if cum_crossed_len >= 100.0:
            crossed_idx.add(i)
            continue
        # (e) BOUNDARY-ALONG-ROUTE: granica polygonu biegnie WZDŁUŻ trasy.
        # Adaptive threshold: jeśli polygon DOTYKA trasy (d ≤ 1pt), wystarcza
        # 20% obwodu w 5pt-buforze. Jeśli polygon jest OFFSETNIĘTY (d > 1pt),
        # wymagamy 25% (Kurka 456/3 ma frac=20% offset — nie GT;
        # Kurka 456/4 ma frac=28.7% offset — JEST GT).
        if not wide_route_buf.is_empty:
            try:
                wb_inter = p.boundary.intersection(wide_route_buf)
                wb_len = wb_inter.length if hasattr(wb_inter, "length") else 0
                perimeter = p.boundary.length
                if perimeter > 0 and wb_len >= 30.0:
                    frac = wb_len / perimeter
                    d_route_e = p.distance(red_union)
                    threshold = 0.20 if d_route_e <= 1.0 else 0.25
                    if frac >= threshold:
                        crossed_idx.add(i)
                        continue
            except Exception:
                pass
        # (e2) TANGENT TOUCH + CONCAVE: polygon dotyka trasy w pojedynczym
        # punkcie I jest CONCAVE (hull_ratio < 0.85). Kształt concave przy
        # tangent-touch wskazuje że polygonize utworzył NOTCH gdzie trasa
        # weszła i wyszła z działki — Shapely sees this as boundary touch
        # ale w rzeczywistości trasa fizycznie ENTERED parcel (np. Grochowska
        # 429/2 — hook trasy przechodzi przez notch działki, hull_ratio=0.75).
        # FPs typu 569, 525/1 mają hull_ratio > 0.92 — convex shape, simple
        # corner touch.
        if not binter.is_empty and p.area >= 5:
            try:
                hull = p.convex_hull
                hull_ratio = (p.area / hull.area) if hull.area > 0 else 1.0
                # boundary touches route at single point or short segment
                d_route_e2 = p.distance(red_union)
                if d_route_e2 <= 0.5 and hull_ratio < 0.85:
                    crossed_idx.add(i)
                    continue
            except Exception:
                pass
        # (f) ENDPOINT TOUCH: polygon dotyka trasy DOKŁADNIE w jej GŁOBALNYM
        # KOŃCU i jest:
        #   - albo ELONGATED (aspect ratio ≥ 5) → typowa działka drogowa
        #     przedłużająca trasę (Polna 536: 365×38, aspect 9.7)
        #   - albo USTAWIONY DOKŁADNIE PRZECIWNIE do trasy (cos ≤ -0.5)
        #     → polygon „za" endpointem, np. Laurowa 260 (cos -0.67)
        # Bez tych warunków łapaliśmy wszelkie polygony blisko endpointów
        # (Kurka 444/1 cos -0.01 aspect 1.6 — perpendicular, nie krzyżuje).
        # Limit do mniejszych polygonów: typowe działki drogowe są <20000 pt²,
        # natomiast Laurowa 16 (FP) jest w wielkim polygonie 30000 pt²
        # przy lewym brzegu mapy.
        if route_endpoints and p.area <= 20000.0:
            try:
                from shapely.ops import nearest_points
                import math as _math
                px0, py0, px1, py1 = p.bounds
                pw, ph = px1-px0, py1-py0
                aspect = max(pw, ph) / max(min(pw, ph), 1.0)
                for ep_idx, ep in enumerate(route_endpoints):
                    if p.distance(ep) > 8.0:
                        continue
                    n_poly, _ = nearest_points(p, ep)
                    v = (n_poly.x - ep.x, n_poly.y - ep.y)
                    other_ep = route_endpoints[1-ep_idx]
                    t = (other_ep.x - ep.x, other_ep.y - ep.y)
                    norm_t = _math.hypot(*t); norm_v = _math.hypot(*v)
                    if norm_t < 1e-6 or norm_v < 1e-6:
                        cos_a = -1.0  # zerowy wektor = uznajemy za "za końcem"
                    else:
                        cos_a = (v[0]*t[0] + v[1]*t[1]) / (norm_t * norm_v)
                    if aspect >= 5.0 or cos_a <= -0.5:
                        crossed_idx.add(i)
                        break
            except Exception:
                pass
    log.info("crossed polygons: %d (standard)", len(crossed_idx))

    # (d) ROAD CORRIDOR classification: polygony które są W WIĘKSZOŚCI
    # wewnątrz wykrytego korytarza drogowego = sub-działki pasa drogowego
    # (np. 470, 423 w Fabryczna). Kluczowe dla małych map gdzie drawn-red
    # to tylko mały fragment korytarza.
    if corridor_poly is not None:
        n_added_corridor = 0
        for i, p in enumerate(polys):
            if i in crossed_idx or not p.is_valid or p.area < 5:
                continue
            # Polygon jest sub-parcel korytarza jeśli:
            #  (i) jego centroid leży w korytarzu
            #  (ii) co najmniej 70% pola polygonu leży w korytarzu
            # To eliminuje sąsiednie parcele które tylko "muskają" brzeg
            try:
                inter = p.intersection(corridor_poly)
                if inter.is_empty:
                    continue
                inter_area = inter.area if hasattr(inter, "area") else 0
                if inter_area / p.area < 0.7:
                    continue
                if not corridor_poly.contains(p.centroid):
                    continue
                crossed_idx.add(i)
                n_added_corridor += 1
            except Exception:
                continue
        log.info("corridor-added crossed polygons: %d", n_added_corridor)

    # FALLBACK: jeśli NIC nie zostało oznaczone jako crossed, a trasa jest
    # W CAŁOŚCI w jakimś jednym polygonie (siedzi wewnątrz jednej dużej
    # działki — np. Teligi) — dodaj ten polygon jako crossed.
    if not crossed_idx and not red_union.is_empty:
        for i, p in enumerate(polys):
            if not p.is_valid:
                continue
            if p.covers(red_union) or p.contains(red_union):
                crossed_idx.add(i)
                log.info("FALLBACK: route fully contained in poly#%d area=%.1f", i, p.area)
                break

    # 8. OCR + match
    if ocr_cache:
        labels = load_ocr_cache(Path(ocr_cache), ocr_scale=ocr_scale)
    else:
        raise RuntimeError("Brak cache OCR.")
    log.info("OCR labels (valid): %d", len(labels))

    # również buduj LISTĘ WSZYSTKICH polygonów (nie tylko tych <= max_area) —
    # etykiety w WIELKIM zewnętrznym polygonie identyfikujemy przez area.
    all_segs = segs  # już zawiera wszystkie warstwy
    all_polys = build_polygons(all_segs, snap_tol=snap_tol)

    # Green segments + crossed borders (dla ray casting backup)
    from analyze_ray import (
        build_green_segments, build_red_union as rb_red,
        compute_crossed_greens, label_is_crossed,
    )
    from shapely.strtree import STRtree
    green_segs_list = build_green_segments(greens)
    red_buf = rb_red(reds_trace, buffer_pt=red_buffer_pt)
    crossed_green_ids = compute_crossed_greens(green_segs_list, red_buf)
    green_tree = STRtree(green_segs_list)

    crossed_labels: dict[str, tuple[Label, int]] = {}
    borderline_labels: dict[str, tuple[Label, int]] = {}
    for lbl in labels:
        pt = Point(lbl.x, lbl.y)
        # znajdź pierwszy polygon zawierający (po filtrze area)
        found_i = None
        for i, p in enumerate(polys):
            if p.contains(pt):
                found_i = i
                break
        if found_i is None:
            # Backup: ray casting, ale TYLKO jeśli etykieta jest w WIELKIM
            # (zewnętrznym) polygonie z all_polys — unikając sliverów
            # takich jak 389/390 które siedzą w małych sub-polygonach.
            if ray_backup:
                in_large = False
                for p in all_polys:
                    if p.area >= ray_backup_min_poly_area and p.contains(pt):
                        in_large = True
                        break
                if in_large:
                    # Filtr dystansowy: etykieta nie może być zbyt BLISKO
                    # trasy (wtedy jest stacją pomiarową lub sliverem w pasie)
                    # ani zbyt DALEKO (sąsiednia działka, której trasa tylko
                    # pobliscy w innej sekcji).
                    d_trace = red_buf.distance(pt)
                    if not (ray_backup_min_d_trace <= d_trace
                            <= ray_backup_max_d_trace):
                        borderline_labels.setdefault(lbl.text, (lbl, -1))
                        continue
                    is_crossed, _ = label_is_crossed(
                        pt, green_segs_list, green_tree, crossed_green_ids,
                        n_rays=ray_backup_n_rays,
                        max_ray_pt=ray_backup_max_pt,
                        all_hits=True,
                    )
                    if is_crossed:
                        prev = crossed_labels.get(lbl.text)
                        if prev is None or lbl.conf > prev[0].conf:
                            crossed_labels[lbl.text] = (lbl, -1)
                        continue
            borderline_labels.setdefault(lbl.text, (lbl, -1))
            continue
        if found_i in crossed_idx:
            prev = crossed_labels.get(lbl.text)
            if prev is None or lbl.conf > prev[0].conf:
                crossed_labels[lbl.text] = (lbl, found_i)
            continue
        # (e) CORRIDOR LABEL CLASSIFICATION: label siedzi w LEAKED polygonie
        # (polygon który nie był w crossed_idx lub był odfiltrowany przez
        # area), ale sam POINT etykiety jest w korytarzu drogowym →
        # działka drogowa (np. 470 w Fabryczna).
        if corridor_poly is not None and corridor_poly.contains(pt):
            prev = crossed_labels.get(lbl.text)
            if prev is None or lbl.conf > prev[0].conf:
                crossed_labels[lbl.text] = (lbl, -2)

    # FINAL FALLBACK: jeśli po standardowej i ray-backup klasyfikacji nic
    # nie zostało zaznaczone jako crossed, a trasa jest obecna — działka
    # najbliższa trasie (z rozsądnego dystansu) jest najprawdopodobniej
    # TĄ DZIAŁKĄ w której trasa się znajduje. To sytuacja "trasa wewnątrz
    # jednej działki bez przecinania granicy" (np. Teligi).
    if not crossed_labels and not red_union.is_empty and labels:
        route_centroid = red_union.centroid
        # policz dystans OCR etykiety → trasa i wybierz najbliższą
        scored = []
        for lbl in labels:
            d = red_union.distance(Point(lbl.x, lbl.y))
            scored.append((d, lbl))
        scored.sort()
        if scored:
            d_best, lbl = scored[0]
            # akceptuj tylko jeśli BARDZO blisko trasy (< kilkudziesięciu pt)
            if d_best < 200.0:
                crossed_labels[lbl.text] = (lbl, -1)
                log.info("FINAL FALLBACK: nearest-to-route label %s at d=%.1fpt",
                         lbl.text, d_best)

    crossed = sorted(crossed_labels.keys())
    borderline = sorted(t for t in borderline_labels if t not in crossed_labels)
    log.info("crossed=%d borderline=%d (%.1fs)",
             len(crossed), len(borderline), time.time() - t0)

    return Result(
        crossed=crossed, borderline=borderline,
        debug={
            "n_polys": len(polys),
            "n_crossed_polys": len(crossed_idx),
            "crossed_detail": [
                {"text": t, "x": l.x, "y": l.y, "poly_idx": pi}
                for t, (l, pi) in crossed_labels.items()
            ],
        },
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument("--ocr-cache", default="/tmp/ocr_cache_v2.pkl")
    p.add_argument("--tj-extend", type=float, default=25.0)
    p.add_argument("--pin-crossline-half", type=float, default=15.0)
    p.add_argument("--snap-tol", type=float, default=3.0)
    p.add_argument("--interior-buffer", type=float, default=0.5)
    p.add_argument("--max-poly-area", type=float, default=1e6)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    res = analyze(
        args.pdf, scale=args.scale,
        line_thickness=args.line_thickness,
        ocr_cache=args.ocr_cache,
        tj_extend_pt=args.tj_extend,
        pin_crossline_half_pt=args.pin_crossline_half,
        snap_tol=args.snap_tol,
        interior_buffer_pt=args.interior_buffer,
        max_poly_area_pt2=args.max_poly_area,
    )
    print(f"DZIAŁKI PRZECIĘTE ({len(res.crossed)}):")
    print("  " + (", ".join(res.crossed) if res.crossed else "—"))
    print(f"DZIAŁKI DO SPRAWDZENIA ({len(res.borderline)}):")
    print("  " + (", ".join(res.borderline) if res.borderline else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
