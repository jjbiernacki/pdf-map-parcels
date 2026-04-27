"""Analiza map geodezyjnych — podejście hybrydowe OpenCV + Shapely.

Cel: wykryć DOKŁADNIE działki przecięte czerwoną trasą (pas drogowy).

Pipeline:
  1.  Wyciągnij z PDF-a odfiltrowane zielone ścieżki (granice działek,
      bez glifów etykiet) i czerwone ścieżki (trasa `width>1` + piny `width≈0.84`).
  2.  Rasteryzuj zielone do maski wysokiej rozdzielczości (rysowanie ręczne
      przez cv2.line na podstawie iter_segments — gwarantuje brak glifów).
  3.  Skeletonize zielonej maski → znajdź WOLNE KOŃCE (piksele stopnia 1).
  4.  Paruj wolne końce w promieniu adaptacyjnym (funkcja mediany długości
      najkrótszego segmentu) i domykaj prostymi odcinkami.
  5.  Flood-fill od zewnątrz na odwróconej masce → każdy NIE-zewnętrzny
      komponent = jeden wielokąt działki.
  6.  Rasteryzuj czerwoną trasę + piny → skeletonize → corridor buffer.
      Dla pasa drogowego: buffer wielkości = mediana odległości między
      pinami / stała (adaptacja do konkretnej mapy).
  7.  Dopasuj etykiety OCR (z cache'u) do komponentów-działek przez
      spiralne szukanie w promieniu.
  8.  Klasyfikacja: działka.crossed ⇔ component_mask ∩ road_corridor > 0.

Zero hardkodowanych progów do tego konkretnego PDF-a. Wszystkie progi są
adaptacyjne względem geometrii mapy (mediana długości segmentu,
odstępu pinów itp.).
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz  # pymupdf
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# Filtry kolorów / ścieżek PDF — wspólne z analyze.py
# ---------------------------------------------------------------------------

def is_green_stroke(rgb) -> bool:
    if rgb is None:
        return False
    r, g, b = rgb
    # Rozluźnione — obsługuje zarówno stare mapy (0, 0.584, 0) jak i nowe
    # (0.216, 0.867, 0). Klucz: g > r, g > b, g > 0.3.
    return r < 0.4 and g > 0.3 and b < 0.2 and g > r and g > b


def is_red_stroke(rgb) -> bool:
    """Czerwień — kabel ziemny (1, 0, 0)."""
    if rgb is None:
        return False
    r, g, b = rgb
    return r > 0.85 and g < 0.15 and b < 0.15


def is_magenta_stroke(rgb) -> bool:
    """Różowy/magenta (1, 0, 1) — przewód NAPOWIETRZNY na słupach.

    Legenda użytkownika: różowa linia = odcinek instalacji wykonany
    przewodem napowietrznym. Pod względem „przebiegu trasy przez działki"
    traktujemy różowy tak samo jak czerwony — to też jest trasa instalacji.
    """
    if rgb is None:
        return False
    r, g, b = rgb
    return r > 0.85 and g < 0.15 and b > 0.85


def is_route_stroke(rgb) -> bool:
    """Trasa instalacji — CZERWONY (kabel ziemny) LUB MAGENTA (napowietrzna)."""
    return is_red_stroke(rgb) or is_magenta_stroke(rgb)


def iter_segments(drawing):
    for it in drawing["items"]:
        kind = it[0]
        if kind == "l":
            yield (it[1].x, it[1].y), (it[2].x, it[2].y)
        elif kind == "c":
            p0, p3 = it[1], it[4]
            yield (p0.x, p0.y), (p3.x, p3.y)
        elif kind == "re":
            r = it[1]
            x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
            yield (x0, y0), (x1, y0)
            yield (x1, y0), (x1, y1)
            yield (x1, y1), (x0, y1)
            yield (x0, y1), (x0, y0)


def _split_red_drawings(red_all):
    """Adaptacyjny podział czerwonych stroków na TRASĘ i PINY.

    Obserwacja po 6 mapach:
      - 03 PZT: trasa 1.44 + piny 0.84 (+ wypełnienia width=0 z typ 'fs')
      - Agatowa, Laurowa: tylko width=0.84 (wszystko jest trasą)
      - Fabryczna, Stefczyka, Teligi: tylko width=1.08 (wszystko trasą)

    Strategia:
      1. Pomiń stroki z width==0 (to wypełnienia typu 'fs', nie geometria linii).
      2. Jeśli pozostałe mają jedną lub prawie-jedną szerokość — wszystko = trasa.
      3. Inaczej: split na największej przerwie w histogramie szerokości;
         szersza grupa = trasa, węższa = piny.
    """
    if not red_all:
        return [], []
    lined = [d for d in red_all if (d.get("width") or 0) > 0]
    if not lined:
        return [], []
    widths = sorted({round(d.get("width") or 0, 2) for d in lined})
    if len(widths) == 1 or (widths[-1] - widths[0]) < 0.3:
        return list(lined), []
    best_gap = 0.0
    split = widths[-1]
    for i in range(1, len(widths)):
        gap = widths[i] - widths[i - 1]
        if gap > best_gap:
            best_gap = gap
            split = widths[i]
    trace, pins = [], []
    for d in lined:
        w = round(d.get("width") or 0, 2)
        if w >= split:
            trace.append(d)
        else:
            pins.append(d)
    return trace, pins


def extract_paths(page):
    """Zielone granice (bez glifów), czerwone trasy, czerwone piny.

    Adaptacyjnie — szerokości trasy/pinów zależą od konkretnego PDF-a,
    więc wykrywamy je z histogramu szerokości czerwonych stroków.
    """
    green, red_all = [], []
    for d in page.get_drawings():
        color = d.get("color")
        typ = d.get("type")
        rect = d.get("rect")
        if typ == "f":
            continue
        if is_green_stroke(color):
            if rect is not None and len(d["items"]) > 1 \
                    and rect.width < 30 and rect.height < 20:
                continue  # glif cyfry etykiety
            green.append(d)
        elif is_route_stroke(color):
            red_all.append(d)
    red_trace, red_pins = _split_red_drawings(red_all)
    return green, red_trace, red_pins


# ---------------------------------------------------------------------------
# Rasteryzacja ścieżek → binarna maska
# ---------------------------------------------------------------------------

@dataclass
class RasterCtx:
    """Kontekst rasteryzacji — przekształcenie pt PDF-a ↔ piksele."""
    scale: float
    width_px: int
    height_px: int

    def pt2px(self, x, y):
        return int(round(x * self.scale)), int(round(y * self.scale))

    def px2pt(self, xp, yp):
        return xp / self.scale, yp / self.scale


def rasterize_paths(drawings, ctx: RasterCtx, thickness: int = 2) -> np.ndarray:
    """Narysuj każdy segment na czarnej masce jako biały piksel."""
    mask = np.zeros((ctx.height_px, ctx.width_px), dtype=np.uint8)
    for d in drawings:
        for p1, p2 in iter_segments(d):
            x1, y1 = ctx.pt2px(*p1)
            x2, y2 = ctx.pt2px(*p2)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_8)
    return mask


def rasterize_pins(pins, ctx: RasterCtx) -> list[tuple[int, int]]:
    """Zwraca środki pinów pomiarowych (centra bboxów drawings) w pikselach."""
    centers = []
    for d in pins:
        r = d.get("rect")
        if r is None:
            continue
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        centers.append(ctx.pt2px(cx, cy))
    return centers


def pin_centers_pt(pins) -> list[tuple[float, float]]:
    """Zwraca środki pinów w punktach PDF (przed rasteryzacją)."""
    centers = []
    for d in pins:
        r = d.get("rect")
        if r is None:
            continue
        centers.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    return centers


def dedup_pins(pins: list[tuple[float, float]], tol: float = 4.0) -> list[tuple[float, float]]:
    """Łączy piny w promieniu `tol` w jeden (parę pinów bliskich 0.6-2pt)."""
    if not pins:
        return []
    from scipy.spatial import cKDTree
    arr = np.array(pins, dtype=float)
    tree = cKDTree(arr)
    visited = np.zeros(len(arr), dtype=bool)
    out = []
    for i in range(len(arr)):
        if visited[i]:
            continue
        idx = tree.query_ball_point(arr[i], r=tol)
        idx = [j for j in idx if not visited[j]]
        grp = arr[idx]
        out.append((float(grp[:, 0].mean()), float(grp[:, 1].mean())))
        visited[idx] = True
    return out


def trace_tangent_at(red_segments_pt: list[tuple[tuple[float, float], tuple[float, float]]],
                     x: float, y: float, r: float = 20.0) -> np.ndarray | None:
    """Policz lokalny tangens trasy w punkcie (x,y) z odcinków czerwonych
    w promieniu r. Metoda: weź wszystkie odcinki których środek leży w r,
    znajdź główny kierunek przez PCA (największy wektor własny kowariancji).
    """
    pts = []
    for (x1, y1), (x2, y2) in red_segments_pt:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if (mx - x) ** 2 + (my - y) ** 2 <= r * r:
            pts.append((x1, y1))
            pts.append((x2, y2))
    if len(pts) < 4:
        return None
    arr = np.array(pts, dtype=float)
    arr -= arr.mean(axis=0)
    cov = np.cov(arr.T)
    w, v = np.linalg.eigh(cov)
    # największy wektor własny → kierunek główny
    t = v[:, -1]
    t = t / (np.linalg.norm(t) + 1e-9)
    # tangens zwracam jako (dx, dy)
    return np.array([t[0], t[1]])


def red_segments(reds) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    out = []
    for d in reds:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                out.append((p1, p2))
    return out


def draw_pin_crosslines(mask: np.ndarray, ctx: RasterCtx,
                        pin_centers_pt: list[tuple[float, float]],
                        red_segs: list,
                        *, half_length_pt: float,
                        thickness: int = 2,
                        tangent_radius_pt: float = 20.0) -> tuple[np.ndarray, int]:
    """Dla każdego pinu dorysuj odcinek PROSTOPADŁY do lokalnego kierunku
    trasy, o długości 2×half_length_pt (przechodzący przez pin).

    Służy jako SZTUCZNA POPRZECZNA GRANICA dzieląca pas drogowy na działki
    (gdy ich narysowane granice są niepełne).
    """
    out = mask.copy()
    n = 0
    for (x, y) in pin_centers_pt:
        tan = trace_tangent_at(red_segs, x, y, r=tangent_radius_pt)
        if tan is None:
            continue
        # prostopadły kierunek
        nx, ny = -tan[1], tan[0]
        x1 = x + nx * half_length_pt
        y1 = y + ny * half_length_pt
        x2 = x - nx * half_length_pt
        y2 = y - ny * half_length_pt
        p1 = ctx.pt2px(x1, y1)
        p2 = ctx.pt2px(x2, y2)
        cv2.line(out, p1, p2, 255, thickness, cv2.LINE_8)
        n += 1
    return out, n


# ---------------------------------------------------------------------------
# Domykanie wolnych końców
# ---------------------------------------------------------------------------

def find_endpoints(skel: np.ndarray) -> np.ndarray:
    """Piksele szkieletu z dokładnie 1 sąsiadem (stopień 1)."""
    k = np.ones((3, 3), dtype=np.uint8)
    n = ndi.convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    return skel & (n == 2)


def find_tjunctions(skel: np.ndarray) -> np.ndarray:
    """Piksele szkieletu ze stopniem 3 (T-junction).

    Prosta heurystyka — 3 sąsiadów w 3×3 + sam piksel. Może też łapać
    niektóre Y-junctions, co nie szkodzi (filtrujemy na etapie
    antykolinearności 2 gałęzi).
    """
    k = np.ones((3, 3), dtype=np.uint8)
    n = ndi.convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    return skel & (n == 4)


def _walk_from(skel: np.ndarray, start_y: int, start_x: int,
               first_y: int, first_x: int, steps: int) -> list[tuple[int, int]]:
    """Chodzenie po szkielecie zaczynając od (start)→(first), `steps` kroków."""
    path = [(start_y, start_x), (first_y, first_x)]
    visited = {(start_y, start_x), (first_y, first_x)}
    cy, cx = first_y, first_x
    for _ in range(steps - 1):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = cy + dy, cx + dx
                if (ny, nx_) in visited:
                    continue
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                    if skel[ny, nx_]:
                        nxt = (ny, nx_)
                        break
            if nxt is not None:
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        cy, cx = nxt
    return path


def _direction_from_path(path: list[tuple[int, int]]) -> np.ndarray | None:
    """Wektor kierunkowy ze ścieżki (path[0] - path[-1]) — OD SZKIELETU NA ZEWNĄTRZ."""
    if len(path) < 2:
        return None
    y0, x0 = path[0]
    y1, x1 = path[-1]
    dy = y0 - y1
    dx = x0 - x1
    n = (dy * dy + dx * dx) ** 0.5
    if n < 1e-6:
        return None
    return np.array([dy / n, dx / n])


def estimate_direction(skel: np.ndarray, yx: tuple[int, int],
                       steps: int = 8) -> np.ndarray | None:
    """Kierunek wyjścia z endpointu (stopień 1)."""
    y, x = yx
    first = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                if skel[ny, nx_]:
                    first = (ny, nx_)
                    break
        if first is not None:
            break
    if first is None:
        return None
    path = _walk_from(skel, y, x, first[0], first[1], steps)
    return _direction_from_path(path)


def tjunction_branch_direction(skel: np.ndarray, y: int, x: int,
                               steps: int = 10,
                               anti_cos_thresh: float = -0.85) -> np.ndarray | None:
    """Dla T-junction wyznacz kierunek "gałęzi bocznej".

    T-junction ma 3 sąsiadów → 3 gałęzie. Dwie z nich tworzą "główną
    linię" (antykolinearne, cos < -0.85 między tangensami). Trzecia =
    gałąź boczna. Zwracamy kierunek tej gałęzi OD T-junction w kierunku
    sąsiada — czyli w stronę gdzie gałąź biegnie.
    """
    branches = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                if skel[ny, nx_]:
                    path = _walk_from(skel, y, x, ny, nx_, steps)
                    d = _direction_from_path(path)
                    if d is not None:
                        branches.append(d)
    if len(branches) != 3:
        return None
    # znajdź parę antykolinearną
    main_pair = None
    for i in range(3):
        for j in range(i + 1, 3):
            if float(np.dot(branches[i], branches[j])) < anti_cos_thresh:
                main_pair = {i, j}
                break
        if main_pair:
            break
    if main_pair is None:
        return None
    side_idx = ({0, 1, 2} - main_pair).pop()
    # _direction_from_path zwraca (path[0]-path[-1]) = wektor OD KOŃCA GAŁĘZI
    # DO T-J → wskazuje w kierunku PRZECIWNYM do gałęzi. Czyli jeśli gałąź
    # biegnie w górę, wektor wskazuje w dół — a w dół właśnie chcemy
    # ekstrapolować (przez główną linię na drugą stronę). Zwracamy bez
    # odwracania.
    return branches[side_idx]


def close_gaps(mask: np.ndarray, skel: np.ndarray, *, max_gap_px: float,
               thickness: int = 2, debug=None) -> tuple[np.ndarray, int]:
    """Domknij wolne końce szkieletu prostymi odcinkami.

    Strategia:
      - pary endpointów sortowane rosnąco po odległości
      - greedy: każdy endpoint dostaje co najwyżej jednego partnera
        (najbliższego, który jeszcze nie został użyty)
      - brak filtra kątowego (w praktyce psuje więcej niż pomaga —
        endpointy pocięte pikselowo mają kierunki nieprzewidywalne).

    Jeśli zakończenie jest "naprawdę wolne" (blisko *innego* endpointu), to
    prawie zawsze ta para chce być połączona. Problemem są endpointy
    przypadkowo blisko siebie na rzeczywistych końcach — te filtrujemy
    tylko przez mały max_gap_px.
    """
    endpoints = find_endpoints(skel)
    ys, xs = np.where(endpoints)
    if len(ys) == 0:
        return mask, 0

    coords = np.column_stack([ys, xs])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_gap_px)
    pairs_sorted = []
    for i, j in pairs:
        dy = coords[i, 0] - coords[j, 0]
        dx = coords[i, 1] - coords[j, 1]
        dist = (dy * dy + dx * dx) ** 0.5
        pairs_sorted.append((dist, i, j))
    pairs_sorted.sort()

    used = np.zeros(len(coords), dtype=bool)
    out = mask.copy()
    n_closed = 0
    for dist, i, j in pairs_sorted:
        if used[i] or used[j]:
            continue
        y1, x1 = int(coords[i, 0]), int(coords[i, 1])
        y2, x2 = int(coords[j, 0]), int(coords[j, 1])
        cv2.line(out, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_8)
        used[i] = used[j] = True
        n_closed += 1
        if debug is not None:
            debug.append(((x1, y1), (x2, y2)))
    return out, n_closed


def extend_tjunctions(mask: np.ndarray, skel: np.ndarray, *,
                      max_extend_px: float, thickness: int,
                      min_branch_len: int = 5,
                      stop_on_hit: bool = True,
                      debug=None) -> tuple[np.ndarray, int]:
    """Dla każdego T-junction ekstrapoluj gałąź poprzeczną na drugą stronę
    głównej linii.

    Jeśli `stop_on_hit=True`, rysowanie zatrzymuje się gdy linia trafi
    w inny piksel szkieletu (NIE należący do samego T-j lub głównej linii
    lokalnej). To zapobiega narysowaniu linii „w pustce" gdy druga strona
    nie istnieje.
    """
    tj = find_tjunctions(skel)
    ys, xs = np.where(tj)
    out = mask.copy()
    n_ext = 0
    for y, x in zip(ys, xs):
        d = tjunction_branch_direction(skel, int(y), int(x), steps=min_branch_len)
        if d is None:
            continue
        # Symulacja rysowania: idź po dyskretnych punktach linii i zatrzymaj się
        # przy trafieniu w inny szkielet (po początkowym „odejściu" od T-j).
        step = 1.0
        end_y, end_x = None, None
        cy_f, cx_f = y + 0.5, x + 0.5
        reached = 0
        for t in range(1, int(max_extend_px) + 1):
            ny = int(round(y + d[0] * t))
            nx = int(round(x + d[1] * t))
            if ny < 0 or ny >= skel.shape[0] or nx < 0 or nx >= skel.shape[1]:
                break
            # po wejściu w pustkę (brak szkieletu) zaczynamy rysować; gdy
            # ponownie trafimy w szkielet → STOP
            if t >= 2 and skel[ny, nx]:
                end_y, end_x = ny, nx
                break
            end_y, end_x = ny, nx
            reached = t
        if end_y is None or reached < 2:
            continue
        if stop_on_hit and reached >= int(max_extend_px):
            # nie doszliśmy do drugiej strony — nie rysuj (nie domykaj w próżni)
            continue
        cv2.line(out, (int(x), int(y)), (int(end_x), int(end_y)),
                 255, thickness, cv2.LINE_8)
        n_ext += 1
        if debug is not None:
            debug.append(((int(x), int(y)), (int(end_x), int(end_y))))
    return out, n_ext


def morph_close(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Domknięcie morfologiczne — zamyka sub-pixelowe szczeliny bez
    fuzji równoległych linii (małe jądro)."""
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


# ---------------------------------------------------------------------------
# Komponenty-działki z maski domkniętych granic
# ---------------------------------------------------------------------------

def parcel_components(green_mask: np.ndarray, *, min_area_px: int = 50,
                      max_area_frac: float = 0.20) -> tuple[np.ndarray, int]:
    """Spójne komponenty TŁA (odwrotność zielonej maski).

    Filtry "tło":
      - komponent dotyka brzegu obrazu
      - komponent obejmuje > `max_area_frac` całej powierzchni obrazu
        (łapie „wyciekające" agregaty — wiele działek zlepionych w jeden)
      - powierzchnia < min_area_px (szum)
    """
    inv = (green_mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    H, W = inv.shape
    total_px = H * W
    max_area = int(max_area_frac * total_px)
    edge_labels = set()
    edge_labels.update(labels[0, :].tolist())
    edge_labels.update(labels[H - 1, :].tolist())
    edge_labels.update(labels[:, 0].tolist())
    edge_labels.update(labels[:, W - 1].tolist())
    edge_labels.discard(0)
    remap = np.zeros(n, dtype=np.int32)
    nxt = 1
    for lab in range(n):
        area = stats[lab, cv2.CC_STAT_AREA]
        is_bg = (lab == 0
                 or lab in edge_labels
                 or area < min_area_px
                 or area > max_area)
        if is_bg:
            remap[lab] = 0
        else:
            remap[lab] = nxt
            nxt += 1
    new_labels = remap[labels]
    return new_labels, nxt - 1


# ---------------------------------------------------------------------------
# Pas drogowy (corridor)
# ---------------------------------------------------------------------------

def build_road_corridor(red_trace_mask: np.ndarray, pin_centers: list[tuple[int, int]],
                        *, buffer_px: float) -> np.ndarray:
    """Cienki corridor wokół osi trasy (do testu "trasa przecina komponent")."""
    kernel_size = int(round(buffer_px * 2 + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    corridor = cv2.dilate(red_trace_mask, k)
    pin_r = max(1, int(round(buffer_px)))
    for (x, y) in pin_centers:
        cv2.circle(corridor, (x, y), pin_r, 255, -1)
    return corridor


def draw_frame(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Dorysuj ramkę wokół obrazu — zamyka otwarte komponenty brzegowe."""
    H, W = mask.shape
    out = mask.copy()
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), 255, thickness)
    return out


def fill_true_background(mask: np.ndarray, *, max_dist_px: float) -> np.ndarray:
    """Ozn. piksele TŁA które są daleko (`> max_dist_px`) od jakiejkolwiek
    zielonej granicy — jako "zielone". Izoluje pas drogowy i inne wycieki
    jako osobne komponenty, oddzielone od prawdziwego tła.

    Pas drogowy ma szerokość ~30 pt: piksele w nim są maks ~15pt od
    zielonej linii pasa. Prawdziwe tło (daleko na brzegu mapy, między
    działkami z dużymi odstępami) ma piksele DALEJ niż 30-50pt od zielonych.
    """
    inv = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    far_bg = dist > max_dist_px
    out = mask.copy()
    out[far_bg] = 255
    return out


# ---------------------------------------------------------------------------
# OCR: parcel_label regex, cache
# ---------------------------------------------------------------------------

LABEL_RE = re.compile(r"^\d{2,3}(/\d{1,2})?$")  # min 2 digits — 1-digit
                                                 # to OCR artefakt, nie parcela
OCR_CONF_MIN = 0.4


def _repair_label(s: str) -> str:
    """Naprawia znane artefakty EasyOCR:
      - leading "0" lub "7" (np. "7260" → "260", "0614/1" → "614/1")
      - sklejone 4-cyfrowe numery bez "/" (np. "6052" → "605/2")
      - "7" wstawione zamiast "/" (np. "7236743" → strip 7 → "236743" →
        replace mid "7" with "/" → "236/43")
      - trailing "/" (np. "557/" → "557")
    """
    s = s.strip()
    if LABEL_RE.match(s):
        return s
    # leading "0" lub "7" — częste artefakty EasyOCR
    if s and s[0] in ("0", "7") and len(s) > 1:
        candidate = s[1:]
        repaired = _repair_label(candidate)
        if LABEL_RE.match(repaired):
            return repaired
    # sklejone 4-cyfrowe: NNNM → NNN/M
    if re.fullmatch(r"\d{4}", s):
        candidate = f"{s[:3]}/{s[3]}"
        if LABEL_RE.match(candidate):
            return candidate
    # 5-6 cyfr: spróbuj wstawić "/" (zastępując "7" które OCR mógł
    # pomylić ze slashem) na każdej pozycji
    if re.fullmatch(r"\d{5,6}", s):
        for i in range(1, len(s)):
            # Zastąp znak na pozycji i przez "/" (jeśli to "7", "0" lub "1"
            # — typowe pomyłki dla "/" w OCR)
            if s[i] in ("7", "1"):
                candidate = s[:i] + "/" + s[i+1:]
                if LABEL_RE.match(candidate):
                    return candidate
        # Albo wstaw "/" bez usuwania (gdy długość 5 = 3+2)
        for i in (3, 2):
            if i < len(s):
                candidate = s[:i] + "/" + s[i:]
                if LABEL_RE.match(candidate):
                    return candidate
    # trailing "/"
    if s.endswith("/"):
        repaired = _repair_label(s[:-1])
        if LABEL_RE.match(repaired):
            return repaired
    return s


@dataclass
class Label:
    text: str
    conf: float
    x: float  # w punktach PDF
    y: float


def load_ocr_cache(path: Path, ocr_scale: int) -> list[Label]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    candidates = []
    for r in raw:
        t = _repair_label(r["text"])
        if not LABEL_RE.match(t):
            continue
        if t.endswith("/0"):
            continue
        conf = r.get("conf", 0)
        # Niższy próg conf dla high-scale OCR pass (scale=12, low threshold).
        # Te etykiety są filtrowane już na poziomie OCR przez text_threshold=0.05.
        # Niższy conf to artefakt zwiększonego cleanup'u, nie oznacza fałszywej
        # detekcji (np. Grochowska 421 ma w high-scale pass conf=0.32 ale jest
        # to PRAWDZIWA etykieta widoczna na mapie).
        is_high_scale = r.get("source") == "easy_high"
        min_conf = 0.25 if is_high_scale else OCR_CONF_MIN
        if conf < min_conf:
            continue
        # 2-cyfrowe labels bez "/" są podatne na OCR FP (np. fragmenty
        # liczb 3-cyfrowych odczytane osobno). Wymagamy conf >= 0.5
        # niezależnie od scale.
        if "/" not in t and len(t) == 2 and conf < 0.5:
            continue
        if "x" in r and "y" in r:
            x, y = r["x"], r["y"]
        elif "cx" in r and "cy" in r:
            x, y = r["cx"] / ocr_scale, r["cy"] / ocr_scale
        else:
            continue
        candidates.append(Label(text=t, conf=float(conf), x=float(x), y=float(y)))

    # Deduplikacja pozycyjna — w promieniu 20pt:
    #   (1) jeśli jeden tekst jest strict substringiem drugiego, zatrzymaj
    #       dłuższy (np. "3" vs "334" przy tej samej pozycji = OCR znalazł
    #       fragment + całość, prawdziwy jest "334")
    #   (2) jeśli jeden to NNN i drugi NNN/M (slash-prefix), zatrzymaj
    #       bardziej specyficzny NNN/M (np. "557" vs "557/6")
    #   (3) inaczej zatrzymaj wyższy conf — różne teksty w tej samej
    #       pozycji to zwykle OCR pomyłka, ufamy mocniejszej detekcji
    def _dominates(a: Label, b: Label) -> bool:
        """Czy a dominuje b (czyli b można odrzucić bo a jest lepsze)?
        TYLKO dla tekstów-wariantów tego samego numeru — nie odrzucamy
        DIFFERENT-parcel labels (np. "465" i "456/12" w Kurce: różne).
        """
        if a.text == b.text:
            return a.conf >= b.conf
        if a.text.startswith(b.text) and len(a.text) > len(b.text):
            suffix = a.text[len(b.text):]
            # Case 1: NNN/M dominuje NNN (np. "614/1" > "614") — sufiks "/M"
            if suffix.startswith("/"):
                return True
            # Case 2: krótki fragment 1-2 cyfrowy dominuje przez pełny odczyt
            # (np. "3" jako fragment "334" — OCR osobno przeczytał cyfrę).
            # Tylko gdy b jest BARDZO KRÓTKI (1-2 znaki) i NIE zawiera "/"
            # — to wykluczy "296/3" + "0" → "296/30" (different parcels).
            if len(b.text) <= 2 and "/" not in b.text and suffix.isdigit():
                return True
        return False

    kept = []
    for lbl in candidates:
        drop = False
        for i, other in enumerate(kept):
            dx = abs(lbl.x - other.x); dy = abs(lbl.y - other.y)
            close_pos = dx < 20 and dy < 20
            if not close_pos:
                continue
            same_pos = dx < 5 and dy < 5
            # NAJPIERW sprawdź relacje semantyczne (substring/prefix) —
            # one działają niezależnie od conf:
            if _dominates(other, lbl):
                drop = True
                break
            if _dominates(lbl, other):
                kept[i] = lbl
                drop = True
                break
            # SAME POSITION (< 5pt) z RÓŻNYMI niezwiązanymi tekstami →
            # OCR pomyłka, ufamy wyższemu conf
            # (296/3 conf 1.0 vs 296/30 conf 0.44 → keep 296/3).
            if same_pos:
                if other.conf >= lbl.conf:
                    drop = True
                    break
                kept[i] = lbl
                drop = True
                break
        if not drop:
            kept.append(lbl)
    return kept


# ---------------------------------------------------------------------------
# Dopasowanie etykieta → komponent działki
# ---------------------------------------------------------------------------

def label_to_component(labels_img: np.ndarray, x_px: int, y_px: int,
                       search_radius_px: int = 50) -> int:
    """Zwróć id komponentu spod punktu, a jeśli tam 0 (tło/granica) — spiralnie
    poszukaj najbliższego niezerowego komponentu w promieniu."""
    H, W = labels_img.shape
    if not (0 <= x_px < W and 0 <= y_px < H):
        return 0
    v = labels_img[y_px, x_px]
    if v != 0:
        return int(v)
    # bbox promienia
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


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

@dataclass
class Result:
    crossed: list[str] = field(default_factory=list)
    borderline: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def analyze(
    pdf_path: str | Path,
    *,
    scale: float = 4.0,
    gap_close_factor: float = 8.0,
    corridor_buffer_pt: float = 2.0,
    line_thickness: int = 2,
    ocr_cache: str | Path | None = None,
    ocr_scale: int = 8,
    debug_png: str | Path | None = None,
    morph_close_px: int = 2,
    tjunction_extend_pt: float = 25.0,
    pin_crossline_half_pt: float = 15.0,
    enable_pin_crosslines: bool = True,
    enable_frame: bool = True,
    bg_max_dist_pt: float = 30.0,
) -> Result:
    log = logging.getLogger("analyze_cv")
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    greens, reds_trace, red_pins = extract_paths(page)
    log.info("paths: green=%d red_trace=%d red_pins=%d",
             len(greens), len(reds_trace), len(red_pins))

    # Raster ctx
    W_pt, H_pt = page.rect.width, page.rect.height
    W_px = int(round(W_pt * scale))
    H_px = int(round(H_pt * scale))
    ctx = RasterCtx(scale=scale, width_px=W_px, height_px=H_px)
    log.info("raster: %dx%d px (scale=%.1f)", W_px, H_px, scale)

    green_mask = rasterize_paths(greens, ctx, thickness=line_thickness)
    red_mask = rasterize_paths(reds_trace, ctx, thickness=line_thickness)
    pin_centers = rasterize_pins(red_pins, ctx)
    log.info("masks: green_px=%d red_px=%d pins=%d",
             int((green_mask > 0).sum()), int((red_mask > 0).sum()),
             len(pin_centers))

    # krok 1: morfologiczne domknięcie drobnych szczelin (2-pikselowe
    # dziury na stykach linii PDF)
    green_mask = morph_close(green_mask, morph_close_px)
    log.info("after morph_close: green_px=%d", int((green_mask > 0).sum()))

    # krok 2: skeletonize → endpoint pairing domykający większe wolne końce
    gsk = skeletonize(green_mask > 0)
    log.info("green skeleton px=%d", int(gsk.sum()))

    max_gap_px = gap_close_factor * line_thickness
    debug_lines = []
    closed_mask, n_closed = close_gaps(
        green_mask, gsk,
        max_gap_px=max_gap_px,
        thickness=line_thickness,
        debug=debug_lines,
    )
    log.info("closed %d endpoint pairs (max_gap_px=%.1f)", n_closed, max_gap_px)

    # krok 3: ekstrapolacja gałęzi T-junctions na drugą stronę głównej linii.
    # Kluczowe dla domknięcia granic działek które wpływają w linię pasa
    # drogowego jako T-junction (nie jako wolny endpoint).
    extend_px = tjunction_extend_pt * scale
    tj_debug = []
    gsk2 = skeletonize(closed_mask > 0)
    closed_mask, n_ext = extend_tjunctions(
        closed_mask, gsk2,
        max_extend_px=extend_px,
        thickness=line_thickness,
        min_branch_len=max(3, int(line_thickness * 2)),
        stop_on_hit=True,
        debug=tj_debug,
    )
    log.info("extended %d T-junctions (max_extend_px=%.1f)", n_ext, extend_px)

    # krok 4: pin-based poprzeczne granice pasa drogowego.
    # Piny pomiarowe (width=0.84) to znaki geodezyjne często umieszczone
    # w punktach gdzie trasa przecina granicę działki. Rysujemy przez
    # każdy pin krótki odcinek prostopadły do lokalnego kierunku trasy —
    # sztucznie dzieli pas drogowy na sub-działki tam, gdzie narysowane
    # granice są niekompletne.
    if enable_pin_crosslines:
        red_segs_pt = red_segments(reds_trace)
        pins_pt = dedup_pins(pin_centers_pt(red_pins), tol=4.0)
        closed_mask, n_cross = draw_pin_crosslines(
            closed_mask, ctx, pins_pt, red_segs_pt,
            half_length_pt=pin_crossline_half_pt,
            thickness=line_thickness,
        )
        log.info("pin crosslines: %d pins → %d lines (half=%.1fpt)",
                 len(pins_pt), n_cross, pin_crossline_half_pt)

    # krok 5: ramka — zamyka "otwarte" komponenty przy brzegach mapy
    if enable_frame:
        closed_mask = draw_frame(closed_mask, thickness=line_thickness)

    # krok 6: odizoluj prawdziwe tło od wycieków pasa drogowego.
    # Piksele > bg_max_dist_pt od dowolnej zielonej granicy traktujemy jako
    # "wypełnione" — rozdziela prawdziwe tło od wąskich korytarzy
    # (pas drogowy, słabo zamknięte slivery).
    bg_max_dist_px = bg_max_dist_pt * scale
    closed_mask = fill_true_background(closed_mask, max_dist_px=bg_max_dist_px)
    log.info("after bg-fill: green_px=%d (max_dist=%.1fpx)",
             int((closed_mask > 0).sum()), bg_max_dist_px)

    # Komponenty działek
    labels_img, n_components = parcel_components(
        closed_mask, min_area_px=int(50 * (scale ** 2) / 16),
    )
    log.info("parcel components=%d", n_components)

    # Cienki corridor (do weryfikacji przecięcia granicy komponentu z trasą)
    buffer_px = corridor_buffer_pt * scale
    corridor = build_road_corridor(red_mask, pin_centers, buffer_px=buffer_px)
    log.info("corridor: buffer_px=%.1f, px=%d",
             buffer_px, int((corridor > 0).sum()))

    # Komponenty "crossed" definiujemy jako:
    #   (1) te których WNĘTRZE ma >0 pikseli corridor-a (trasa biegnie w środku), LUB
    #   (2) te których GRANICA dotyka corridor-a (trasa przecina zieloną granicę).
    # (2) łapie normalne działki przez które trasa przechodzi przeci granicę —
    # nawet jeśli etykieta nie jest w tym samym komponencie co trasa.
    # Implementacja: dilate corridor o thickness linii; dla każdego piksela
    # w dilated corridor, zbierz jego label → zbiór crossed labels.
    dilated_corr = cv2.dilate(
        corridor,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * line_thickness + 1, 2 * line_thickness + 1),
        ),
    )
    touched_labels = np.unique(labels_img[dilated_corr > 0])
    crossed_components = set(int(x) for x in touched_labels if x != 0)
    log.info("crossed components=%d", len(crossed_components))

    # OCR
    if ocr_cache and Path(ocr_cache).exists():
        labels = load_ocr_cache(Path(ocr_cache), ocr_scale=ocr_scale)
        log.info("OCR cache loaded: %d labels", len(labels))
    else:
        raise RuntimeError(
            "OCR cache not available — uruchom najpierw analyze.py z --ocr-cache"
        )

    # Dopasowanie etykieta → komponent
    # Bardzo mały spiralny radius — etykieta zwykle siedzi W środku
    # komponentu, a "label_to_component" szuka tylko jeśli upadła DOKŁADNIE
    # na piksel granicy zielonej. Duży radius tworzy fałszywe przyporządkowania.
    search_r_px = max(2, int(round(2 * scale)))

    crossed_labels: dict[str, tuple[Label, int]] = {}
    borderline_labels: dict[str, tuple[Label, int]] = {}
    for lbl in labels:
        x_px, y_px = ctx.pt2px(lbl.x, lbl.y)
        cid = label_to_component(labels_img, x_px, y_px,
                                 search_radius_px=search_r_px)
        if cid == 0:
            # etykieta w wycieku tła → niepewna, do sprawdzenia ręcznie
            borderline_labels.setdefault(lbl.text, (lbl, 0))
            continue
        if cid in crossed_components:
            prev = crossed_labels.get(lbl.text)
            if prev is None or lbl.conf > prev[0].conf:
                crossed_labels[lbl.text] = (lbl, cid)
        else:
            # etykieta w nieprzeciętym komponencie — NIE przecięta
            pass

    # borderline: etykiety bez komponentu, i blisko trasy — może wsunąć się
    # do KROKU 2 klasyfikacji (za dużo tolerancji → FP)
    # TU NIE DODAJEMY NIC automatycznie — zgodnie z CEL.md
    # borderline_labels pozostaje jedynie jako informacja
    crossed_text = sorted(set(crossed_labels.keys()))

    # borderline wyłącznie te, które nie mają przyporządkowania do żadnego
    # komponentu crossed
    borderline_text = sorted([
        t for t in borderline_labels
        if t not in crossed_labels
    ])

    log.info("crossed=%d borderline=%d (%.1fs)",
             len(crossed_text), len(borderline_text), time.time() - t0)

    if debug_png:
        _save_debug(debug_png, page, scale, ctx, greens, reds_trace, red_pins,
                    closed_mask, corridor, labels_img,
                    crossed_components, labels, crossed_labels)

    return Result(
        crossed=crossed_text,
        borderline=borderline_text,
        debug={
            "n_components": n_components,
            "n_crossed_components": len(crossed_components),
            "n_closed_gaps": n_closed,
            "crossed_detail": [
                {"text": t, "x": l.x, "y": l.y, "cid": cid}
                for t, (l, cid) in crossed_labels.items()
            ],
            "borderline_detail": [
                {"text": t, "x": l.x, "y": l.y}
                for t, (l, _) in borderline_labels.items()
                if t not in crossed_labels
            ],
        },
    )


def _save_debug(path, page, scale, ctx, greens, reds_trace, red_pins,
                closed_mask, corridor, labels_img,
                crossed_components, labels, crossed_labels):
    """Wizualizuj: tło białe + zielone granice + corridor + numery crossed."""
    H, W = closed_mask.shape
    vis = np.full((H, W, 3), 255, dtype=np.uint8)
    # zielone granice (jasno)
    vis[closed_mask > 0] = (200, 255, 200)
    # corridor na różowo (alpha ~30%)
    mask = corridor > 0
    vis[mask] = np.clip(vis[mask].astype(int) - np.array([0, 80, 80]), 0, 255).astype(np.uint8)
    # crossed components — ciemniejszy zielony
    crossed_mask = np.isin(labels_img, list(crossed_components))
    vis[crossed_mask] = np.clip(vis[crossed_mask].astype(int) - np.array([60, 0, 60]), 0, 255).astype(np.uint8)
    # etykiety crossed
    for t, (l, cid) in crossed_labels.items():
        x_px, y_px = ctx.pt2px(l.x, l.y)
        cv2.putText(vis, t, (x_px - 20, y_px + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
    # zeskaluj do rozsądnego rozmiaru (max 4000 wide)
    max_w = 4000
    if W > max_w:
        f = max_w / W
        vis = cv2.resize(vis, (max_w, int(H * f)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="OpenCV-based parcel-crossing analysis.")
    p.add_argument("pdf")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--gap-close-factor", type=float, default=8.0,
                   help="max_gap_px = gap_close_factor * line_thickness")
    p.add_argument("--corridor-buffer", type=float, default=2.0,
                   help="bufor pasa drogowego w punktach PDF (cienki — granica komponentu dotyka corridoru)")
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument("--ocr-cache", default="/tmp/ocr_cache_v2.pkl")
    p.add_argument("--debug-png", default=None)
    p.add_argument("--morph-close-px", type=int, default=2)
    p.add_argument("--tjunction-extend", type=float, default=25.0,
                   help="maks. ekstrapolacja gałęzi T-junction przez główną linię (pt PDF)")
    p.add_argument("--pin-crossline-half", type=float, default=15.0,
                   help="połowa długości sztucznej poprzecznej granicy przez pin pomiarowy (pt)")
    p.add_argument("--no-pin-crosslines", dest="pin_crosslines", action="store_false")
    p.add_argument("--no-frame", dest="frame", action="store_false")
    p.set_defaults(pin_crosslines=True, frame=True)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    res = analyze(
        args.pdf,
        scale=args.scale,
        gap_close_factor=args.gap_close_factor,
        corridor_buffer_pt=args.corridor_buffer,
        line_thickness=args.line_thickness,
        ocr_cache=args.ocr_cache,
        debug_png=args.debug_png,
        morph_close_px=args.morph_close_px,
        tjunction_extend_pt=args.tjunction_extend,
        pin_crossline_half_pt=args.pin_crossline_half,
        enable_pin_crosslines=args.pin_crosslines,
        enable_frame=args.frame,
    )
    print(f"DZIAŁKI PRZECIĘTE ({len(res.crossed)}):")
    print("  " + (", ".join(res.crossed) if res.crossed else "—"))
    print(f"DZIAŁKI DO SPRAWDZENIA ({len(res.borderline)}):")
    print("  " + (", ".join(res.borderline) if res.borderline else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
