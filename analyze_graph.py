"""Graph-based polygon detection: dla każdej etykiety znajduję najmniejszy
cykl w grafie zielonych linii który ją otacza, ZAMIAST polegać na globalnym
polygonize.

Algorytm (clockwise next-edge face traversal):
  1. Zbuduj graf granic — każdy węzeł to unikalny punkt (po snap-tolerancji),
     każda krawędź to segment między dwoma węzłami.
  2. Dla każdej etykiety L:
     a. Rzuć promień z L w prawo (kierunek +X).
     b. Pierwsza krawędź trafiona przez promień = start granicy działki L,
        z punktem przecięcia jako punktem wejścia.
     c. Idź tą krawędzią w kierunku gdzie etykieta jest PO PRAWEJ STRONIE
        (clockwise traversal).
     d. Na każdym kolejnym wierzchołku wybieraj NASTĘPNĄ krawędź po lewej
        (najmniejszy kąt lewoskrętny od wejścia) — to zapewnia clockwise
        traversal po face'u.
     e. Zatrzymaj gdy wrócisz do punktu startowego.
  3. Zebrane krawędzie = granica działki. Sprawdź czy którakolwiek
     krawędź przecina trasę.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from analyze_cv import (
    extract_paths, iter_segments, load_ocr_cache,
)


@dataclass
class Result:
    crossed: list[str] = field(default_factory=list)
    borderline: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def _snap_pt(p: tuple[float, float], tol: float) -> tuple[int, int]:
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))


class BorderGraph:
    def __init__(self, snap_tol: float = 1.0):
        self.snap_tol = snap_tol
        # nodes: canonical_key → (x, y)
        self.nodes: dict[tuple[int, int], tuple[float, float]] = {}
        # adj: node_key → set of (neighbor_key, segment_LineString)
        self.adj: dict[tuple[int, int], list[tuple[tuple[int, int], LineString]]] = defaultdict(list)

    def add_segment(self, p1: tuple[float, float], p2: tuple[float, float]):
        k1 = _snap_pt(p1, self.snap_tol)
        k2 = _snap_pt(p2, self.snap_tol)
        if k1 == k2:
            return
        self.nodes.setdefault(k1, p1)
        self.nodes.setdefault(k2, p2)
        ls = LineString([self.nodes[k1], self.nodes[k2]])
        self.adj[k1].append((k2, ls))
        self.adj[k2].append((k1, ls))

    def build_from_segments(self, segs):
        for p1, p2 in segs:
            self.add_segment(p1, p2)

    def all_edges(self) -> list[LineString]:
        seen = set()
        out = []
        for k, neighs in self.adj.items():
            for nb, ls in neighs:
                key = tuple(sorted([k, nb]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(ls)
        return out


def _angle_from(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Kąt w radianach wektora p1→p2."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def find_face_containing_label(graph: BorderGraph, lbl_pt: Point,
                               *, max_steps: int = 5000,
                               ray_len: float = 1e5) -> list[LineString] | None:
    """Znajdź najmniejszy face w grafie zawierający punkt etykiety.

    Strategia: rzuć promień w prawo, znajdź pierwszą krawędź. Potem CW
    traversal face'u.
    """
    # Pierwsza krawędź od etykiety w kierunku +X
    ray = LineString([(lbl_pt.x, lbl_pt.y), (lbl_pt.x + ray_len, lbl_pt.y)])
    best_d = None
    best_edge = None  # (ls, k1, k2)
    best_hit = None
    # iteracja po wszystkich krawędziach — liniowe, ok dla niedużych grafów
    for k1, neighs in graph.adj.items():
        for k2, ls in neighs:
            if (k1, k2) in {(a, b) for b, _ in []}:
                pass
            # przetwarzamy każdą unieważnioną krawędź raz
            if k1 >= k2:
                continue
            if not ls.intersects(ray):
                continue
            ip = ls.intersection(ray)
            if ip.geom_type == "Point":
                d = lbl_pt.distance(ip)
            elif ip.geom_type == "MultiPoint":
                d = min(lbl_pt.distance(g) for g in ip.geoms)
            else:
                continue
            if d < 1e-6:
                continue
            if best_d is None or d < best_d:
                best_d = d
                best_edge = (ls, k1, k2)
                best_hit = ip
    if best_edge is None:
        return None

    # Start: wybieramy węzeł (u) który jest "po właściwej stronie" żeby
    # pójść CW wokół face'u zawierającego etykietę.
    ls, k1, k2 = best_edge
    # Kierunek początkowy: wzdłuż krawędzi, z wyborem k1→k2 lub k2→k1.
    # Dla face'u zawierającego etykietę (która jest PO LEWEJ od ray
    # punkt-trafienia→prawo), musimy pójść tak żeby etykieta była PO PRAWEJ.
    # Prościej: wybierz kierunek taki, że cross product (hit→węzeł) ×
    # (hit→etykieta) jest DODATNI (CCW w układzie matematycznym).
    p1 = graph.nodes[k1]
    p2 = graph.nodes[k2]
    hx, hy = best_hit.x, best_hit.y
    # Wektor od hit do etykiety
    lx = lbl_pt.x - hx
    ly = lbl_pt.y - hy
    # Wybór: jeśli cross((p2-hit), (lbl-hit)) > 0, idziemy przez k2.
    # Cross > 0 = lbl jest po LEWEJ od kierunku hit→p2 (w PDF coord y w dół
    # = standardowa konwencja ekranu, ale shapely traktuje to jako y rośnie
    # w dół; cross sign flipuje). Dla CW traversal gdzie etykieta po prawej,
    # bierzemy kierunek gdzie LBL jest PO LEWEJ (bo idziemy CW, face po lewej).
    cross_to_p2 = (p2[0] - hx) * ly - (p2[1] - hy) * lx
    if cross_to_p2 > 0:
        start_from, start_to = k1, k2
    else:
        start_from, start_to = k2, k1
    # Jeżeli k1 jest trafieniem — pomijamy (punkt hit jest IN MIDDLE krawędzi)
    # Nasze traversowanie startuje z węzła `start_to` idąc Z wewnątrz
    # graph.adj[start_to].

    visited_edges = set()
    current_from = start_from
    current_to = start_to
    cycle: list[LineString] = []
    for _ in range(max_steps):
        # add edge current_from→current_to
        edge_key = tuple(sorted([current_from, current_to]))
        if edge_key in visited_edges:
            # powrót do krawędzi = koniec cyklu
            break
        visited_edges.add(edge_key)
        # znajdź krawędź current_from→current_to
        edge_ls = None
        for nb, ls in graph.adj[current_from]:
            if nb == current_to:
                edge_ls = ls
                break
        if edge_ls is not None:
            cycle.append(edge_ls)
        # na węźle current_to, wybierz NASTĘPNĄ krawędź idąc CW:
        # - kierunek wejścia: from current_from, at angle θ_in = angle(current_to→current_from)
        # - wszystkie sąsiady current_to (oprócz current_from) mają kąty θ_out
        # - wybierz sąsiada gdzie (θ_out - θ_in) jest NAJWIĘKSZE CCW (najmniejsze CW)
        incoming_angle = _angle_from(graph.nodes[current_to], graph.nodes[current_from])
        best_next = None
        best_delta = None
        for nb, _ls in graph.adj[current_to]:
            if nb == current_from:
                # pozwól na U-turn tylko jeśli to jedyna opcja (dead-end)
                continue
            out_angle = _angle_from(graph.nodes[current_to], graph.nodes[nb])
            # delta = (out - in) mod 2π; najmniejszy delta = najbardziej CW
            delta = (out_angle - incoming_angle) % (2 * math.pi)
            if delta < 1e-9:
                delta += 2 * math.pi
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_next = nb
        if best_next is None:
            # dead-end — u-turn
            best_next = current_from
        current_from, current_to = current_to, best_next
        if (current_from == start_from and current_to == start_to):
            break
    if not cycle:
        return None
    return cycle


def classify_via_graph(greens, reds_trace, labels_, *,
                        snap_tol: float = 2.0,
                        red_buffer_pt: float = 2.0,
                        min_cycle_edges: int = 3) -> tuple[set[str], dict]:
    # zbierz segmenty zielone jako pary pt (oryginalnego PDF-a)
    segs = []
    for d in greens:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                segs.append((p1, p2))
    g = BorderGraph(snap_tol=snap_tol)
    g.build_from_segments(segs)

    red_lines = []
    for d in reds_trace:
        for p1, p2 in iter_segments(d):
            if p1 != p2:
                red_lines.append(LineString([p1, p2]))
    if red_lines:
        red_union = unary_union(MultiLineString(red_lines))
    else:
        red_union = LineString()
    if red_buffer_pt > 0 and not red_union.is_empty:
        red_buf = red_union.buffer(red_buffer_pt, cap_style=2, join_style=2)
    else:
        red_buf = red_union

    crossed_text: dict[str, float] = {}
    debug = {}
    for lbl in labels_:
        lp = Point(lbl.x, lbl.y)
        cycle = find_face_containing_label(g, lp)
        if not cycle or len(cycle) < min_cycle_edges:
            continue
        # czy którakolwiek krawędź cyklu jest crossed przez trasę
        any_crossed = False
        for e in cycle:
            if not red_buf.is_empty and e.intersects(red_buf):
                any_crossed = True
                break
        if any_crossed:
            prev = crossed_text.get(lbl.text, -1)
            if lbl.conf > prev:
                crossed_text[lbl.text] = lbl.conf
    return set(crossed_text.keys()), debug


def analyze(pdf_path: str | Path, *, ocr_cache: str | Path,
            ocr_scale: int = 8,
            snap_tol: float = 2.0,
            red_buffer_pt: float = 2.0) -> Result:
    log = logging.getLogger("analyze_graph")
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    greens, reds_trace, red_pins = extract_paths(page)
    labels = load_ocr_cache(Path(ocr_cache), ocr_scale=ocr_scale)
    crossed, _ = classify_via_graph(
        greens, reds_trace, labels,
        snap_tol=snap_tol, red_buffer_pt=red_buffer_pt,
    )
    return Result(crossed=sorted(crossed))
