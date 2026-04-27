"""Testy pipeline'u analizy mapy.

Dwie grupy:
  - testy jednostkowe (bez PDF-a): klasyfikacja kolorów, regex etykiet,
    polygonizacja na syntetycznych danych, klasyfikacja crossed/borderline
  - test end-to-end: uruchomienie na realnym PDF-ie z katalogu projektu
    (`03 PZT granice.pdf`) + porównanie z `numery działek.txt`.
    Używa cache'a OCR jeśli istnieje — inaczej pomija test (OCR ~5 min).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

# Tak, żeby pytest uruchamiany z katalogu projektu widział `analyze`:
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import (  # noqa: E402
    LABEL_RE,
    Label,
    analyze,
    build_red_union,
    classify,
    is_green_stroke,
    is_red_stroke,
    parcel_label_matches,
    polygonize_boundaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "03 PZT granice.pdf"
GT_PATH = PROJECT_ROOT / "numery działek.txt"
OCR_CACHE = Path("/tmp/ocr_cache_v2.pkl")


def _load_gt() -> set[str]:
    """Wczytaj zbiór oczekiwanych numerów działek z `numery działek.txt`."""
    lines = [ln.strip() for ln in GT_PATH.read_text(encoding="utf-8").splitlines()]
    return {ln for ln in lines if ln}


# ---------------------------------------------------------------------------
# UNIT: klasyfikacja koloru
# ---------------------------------------------------------------------------

class TestColorClassification:
    def test_red_pure(self):
        assert is_red_stroke((1.0, 0.0, 0.0))

    def test_red_tolerance(self):
        # niewielki szum jest akceptowany
        assert is_red_stroke((0.98, 0.02, 0.01))

    def test_not_red_pinkish(self):
        assert not is_red_stroke((0.9, 0.4, 0.4))

    def test_green_pure(self):
        assert is_green_stroke((0.0, 0.584, 0.0))

    def test_green_tolerance(self):
        assert is_green_stroke((0.05, 0.6, 0.05))

    def test_not_green_dark(self):
        assert not is_green_stroke((0.0, 0.2, 0.0))  # za ciemny

    def test_none_is_neither(self):
        assert not is_red_stroke(None)
        assert not is_green_stroke(None)


# ---------------------------------------------------------------------------
# UNIT: regex etykiet
# ---------------------------------------------------------------------------

class TestLabelRegex:
    @pytest.mark.parametrize("s", ["391", "336", "296/3", "48/1", "1", "99/99"])
    def test_accept(self, s):
        assert parcel_label_matches(s)

    @pytest.mark.parametrize(
        "s",
        [
            "",
            "abc",
            "296/",
            "/3",
            "3/3/3",
            "12.3",
            "strona 3",
            "2024",       # 4-cyfrowy = artefakt OCR w tym katastrze
            "7283",       # artefakt z realnego PDF-a
            "1315/16",    # 4-cyfrowy prefix
        ],
    )
    def test_reject(self, s):
        assert not parcel_label_matches(s)

    def test_regex_binding(self):
        # exact full-match — "391abc" nie może przejść
        assert LABEL_RE.match("391abc") is None


# ---------------------------------------------------------------------------
# UNIT: polygonizacja na syntetyku
# ---------------------------------------------------------------------------

def _fake_green(segments: list[tuple[tuple[float, float], tuple[float, float]]]):
    """Zbuduj „drawing"-podobne dict-y zgodne z interfejsem `iter_segments`."""
    class _P:  # imitacja fitz.Point
        def __init__(self, x, y):
            self.x, self.y = x, y

    out = []
    for p1, p2 in segments:
        items = [("l", _P(*p1), _P(*p2))]
        out.append({"items": items})
    return out


class TestPolygonize:
    def test_two_squares_produce_two_polygons(self):
        # Dwa rozłączne kwadraty: (0,0)-(10,10) i (20,0)-(30,10)
        sq1 = [((0, 0), (10, 0)), ((10, 0), (10, 10)),
               ((10, 10), (0, 10)), ((0, 10), (0, 0))]
        sq2 = [((20, 0), (30, 0)), ((30, 0), (30, 10)),
               ((30, 10), (20, 10)), ((20, 10), (20, 0))]
        polys = polygonize_boundaries(_fake_green(sq1 + sq2), snap_tol=0.1)
        assert len(polys) == 2
        assert all(p.area == pytest.approx(100.0) for p in polys)

    def test_snap_closes_small_gap(self):
        # Kwadrat z 1 pt luki w narożniku
        segs = [((0, 0), (10, 0)), ((10, 0), (10, 10)),
                ((10, 10), (0, 10)), ((0, 10), (0.0, 1.0))]  # nie zamyka w 0,0
        polys_no_snap = polygonize_boundaries(_fake_green(segs), snap_tol=0.0)
        polys_snap = polygonize_boundaries(_fake_green(segs), snap_tol=2.0)
        assert len(polys_no_snap) == 0
        assert len(polys_snap) == 1


# ---------------------------------------------------------------------------
# UNIT: classify() — crossed vs borderline
# ---------------------------------------------------------------------------

def _square(x0, y0, w, h) -> Polygon:
    return Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h),
                    (x0, y0 + h), (x0, y0)])


class TestClassify:
    def test_line_through_interior_is_crossed(self):
        poly = _square(0, 0, 10, 10)
        route = LineString([(-5, 5), (15, 5)])
        lbl = Label(text="100", conf=1.0, x=5, y=5)
        crossed, bd = classify([poly], [lbl], route)
        assert [t for t, _, _ in crossed] == ["100"]
        assert bd == []

    def test_line_grazing_boundary_is_borderline(self):
        poly = _square(0, 0, 10, 10)
        # Trasa BIEGNIE TUŻ POD spodem (y=-0.3) — muska polygonu w tolerancji
        route = LineString([(-5, -0.3), (15, -0.3)])
        lbl = Label(text="101", conf=1.0, x=5, y=5)
        crossed, bd = classify([poly], [lbl], route,
                                borderline_dist_pt=1.0,
                                corridor_dist_pt=20.0)
        assert [t for t, _, _ in crossed] == []
        assert [t for t, _, _ in bd] == ["101"]

    def test_label_outside_polygon_near_route(self):
        # Etykieta w "pasie drogowym" — poza zamkniętymi wielokątami
        poly = _square(0, 0, 10, 10)
        route = LineString([(30, 0), (60, 0)])  # daleko od poly
        lbl = Label(text="302", conf=1.0, x=45, y=2)
        crossed, bd = classify([poly], [lbl], route,
                                corridor_dist_pt=10.0)
        assert crossed == []
        assert [t for t, _, _ in bd] == ["302"]

    def test_label_far_from_everything_ignored(self):
        poly = _square(0, 0, 10, 10)
        route = LineString([(0, 5), (20, 5)])
        lbl = Label(text="999", conf=1.0, x=500, y=500)
        crossed, bd = classify([poly], [lbl], route,
                                corridor_dist_pt=10.0,
                                endpoint_dist_pt=10.0)
        assert crossed == []
        assert bd == []

    def test_label_near_route_endpoint_is_borderline(self):
        # Etykieta za końcem trasy — typowe dla działki na końcówce pasa
        route = LineString([(0, 0), (100, 0)])
        lbl = Label(text="284", conf=1.0, x=300, y=10)  # 200pt za końcem
        crossed, bd = classify([], [lbl], route,
                                corridor_dist_pt=10.0,
                                endpoint_dist_pt=250.0)
        assert [t for t, _, _ in bd] == ["284"]

    def test_duplicate_text_dedup_keeps_closest_to_route(self):
        # Ten sam tekst w dwóch miejscach — jeden w pasie, drugi daleko
        route = LineString([(0, 0), (100, 0)])
        l_near = Label(text="280", conf=1.0, x=50, y=5)   # przy trasie
        l_far = Label(text="280", conf=1.0, x=50, y=500)  # daleko
        crossed, bd = classify([], [l_near, l_far], route,
                                corridor_dist_pt=20.0)
        # dokładnie jeden wpis, bliższy trasy
        assert len(bd) == 1
        _, lbl, _ = bd[0]
        assert lbl.y == pytest.approx(5)


# ---------------------------------------------------------------------------
# END-TO-END: realny PDF + numery działek.txt
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not PDF_PATH.exists(),
    reason="PDF z mapą nie jest dostępny — pomiń test e2e.",
)
@pytest.mark.skipif(
    not OCR_CACHE.exists(),
    reason="Cache OCR /tmp/ocr_cache_v2.pkl nie istnieje — test e2e "
           "bez cache'a trwa >5min; uruchom `analyze.py --ocr-cache` raz "
           "żeby go wygenerować.",
)
class TestGroundTruthE2E:
    """Najważniejszy test: cały pipeline na realnym PDF-ie.

    Kryterium akceptacji spec'a:
      - żadnych FP w zbiorze `crossed` (nie może tam być działki spoza GT)
      - WSZYSTKIE działki z `numery działek.txt` znajdują się w unii
        `crossed ∪ borderline` (żadna nie jest pominięta)
    """

    @pytest.fixture(scope="class")
    def result(self):
        return analyze(
            PDF_PATH,
            ocr_scale=8,
            snap_tol=5.0,
            corridor_dist_pt=100.0,
            endpoint_dist_pt=250.0,
            ocr_cache=str(OCR_CACHE),
        )

    def test_gt_file_has_19_entries(self):
        assert len(_load_gt()) == 19

    def test_crossed_has_no_false_positives(self, result):
        gt = _load_gt()
        extras = set(result.crossed) - gt
        assert extras == set(), f"FP w crossed: {extras}"

    def test_all_gt_parcels_covered(self, result):
        gt = _load_gt()
        covered = set(result.crossed) | set(result.borderline)
        missing = gt - covered
        assert missing == set(), f"pominięte działki GT: {missing}"

    def test_crossed_contains_core_crossings(self, result):
        # Działki geometrycznie wyraźnie przeciętej trasy
        core = {
            "277", "278", "279", "280", "281", "282",
            "254", "296/2", "296/3",
            "334", "336", "337",
        }
        missing = core - set(result.crossed)
        assert missing == set(), (
            f"oczekiwane pewne przecięcia brakujące w crossed: {missing}"
        )

    def test_borderline_contains_corridor_parcels(self, result):
        # Działki w „pasie drogowym" / przy końcówce — wymagają weryfikacji
        corridor = {"391", "310", "48/1", "49/1", "51/1", "283", "284"}
        missing = corridor - set(result.borderline)
        assert missing == set(), (
            f"oczekiwane sporne działki pasa drogowego brakujące w borderline: "
            f"{missing}"
        )

    def test_crossed_and_borderline_are_disjoint(self, result):
        overlap = set(result.crossed) & set(result.borderline)
        assert overlap == set(), (
            f"ta sama działka w obu listach: {overlap}"
        )
