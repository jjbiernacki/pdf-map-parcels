"""Test e2e: analyze_hybrid na wszystkich mapach w Mapy/.

GT pochodzi z pliku "Mapy/Działki na mapach.txt". Każda mapa to osobny test.

Warunek akceptacji: `set(crossed) == GT` (0 FP, 0 FN).
Mapy z nieprzezwyciężalnymi problemami geometrycznymi są obecnie
oznaczone jako `xfail` z komentarzem objaśniającym przyczynę.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import analyze_hybrid  # noqa: E402
import fitz  # noqa: E402

MAPY_DIR = PROJECT_ROOT / "Mapy"
GT_FILE = MAPY_DIR / "Działki na mapach.txt"
OCR_CACHE_DIR = Path("/tmp/mapy_ocr")


def parse_gt(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            current = None
            continue
        if line.endswith(":"):
            current = line[:-1].strip()
            result.setdefault(current, set())
        elif current is not None:
            for token in line.split(","):
                t = token.strip()
                if t:
                    result[current].add(t)
    return result


GT_BY_NAME = parse_gt(GT_FILE)


def _require_ocr_cache(pdf_stem: str) -> Path:
    cache = OCR_CACHE_DIR / f"{pdf_stem}.pkl"
    if not cache.exists():
        pytest.skip(
            f"Brak OCR cache {cache}. Uruchom `python run_all_maps.py` "
            f"żeby wygenerować cache"
        )
    return cache


# Mapy które obecnie NIE osiągają pełnego 0 FP / 0 FN — oznaczone jako
# xfail z wyjaśnieniem (vs. test oznaczony, że wynik się poprawił).
EXPECTED_FAILURES: dict[str, str] = {
    "PZT Grochowska-Model": (
        "0 FN (wszystkie GT wypisane), ale 2 FP: 420 i 501 — to realne "
        "sąsiednie działki adjacent do trasy, z silnym sygnałem geometrycznym "
        "(boundary-along-route 33-48%, area_ratio_hull niskie). "
        "User akceptuje FP > FN"
    ),
}


@pytest.fixture(scope="module")
def maps_with_gt():
    return sorted(p for p in MAPY_DIR.glob("*.pdf") if p.stem in GT_BY_NAME)


@pytest.mark.parametrize(
    "map_name",
    sorted(GT_BY_NAME.keys()),
    ids=list(sorted(GT_BY_NAME.keys())),
)
def test_map(map_name):
    pdf = MAPY_DIR / f"{map_name}.pdf"
    if not pdf.exists():
        pytest.skip(f"Brak PDF-a {pdf}")
    gt = GT_BY_NAME[map_name]
    cache = _require_ocr_cache(map_name)

    res = analyze_hybrid.analyze(str(pdf), ocr_cache=str(cache))
    got = set(res.crossed)

    fp = got - gt
    fn = gt - got
    msg = (
        f"{map_name}: TP {len(got & gt)}/{len(gt)}  "
        f"FP {sorted(fp) or '—'}  FN {sorted(fn) or '—'}"
    )
    if map_name in EXPECTED_FAILURES and (fp or fn):
        pytest.xfail(f"{msg} | {EXPECTED_FAILURES[map_name]}")
    assert not fp, f"False positives: {msg}"
    assert not fn, f"False negatives: {msg}"


def test_03_pzt_perfect():
    """Sanity: 03 PZT musi dawać pełny 15/15, 0 FP, 0 FN."""
    map_name = "03 PZT granice"
    gt = GT_BY_NAME[map_name]
    cache = _require_ocr_cache(map_name)
    res = analyze_hybrid.analyze(str(MAPY_DIR / f"{map_name}.pdf"),
                                 ocr_cache=str(cache))
    got = set(res.crossed)
    assert got == gt, f"regression: expected {sorted(gt)}, got {sorted(got)}"


@pytest.mark.parametrize(
    "map_name",
    sorted(GT_BY_NAME.keys()),
    ids=list(sorted(GT_BY_NAME.keys())),
)
def test_no_false_negatives(map_name):
    """Każda mapa MUSI mieć 0 FN — wszystkie GT muszą być wypisane.
    User priority: lepiej kilka FP niż brak działki w wypisie."""
    pdf = MAPY_DIR / f"{map_name}.pdf"
    if not pdf.exists():
        pytest.skip(f"Brak PDF-a {pdf}")
    gt = GT_BY_NAME[map_name]
    cache = _require_ocr_cache(map_name)
    res = analyze_hybrid.analyze(str(pdf), ocr_cache=str(cache))
    got = set(res.crossed)
    fn = gt - got
    assert not fn, f"{map_name} ma FN: {sorted(fn)} (TP {len(got&gt)}/{len(gt)})"
