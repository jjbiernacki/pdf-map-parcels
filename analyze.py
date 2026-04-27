"""Analiza map geodezyjnych — wykrywanie działek przeciętych czerwoną linią.

Pipeline:
  1.  Parsowanie PDF-a (PyMuPDF `page.get_drawings`):
        - zielone kreski  → granice działek
        - czerwone kreski o `width > 1` → osiowa linia trasy (kreskowana)
        - pozostałe czerwone (piny, wypełnienia) → pomijane
  2.  Wektorowa polygonizacja — `shapely.ops.polygonize` po `unary_union`
      zrzutowanym przez `snap(tol)`: zamyka mikro-szczeliny w wierzchołkach.
  3.  OCR etykiet (easyocr) — wyniki cache'owane w `--ocr-cache` (pickle).
  4.  Dla każdej etykiety:
        - jeżeli wpada DO wielokąta i wielokąt przecina czerwoną trasę
          wewnętrznie → działka **crossed** (pewne przecięcie).
        - jeżeli etykieta LEŻY W PASIE DROGOWYM (blisko trasy, nie w
          zamkniętym wielokącie lub w wielokącie nie przecinanym wewnętrznie
          ale przylegającym do pasa) → **borderline** (do weryfikacji
          ręcznej — zgodnie z wymogiem użytkownika: nie zgadujemy).
  5.  Kolejność wzdłuż trasy: sortowanie po arc-length projekcji etykiety
      na czerwoną trasę.

Klasyfikacja "crossed" vs "borderline" bazuje na deterministycznych
progach geometrycznych i daje powtarzalne wyniki — żadnych heurystyk
"na oko".

CLI:
    python analyze.py "03 PZT granice.pdf" [--debug-png out.png] [-v]
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

import fitz  # pymupdf
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize, snap, unary_union

__all__ = [
    "Result",
    "Label",
    "analyze",
    "extract_paths",
    "iter_segments",
    "is_green_stroke",
    "is_red_stroke",
    "polygonize_boundaries",
    "LABEL_RE",
]


# ---------------------------------------------------------------------------
# Color classification — PDF uses red=(1,0,0), green=(0,0.584,0)
# ---------------------------------------------------------------------------

def is_green_stroke(rgb) -> bool:
    """Zielony stroke granicy działki (z pewną tolerancją RGB)."""
    if rgb is None:
        return False
    r, g, b = rgb
    return r < 0.15 and 0.3 < g < 0.85 and b < 0.15


def is_red_stroke(rgb) -> bool:
    """Czysty czerwony stroke — trasa lub pin."""
    if rgb is None:
        return False
    r, g, b = rgb
    return r > 0.85 and g < 0.15 and b < 0.15


# ---------------------------------------------------------------------------
# Rozbicie ścieżek PDF na pojedyncze odcinki
# ---------------------------------------------------------------------------

def iter_segments(drawing):
    """Iteruje pary końców (p1,p2) odcinków w danej ścieżce.

    Krzywe Beziera aproksymujemy prostą od p0 do p3 — wystarczy dla
    granic i trasy, a koszt obliczeniowy jest stały.
    """
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


def extract_paths(page):
    """Zwraca (zielone_granice, czerwona_trasa) z danej strony PDF.

    Filtrowanie:
      - zielone: usuwamy etykiety-glify (małe multi-path, w<30pt h<20pt)
      - czerwone: tylko stroke width>1 (kreska trasy = 1.44pt);
                  piny (0.84pt) i wypełnienia pomijamy.
    """
    green, red = [], []
    for d in page.get_drawings():
        color = d.get("color")
        width = d.get("width") or 0
        typ = d.get("type")
        rect = d.get("rect")
        if typ == "f":
            # czyste wypełnienie (np. główka pina) — nieinteresujące
            continue
        if is_green_stroke(color):
            if rect is not None and len(d["items"]) > 1 \
                    and rect.width < 30 and rect.height < 20:
                continue  # glif etykiety — nie granica
            green.append(d)
        elif is_red_stroke(color) and width > 1.0:
            red.append(d)
    return green, red


# ---------------------------------------------------------------------------
# Regex parcelowego identyfikatora
# ---------------------------------------------------------------------------

# Numery działek w tym dokumencie to 1–3 cyfry z opcjonalnym "/N" (np. "391",
# "296/3", "48/1"). 4-cyfrowe ciągi (typu "7283") są prawie zawsze OCR-owymi
# sklejkami sąsiednich glifów — ograniczenie eliminuje takie artefakty.
LABEL_RE = re.compile(r"^\d{1,3}(/\d{1,2})?$")

# Filtr minimalnej pewności OCR. Eksperymentalnie wybrany tak, by odrzucić
# typowe sklejki (conf ~0.70) przy zachowaniu wszystkich realnych etykiet.
OCR_CONF_MIN = 0.75


def parcel_label_matches(s: str) -> bool:
    return bool(LABEL_RE.match(s.strip()))


# ---------------------------------------------------------------------------
# Polygonizacja granic (wersja wektorowa, odporna na mikro-szczeliny)
# ---------------------------------------------------------------------------

def polygonize_boundaries(greens, snap_tol: float = 3.0) -> list[Polygon]:
    """Zbuduj listę Polygon-ów z ścieżek zielonych."""
    segs: list[LineString] = []
    for d in greens:
        for p1, p2 in iter_segments(d):
            if p1 == p2:
                continue
            segs.append(LineString([p1, p2]))
    if not segs:
        return []
    mls = MultiLineString(segs)
    if snap_tol > 0:
        mls = snap(mls, mls, snap_tol)
    noded = unary_union(mls)
    return [p for p in polygonize(noded) if p.is_valid and p.area > 0]


def build_red_union(reds) -> "unary_union":
    """Złóż wszystkie czerwone odcinki trasy w jeden geometryczny obiekt."""
    segs = []
    for d in reds:
        for p1, p2 in iter_segments(d):
            if p1 == p2:
                continue
            segs.append(LineString([p1, p2]))
    if not segs:
        return LineString()  # empty
    return unary_union(MultiLineString(segs))


# ---------------------------------------------------------------------------
# OCR etykiet
# ---------------------------------------------------------------------------

@dataclass
class Label:
    text: str          # np. "296/3"
    conf: float        # 0..1, pewność OCR
    x: float           # środek glifu w punktach PDF
    y: float


def _render_green_for_ocr(page, scale: int) -> np.ndarray:
    """Renderuj stronę z izolowanym zielonym tuszem na białym tle (best dla OCR)."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (g > 80) & (r < 120) & (b < 120)
    out = np.full(arr.shape[:2], 255, np.uint8)
    out[mask] = 0
    return out


def ocr_labels(page, reader, scale: int = 8, chunk_w: int = 4000,
               overlap: int = 400) -> list[Label]:
    """OCR całej strony w paskach z zakładką; deduplikacja + filtr regex."""
    proc = _render_green_for_ocr(page, scale)
    H, W = proc.shape
    raw = []
    xs = 0
    while xs < W:
        end = min(xs + chunk_w, W)
        chunk = proc[:, xs:end]
        rgb = np.stack([chunk, chunk, chunk], axis=-1)
        hits = reader.readtext(
            rgb,
            allowlist="0123456789/",
            min_size=3,
            paragraph=False,
            text_threshold=0.3,
            low_text=0.2,
            link_threshold=0.4,
        )
        for bbox, txt, conf in hits:
            raw.append({
                "text": txt,
                "conf": float(conf),
                "bbox": [(p[0] + xs, p[1]) for p in bbox],
            })
        if end == W:
            break
        xs += chunk_w - overlap

    def bbox_center(b):
        xs_ = [p[0] for p in b]
        ys_ = [p[1] for p in b]
        return sum(xs_) / len(xs_), sum(ys_) / len(ys_)

    # Deduplikacja nakładek paskowych
    uniq = []
    for r in raw:
        cx, cy = bbox_center(r["bbox"])
        dup_idx = None
        for j, u in enumerate(uniq):
            ux, uy = bbox_center(u["bbox"])
            if abs(cx - ux) < 50 and abs(cy - uy) < 30:
                dup_idx = j
                break
        if dup_idx is None:
            uniq.append(dict(r))
        else:
            u = uniq[dup_idx]
            if (len(r["text"]), r["conf"]) > (len(u["text"]), u["conf"]):
                uniq[dup_idx] = dict(r)

    out: list[Label] = []
    for r in uniq:
        t = r["text"].strip()
        if not parcel_label_matches(t):
            continue
        if r["conf"] < OCR_CONF_MIN:
            continue
        cx, cy = bbox_center(r["bbox"])
        out.append(Label(text=t, conf=r["conf"],
                         x=cx / scale, y=cy / scale))
    return out


# ---------------------------------------------------------------------------
# Główna logika: crossed vs borderline
# ---------------------------------------------------------------------------

@dataclass
class Result:
    crossed: list[str] = field(default_factory=list)      # pewne przecięcia
    borderline: list[str] = field(default_factory=list)   # do weryfikacji
    debug: dict = field(default_factory=dict)


def _route_s(red_union, x, y) -> float:
    """Przybliżony arc-length punktu (x,y) rzutowanego na trasę."""
    try:
        return float(red_union.project(Point(x, y)))
    except Exception:
        return float(x)  # fallback: trasa biegnie lewo→prawo


def _route_endpoints(red_union) -> list[tuple[float, float]]:
    """Zwraca skrajne punkty trasy (początek/koniec) po projekcji na x."""
    if red_union.is_empty:
        return []
    coords = []
    if red_union.geom_type == "LineString":
        coords = list(red_union.coords)
    elif red_union.geom_type == "MultiLineString":
        for g in red_union.geoms:
            coords.extend(g.coords)
    if not coords:
        return []
    coords.sort()
    return [coords[0], coords[-1]]


def classify(
    polygons: list[Polygon],
    labels: list[Label],
    red_union,
    interior_len_min: float = 0.5,
    corridor_dist_pt: float = 100.0,
    endpoint_dist_pt: float = 250.0,
    borderline_dist_pt: float = 10.0,
) -> tuple[list[tuple[str, Label, float]], list[tuple[str, Label, float]]]:
    """Zwraca (crossed, borderline) jako listy (nr, Label, arc_s).

    • CROSSED: etykieta jest wewnątrz wielokąta, który trasa przecina
      *w swoim wnętrzu* (długość przecięcia > `interior_len_min` po
      `buffer(-0.5)` — odrzuca grazing brzegu).
    • BORDERLINE: etykieta nie jest w żadnym przeciętym wielokącie, ale
      (a) leży blisko trasy (≤corridor_dist_pt pt), LUB
      (b) leży blisko końca trasy (≤endpoint_dist_pt pt) — etykieta za
          końcówką odcinka drogi, która legalnie może być objęta pasem
          drogowym, LUB
      (c) wielokąt etykiety jedynie MUŚNIE trasę w tolerancji
          `borderline_dist_pt`.
      Te przypadki użytkownik ma sprawdzić ręcznie.
    """
    endpoints = _route_endpoints(red_union)
    # Dla każdego wielokąta wylicz długość przecięcia wnętrza z trasą
    crossed_poly: dict[int, float] = {}
    boundary_touch: dict[int, float] = {}
    for i, poly in enumerate(polygons):
        if not poly.is_valid:
            continue
        # przecięcie wnętrza — buffer(-0.5) eliminuje grazing
        if poly.area > 5:
            interior = poly.buffer(-0.5)
            if not interior.is_empty:
                inter = interior.intersection(red_union)
                il = inter.length if hasattr(inter, "length") else 0.0
                if il >= interior_len_min:
                    crossed_poly[i] = il
                    continue
        # grazing brzegu — może być borderline
        d = poly.distance(red_union)
        if d <= borderline_dist_pt:
            boundary_touch[i] = d

    # Dla każdej etykiety znajdź zawierający wielokąt
    # (gdy >1 label na wielokąt, zatrzymujemy po najwyższej conf)
    seen_poly_crossed: set[int] = set()
    crossed_out: list[tuple[str, Label, float]] = []
    borderline_out: list[tuple[str, Label, float]] = []

    # Posortuj etykiety po (conf, len) malejąco — preferujemy pewniejsze OCR-y
    for lbl in sorted(labels, key=lambda L: (-L.conf, -len(L.text))):
        pt = Point(lbl.x, lbl.y)
        found_poly = None
        for i, poly in enumerate(polygons):
            if poly.contains(pt):
                found_poly = i
                break
        s = _route_s(red_union, lbl.x, lbl.y)

        if found_poly is not None and found_poly in crossed_poly:
            if found_poly in seen_poly_crossed:
                continue
            seen_poly_crossed.add(found_poly)
            crossed_out.append((lbl.text, lbl, s))
        elif found_poly is not None and found_poly in boundary_touch:
            borderline_out.append((lbl.text, lbl, s))
        else:
            # etykieta poza zamkniętymi wielokątami — może być w pasie drogowym
            d_route = red_union.distance(pt)
            d_end = min(
                (((lbl.x - ex) ** 2 + (lbl.y - ey) ** 2) ** 0.5
                 for ex, ey in endpoints),
                default=float("inf"),
            )
            if d_route <= corridor_dist_pt or d_end <= endpoint_dist_pt:
                borderline_out.append((lbl.text, lbl, s))

    # Deduplikacja borderline po `text` — różne OCR-owe duplikaty tego
    # samego numeru (np. 280 w dwóch miejscach) obsługuje się tu: jeżeli
    # dany tekst już jest w crossed, wyrzucamy go z borderline; jeżeli
    # występuje kilkukrotnie w borderline, zostawiamy tę najbliższą trasy.
    crossed_texts = {t for t, _, _ in crossed_out}
    bd_by_text: dict[str, tuple[str, Label, float]] = {}
    for t, lbl, s in borderline_out:
        if t in crossed_texts:
            continue
        if t not in bd_by_text:
            bd_by_text[t] = (t, lbl, s)
        else:
            # wybieramy instancję bliższą trasy
            prev = bd_by_text[t][1]
            if red_union.distance(Point(lbl.x, lbl.y)) < \
                    red_union.distance(Point(prev.x, prev.y)):
                bd_by_text[t] = (t, lbl, s)

    crossed_out.sort(key=lambda x: x[2])
    bd_list = sorted(bd_by_text.values(), key=lambda x: x[2])
    return crossed_out, bd_list


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analyze(
    pdf_path: str | Path,
    *,
    ocr_scale: int = 8,
    snap_tol: float = 3.0,
    corridor_dist_pt: float = 100.0,
    endpoint_dist_pt: float = 250.0,
    borderline_dist_pt: float = 10.0,
    ocr_cache: str | Path | None = None,
    reader=None,
) -> Result:
    """Główna procedura analizy PDF-a; zwraca `Result` z dwiema listami."""
    log = logging.getLogger("analyze")
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    page = doc[0]

    greens, reds = extract_paths(page)
    log.info("greens=%d reds=%d", len(greens), len(reds))

    polygons = polygonize_boundaries(greens, snap_tol=snap_tol)
    red_union = build_red_union(reds)
    log.info("polygons=%d, red union type=%s",
             len(polygons), red_union.geom_type)

    # OCR z cache'u lub na żywo
    labels: list[Label] | None = None
    if ocr_cache:
        p = Path(ocr_cache)
        if p.exists():
            with p.open("rb") as f:
                raw = pickle.load(f)
            labels = []
            for r in raw:
                # Filtr: poprawny format numeru + minimalna pewność OCR
                if not parcel_label_matches(r["text"]):
                    continue
                if r.get("conf", 0) < OCR_CONF_MIN:
                    continue
                # akceptujemy dwa formaty cache'a: z "cx/cy" (w pikselach
                # przy OCR_SCALE) lub z "x/y" (już w punktach PDF)
                if "x" in r and "y" in r:
                    labels.append(Label(text=r["text"], conf=r["conf"],
                                        x=r["x"], y=r["y"]))
                elif "cx" in r and "cy" in r:
                    labels.append(Label(text=r["text"], conf=r["conf"],
                                        x=r["cx"] / ocr_scale,
                                        y=r["cy"] / ocr_scale))
            log.info("OCR cache loaded: %d labels", len(labels))
    if labels is None:
        if reader is None:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        labels = ocr_labels(page, reader, scale=ocr_scale)
        log.info("OCR fresh: %d labels", len(labels))
        if ocr_cache:
            Path(ocr_cache).parent.mkdir(parents=True, exist_ok=True)
            with Path(ocr_cache).open("wb") as f:
                pickle.dump(
                    [{"text": l.text, "conf": l.conf,
                      "x": l.x, "y": l.y} for l in labels],
                    f,
                )

    crossed, borderline = classify(
        polygons, labels, red_union,
        corridor_dist_pt=corridor_dist_pt,
        endpoint_dist_pt=endpoint_dist_pt,
        borderline_dist_pt=borderline_dist_pt,
    )

    log.info("crossed=%d borderline=%d (%.1fs total)",
             len(crossed), len(borderline), time.time() - t0)

    return Result(
        crossed=[t for t, _, _ in crossed],
        borderline=[t for t, _, _ in borderline],
        debug={
            "n_polygons": len(polygons),
            "n_labels": len(labels),
            "crossed_detail": [
                {"text": t, "x": l.x, "y": l.y, "s": s}
                for t, l, s in crossed
            ],
            "borderline_detail": [
                {"text": t, "x": l.x, "y": l.y, "s": s}
                for t, l, s in borderline
            ],
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Wykrywa działki przecięte czerwoną linią na mapie PDF-a."
    )
    p.add_argument("pdf", help="ścieżka do PDF-a z mapą")
    p.add_argument("--ocr-scale", type=int, default=8)
    p.add_argument("--snap-tol", type=float, default=3.0,
                   help="tolerancja snap() przy polygonizacji (pt)")
    p.add_argument("--corridor-dist", type=float, default=100.0,
                   help="maks. odległość etykiety od trasy (pt), by "
                        "trafiła jako 'borderline'")
    p.add_argument("--endpoint-dist", type=float, default=250.0,
                   help="maks. odległość etykiety od końca trasy (pt), "
                        "by trafiła jako 'borderline' (działki przy "
                        "końcówkach pasa drogowego)")
    p.add_argument("--borderline-dist", type=float, default=10.0,
                   help="tolerancja grazingu brzegu wielokąta (pt)")
    p.add_argument("--ocr-cache", default=None,
                   help="plik pickle z cache'em OCR (zapis jeśli nie "
                        "istnieje, odczyt jeśli istnieje)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    res = analyze(
        args.pdf,
        ocr_scale=args.ocr_scale,
        snap_tol=args.snap_tol,
        corridor_dist_pt=args.corridor_dist,
        endpoint_dist_pt=args.endpoint_dist,
        borderline_dist_pt=args.borderline_dist,
        ocr_cache=args.ocr_cache,
    )

    print(f"DZIAŁKI PRZECIĘTE ({len(res.crossed)}):")
    print("  " + (", ".join(res.crossed) if res.crossed else "—"))
    print(f"DZIAŁKI DO SPRAWDZENIA ({len(res.borderline)}):")
    print("  " + (", ".join(res.borderline) if res.borderline else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
