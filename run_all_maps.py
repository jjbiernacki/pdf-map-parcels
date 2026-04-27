"""Uruchom analyze_hybrid dla każdej mapy w katalogu Mapy/
i porównaj z GT z pliku "Działki na mapach.txt".

Usage:
    python run_all_maps.py [--rebuild-ocr] [-v]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import analyze_hybrid
import fitz

MAPY_DIR = Path("Mapy")
GT_FILE = MAPY_DIR / "Działki na mapach.txt"
OCR_CACHE_DIR = Path("/tmp/mapy_ocr")


def parse_gt(path: Path) -> dict[str, set[str]]:
    """Parsuje `Działki na mapach.txt` na dict {nazwa_pdf: set(działki)}.

    Format:
        <NAZWA_PDF>:
        <dziaka1>
        <działka2>                       # każda w osobnym wierszu, LUB
        <działka3>, <działka4>, ...      # kilka działek w jednym wierszu
        ...
        (pusty wiersz rozdziela mapy)
    """
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
            # obsłuż wiersz z wieloma działkami oddzielonymi przecinkiem
            for token in line.split(","):
                t = token.strip()
                if t:
                    result[current].add(t)
    return result


def _easyocr_read(proc, reader, chunk_w=4000, overlap=400):
    """OCR Multi-pass: kombinacja chunk-size i progów żeby znaleźć etykiety
    których EasyOCR mógłby przegapić w pojedynczym podejściu.

    Pass 1: standardowe parametry (chunk 4000, threshold 0.3) — szybkie
    Pass 2: ciaśniejsze chunki + niższe progi (chunk 2500, threshold 0.2)
            — łapie etykiety blisko brzegów chunków, niskim kontraście,
              częściowo zasłonięte
    Wyniki obu passów scalamy przez deduplikację pozycyjną (bbox center
    < 30px → ten sam label, wybór wyższego conf).
    """
    import numpy as np
    H, W = proc.shape

    def _pass(chunk_w_, overlap_, text_thresh, low_text):
        raw_ = []
        xs = 0
        while xs < W:
            end = min(xs + chunk_w_, W)
            chunk = proc[:, xs:end]
            rgb = np.stack([chunk, chunk, chunk], axis=-1)
            hits = reader.readtext(
                rgb, allowlist="0123456789/", min_size=3, paragraph=False,
                text_threshold=text_thresh, low_text=low_text,
                link_threshold=0.4,
            )
            for bbox, txt, conf in hits:
                raw_.append({"text": txt, "conf": float(conf),
                             "bbox": [(p[0] + xs, p[1]) for p in bbox],
                             "source": "easy"})
            if end == W:
                break
            xs += chunk_w_ - overlap_
        return raw_

    raw = _pass(chunk_w, overlap, 0.3, 0.2)
    raw2 = _pass(2500, 600, 0.2, 0.1)
    raw3 = _pass(1500, 400, 0.15, 0.08)

    def _center(b):
        xs_ = [p[0] for p in b]; ys_ = [p[1] for p in b]
        return sum(xs_) / 4, sum(ys_) / 4

    # Scal wszystkie passy: dla każdej kolejnej, dodaj hit jeśli nie ma
    # duplikatu (ten sam tekst w promieniu 30px). W przypadku duplikatu —
    # zachowaj wyższy conf.
    out = list(raw)
    centers = [_center(r["bbox"]) for r in out]
    for extras in (raw2, raw3):
        for r2 in extras:
            c2 = _center(r2["bbox"])
            is_dup = False
            for i, c1 in enumerate(centers):
                if abs(c1[0]-c2[0]) < 30 and abs(c1[1]-c2[1]) < 30:
                    if out[i]["text"].strip() == r2["text"].strip():
                        if r2["conf"] > out[i]["conf"]:
                            out[i] = r2
                            centers[i] = c2
                        is_dup = True
                        break
            if not is_dup:
                out.append(r2)
                centers.append(c2)
    return out


def _easyocr_read_low_threshold(proc, reader, chunk_w=2000, overlap=400):
    """Pomocniczy OCR pass przy bardzo niskich progach detekcji.
    Używany na rendrze WYŻSZEJ rozdzielczości (scale=12) dla regionów gdzie
    standardowy multi-pass nie znajduje etykiet (np. Grochowska 421 — etykieta
    blisko czerwonej trasy).
    """
    import numpy as np
    H, W = proc.shape
    raw = []
    xs = 0
    while xs < W:
        end = min(xs + chunk_w, W)
        chunk = proc[:, xs:end]
        rgb = np.stack([chunk, chunk, chunk], axis=-1)
        hits = reader.readtext(
            rgb, allowlist="0123456789/", min_size=2, paragraph=False,
            text_threshold=0.05, low_text=0.02, link_threshold=0.4,
        )
        for bbox, txt, conf in hits:
            raw.append({"text": txt, "conf": float(conf),
                        "bbox": [(p[0] + xs, p[1]) for p in bbox],
                        "source": "easy_high"})
        if end == W:
            break
        xs += chunk_w - overlap
    return raw


def _tesseract_read(proc, chunk_w=4000, overlap=400):
    """Tesseract OCR z psm 11 (sparse text), z chunkowaniem żeby obsłużyć
    bardzo szerokie mapy (tesseract ma limit rozmiaru ~32k px)."""
    import pytesseract
    H, W = proc.shape
    raw = []
    xs = 0
    while xs < W:
        end = min(xs + chunk_w, W)
        chunk = proc[:, xs:end]
        data = pytesseract.image_to_data(
            chunk, lang="eng",
            config="--psm 11 -c tessedit_char_whitelist=0123456789/",
            output_type=pytesseract.Output.DICT,
        )
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            if not txt:
                continue
            conf = int(data["conf"][i])
            if conf <= 0:
                continue
            x = data["left"][i] + xs
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            bbox = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            raw.append({"text": txt, "conf": conf / 100.0,
                        "bbox": bbox, "source": "tess"})
        if end == W:
            break
        xs += chunk_w - overlap
    return raw


def build_ocr_cache(pdf_path: Path, cache_path: Path, ocr_scale: int = 8,
                    use_easyocr: bool = True, use_tesseract: bool = False) -> None:
    """Uruchamia EasyOCR multi-pass i opcjonalnie Tesseract, scala wyniki
    i zapisuje jako pickle.

    Scalanie: wyniki są deduplikowane po pozycji (promień 30px). W przypadku
    duplikatu wybieramy ten o WYŻSZYM CONF. Gdy easyocr zwraca artefakt typu
    "7260" (nie pasuje do regex parcel-label) a tesseract zwraca "260" w tej
    pozycji, wygrywa "260".
    """
    from analyze import _render_green_for_ocr
    import re
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    proc = _render_green_for_ocr(page, ocr_scale)

    raw = []
    if use_easyocr:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        raw.extend(_easyocr_read(proc, reader))
        # Dodatkowy pass przy WYŻSZEJ rozdzielczości (scale=12) dla regionów
        # gdzie etykiety są blisko trasy lub mają niski kontrast — np.
        # Grochowska 421 (czerwona linia tuż obok zielonej etykiety, EasyOCR
        # przy scale=8 z text_threshold=0.15 nie znajduje, przy scale=12 +
        # threshold=0.05 znajduje).
        # Bbox-y skalujemy z powrotem do scale=ocr_scale przy zapisie do cache.
        high_scale = 12
        proc_high = _render_green_for_ocr(page, high_scale)
        scale_ratio = ocr_scale / high_scale  # 8/12 = 0.667
        high_raw = _easyocr_read_low_threshold(proc_high, reader)
        for r in high_raw:
            r["bbox"] = [(p[0] * scale_ratio, p[1] * scale_ratio)
                         for p in r["bbox"]]
            raw.append(r)
    if use_tesseract:
        raw.extend(_tesseract_read(proc))

    def bbox_center(b):
        xs_ = [p[0] for p in b]
        ys_ = [p[1] for p in b]
        return sum(xs_) / len(xs_), sum(ys_) / len(ys_)

    PARCEL_RE = re.compile(r"^\d{1,3}(/\d{1,2})?$")

    def parcel_match(t):
        return bool(PARCEL_RE.match(t.strip()))

    # Deduplikacja dwufazowa:
    # (1) easyocr jest primary źródłem. Tesseract uzupełnia tylko te
    #     miejsca GDZIE easyocr:
    #       - nic nie znalazł (brak etykiety w pozycji), LUB
    #       - znalazł tekst NIE pasujący do regex parcel-label (artefakt
    #         typu "7260" — wtedy tesseract może poprawić).
    # (2) Dla tego samego tekstu w bliskiej pozycji — wyższy conf wygrywa.
    easy_raw = [r for r in raw if r.get("source") in ("easy", "easy_high")]
    tess_raw = [r for r in raw if r.get("source") == "tess"]

    def nearby(a, b, r=30):
        ax, ay = bbox_center(a["bbox"])
        bx, by = bbox_center(b["bbox"])
        return abs(ax - bx) < r and abs(ay - by) < r

    # Post-process easyocr artefactów: EasyOCR na niektórych mapach
    # doradza "7" przed numerem działki (np. "7260" zamiast "260", "7429/2"
    # zamiast "429/2"). Jeśli strip leading "7" daje valid parcel, to zrób to.
    # Podobnie sklejone numery bez "/" (np. "6052" = "605/2").
    def _repair_easy(txt):
        s = txt.strip()
        if parcel_match(s):
            return s
        # leading "7" artifact
        if s.startswith("7") and len(s) > 1 and parcel_match(s[1:]):
            return s[1:]
        # missing "/" in 4-digit numbers that look like NNN+M
        if re.fullmatch(r"\d{4}", s):
            candidate = f"{s[:3]}/{s[3]}"
            if parcel_match(candidate):
                return candidate
        # trailing "/" with nothing after (e.g., "7478/")
        if s.endswith("/"):
            return _repair_easy(s[:-1])
        return s

    # najpierw dedup w ramach easy (po tym samym tekście) + repair artefactów
    uniq = []
    for r in easy_raw:
        rr = dict(r)
        rr["text"] = _repair_easy(rr["text"])
        dup = next((j for j, u in enumerate(uniq)
                    if nearby(rr, u) and rr["text"].strip() == u["text"].strip()), None)
        if dup is None:
            uniq.append(rr)
        elif rr["conf"] > uniq[dup]["conf"]:
            uniq[dup] = rr

    # dołącz tesseract tylko jeśli:
    #   a) w tej pozycji nie ma easyocr wyniku (uniq), LUB
    #   b) jedyny easyocr w tej pozycji NIE pasuje do regex parcel
    # I dodatkowo: tesseract conf >= 0.3 (filtr szumu)
    for r in tess_raw:
        if r["conf"] < 0.3:
            continue
        if not parcel_match(r["text"]):
            continue
        # znajdź easyocr w tej pozycji
        easy_near = [u for u in uniq if nearby(r, u, r=40)]
        if not easy_near:
            # brak easyocr — dodaj tesseract
            uniq.append(dict(r))
            continue
        # jeśli którykolwiek easyocr w tej pozycji PASUJE do regex → pomiń
        # tesseract (easyocr trafił, tesseract może się mylić)
        if any(parcel_match(u["text"]) for u in easy_near):
            continue
        # żaden easyocr nie pasuje (artefakt typu "7260") — zastąp
        # go tesseractem
        uniq.append(dict(r))

    out = []
    for r in uniq:
        cx, cy = bbox_center(r["bbox"])
        out.append({
            "text": r["text"].strip(),
            "conf": r["conf"],
            "cx": cx, "cy": cy,
            "bbox": r["bbox"],
            "source": r.get("source", "unknown"),
        })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(out, f)


def ensure_ocr_cache(pdf_path: Path, rebuild: bool = False) -> Path:
    cache_path = OCR_CACHE_DIR / (pdf_path.stem + ".pkl")
    if rebuild or not cache_path.exists():
        print(f"  [OCR] {pdf_path.name} → {cache_path.name} ...", end="", flush=True)
        t0 = time.time()
        build_ocr_cache(pdf_path, cache_path)
        print(f" {time.time()-t0:.1f}s")
    return cache_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-ocr", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    gt_by_name = parse_gt(GT_FILE)
    all_perfect = True
    totals = {"TP": 0, "FP": 0, "FN": 0, "GT": 0}
    for pdf_path in sorted(MAPY_DIR.glob("*.pdf")):
        name = pdf_path.stem
        gt = gt_by_name.get(name, set())
        if not gt:
            print(f"{name}: no GT, skipping")
            continue
        cache = ensure_ocr_cache(pdf_path, rebuild=args.rebuild_ocr)
        t0 = time.time()
        res = analyze_hybrid.analyze(str(pdf_path), ocr_cache=str(cache))
        elapsed = time.time() - t0
        got = set(res.crossed)
        tp = got & gt; fp = got - gt; fn = gt - got
        totals["TP"] += len(tp); totals["FP"] += len(fp); totals["FN"] += len(fn)
        totals["GT"] += len(gt)
        mark = "✓" if not fp and not fn else "✗"
        print(f"{mark} {name}:  TP {len(tp):>2}/{len(gt):>2}  FP {len(fp):>2}  FN {len(fn):>2}  ({elapsed:.1f}s)")
        if fp:
            print(f"    FP: {sorted(fp)}")
        if fn:
            print(f"    FN: {sorted(fn)}")
        if fp or fn:
            all_perfect = False

    print()
    print(f"TOTAL:  TP {totals['TP']}/{totals['GT']}  FP {totals['FP']}  FN {totals['FN']}")
    return 0 if all_perfect else 1


if __name__ == "__main__":
    sys.exit(main())
