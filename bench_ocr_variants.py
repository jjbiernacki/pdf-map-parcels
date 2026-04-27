"""Benchmark wariantów OCR vs ground truth.

Dla każdego wariantu generuje cache OCR per-mapa, mierzy czas, uruchamia
analyze_hybrid i porównuje z GT. Drukuje tabelę: wariant × mapa × (czas, FP, FN).

Wariant uznajemy za POZYTYWNY jeśli:
  - dla każdej mapy NOT-xfail: 0 FP, 0 FN
  - dla map xfail: dopuszczamy znane FP (ale BEZ NOWYCH FN)

Usage:
    python bench_ocr_variants.py              # wszystkie warianty
    python bench_ocr_variants.py B            # tylko wariant B
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import analyze_hybrid
import run_all_maps

VARIANTS = {
    "A_baseline": dict(enable_pass2=True, enable_high_scale=True),
    "B_no_pass2": dict(enable_pass2=False, enable_high_scale=True),
    "C_no_high":  dict(enable_pass2=True, enable_high_scale=False),
    "D_minimal":  dict(enable_pass2=False, enable_high_scale=False),
}

# Znane xfail z testów — Grochowska ma 2 FP (420, 501) ale 0 FN
KNOWN_FP = {
    "PZT Grochowska-Model": {"420", "501"},
}

MAPY_DIR = ROOT / "Mapy"
GT = run_all_maps.parse_gt(MAPY_DIR / "Działki na mapach.txt")
CACHE_ROOT = Path("/tmp/mapy_ocr_variants")


def _build(variant_name: str, kwargs: dict, pdfs: list[Path]) -> dict:
    """Wygeneruj cache dla wszystkich PDFów w wariancie. Zwróć timing dict."""
    v_dir = CACHE_ROOT / variant_name
    v_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    for pdf in pdfs:
        cache = v_dir / f"{pdf.stem}.pkl"
        if cache.exists():
            timings[pdf.stem] = {"cached": True, "total_s": 0.0}
            continue
        t = {}
        t0 = time.time()
        try:
            run_all_maps.build_ocr_cache(pdf, cache, timings_out=t, **kwargs)
        except Exception as e:
            print(f"  ! {variant_name} {pdf.stem} FAILED: {e}", file=sys.stderr)
            timings[pdf.stem] = {"error": str(e)}
            continue
        t["total_s"] = time.time() - t0
        timings[pdf.stem] = t
        print(f"  · {variant_name} {pdf.stem}: {t['total_s']:.1f}s "
              f"(p1={t.get('pass1', 0):.1f} p2={t.get('pass2', 0):.1f} "
              f"p3={t.get('pass3', 0):.1f} hi={t.get('pass_high', 0):.1f})",
              flush=True)
    return timings


def _evaluate(variant_name: str, pdfs: list[Path]) -> dict:
    """Uruchom analyze_hybrid dla każdego PDFa w wariancie. Zwróć dict per-mapa."""
    v_dir = CACHE_ROOT / variant_name
    out = {}
    for pdf in pdfs:
        cache = v_dir / f"{pdf.stem}.pkl"
        if not cache.exists():
            out[pdf.stem] = {"error": "no cache"}
            continue
        gt = GT.get(pdf.stem, set())
        try:
            res = analyze_hybrid.analyze(str(pdf), ocr_cache=str(cache))
        except Exception as e:
            out[pdf.stem] = {"error": str(e)}
            continue
        got = set(res.crossed)
        out[pdf.stem] = {
            "tp": sorted(got & gt),
            "fp": sorted(got - gt),
            "fn": sorted(gt - got),
            "got": sorted(got),
            "gt": sorted(gt),
        }
    return out


def main(argv):
    pdfs = sorted(p for p in MAPY_DIR.glob("*.pdf") if p.stem in GT)
    chosen = argv[1:] if len(argv) > 1 else list(VARIANTS.keys())
    print(f"Maps with GT: {len(pdfs)} ({', '.join(p.stem for p in pdfs)})")
    print(f"Variants: {chosen}\n")

    all_timings = {}
    all_evals = {}
    for v in chosen:
        if v not in VARIANTS:
            print(f"!! unknown variant {v}, skipping")
            continue
        print(f"\n=== Build OCR caches: {v} ({VARIANTS[v]}) ===")
        all_timings[v] = _build(v, VARIANTS[v], pdfs)
        print(f"\n=== Evaluate: {v} ===")
        all_evals[v] = _evaluate(v, pdfs)

    # podsumowanie tabela
    print("\n\n========== SUMMARY ==========")
    print(f"{'map':<25} | {'variant':<12} | {'time':>6} | "
          f"{'TP':>3}/{'GT':>3} | {'FP':<10} | {'FN':<10} | {'OK?':<3}")
    print("-" * 100)
    for pdf in pdfs:
        for v in chosen:
            t = all_timings[v].get(pdf.stem, {})
            e = all_evals[v].get(pdf.stem, {})
            time_s = t.get("total_s", 0)
            cached_mark = " (c)" if t.get("cached") else ""
            tp = len(e.get("tp", []))
            gt = len(e.get("gt", []))
            fp = e.get("fp", [])
            fn = e.get("fn", [])
            known = KNOWN_FP.get(pdf.stem, set())
            unexpected_fp = set(fp) - known
            ok = (not unexpected_fp) and (not fn)
            mark = "✓" if ok else "✗"
            print(f"{pdf.stem:<25} | {v:<12} | "
                  f"{time_s:>5.1f}s{cached_mark} | {tp:>3}/{gt:>3} | "
                  f"{','.join(fp) if fp else '—':<10} | "
                  f"{','.join(fn) if fn else '—':<10} | {mark}")

    print("\n========== TOTALS ==========")
    for v in chosen:
        total = sum(t.get("total_s", 0) for t in all_timings[v].values())
        any_fail = False
        for pdf in pdfs:
            e = all_evals[v].get(pdf.stem, {})
            known = KNOWN_FP.get(pdf.stem, set())
            unexpected_fp = set(e.get("fp", [])) - known
            if unexpected_fp or e.get("fn"):
                any_fail = True
                break
        verdict = "PASS" if not any_fail else "FAIL"
        print(f"  {v:<12} : OCR total = {total:>6.1f}s  → {verdict}")


if __name__ == "__main__":
    main(sys.argv)
