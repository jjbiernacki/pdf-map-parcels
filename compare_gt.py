"""Uruchom analyze_hybrid i porównaj z GT. Pokaż TP/FP/FN."""
import sys
from pathlib import Path
import analyze_hybrid as analyze_cv

GT = set(line.strip() for line in Path("numery działek.txt").read_text().splitlines() if line.strip())


def main():
    kwargs = {}
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        kwargs[k] = v

    kwargs.setdefault("ocr_cache", "/tmp/ocr_cache_v2.pkl")
    res = analyze_cv.analyze("03 PZT granice.pdf", **kwargs)

    got = set(res.crossed)
    tp = got & GT
    fp = got - GT
    fn = GT - got
    print(f"TP {len(tp):2d}/{len(GT)}: {sorted(tp)}")
    print(f"FP {len(fp):2d}   : {sorted(fp)}")
    print(f"FN {len(fn):2d}   : {sorted(fn)}")
    print(f"Borderline: {res.borderline}")
    if not fp and not fn:
        print("PERFECT MATCH ✓")
    return 0


if __name__ == "__main__":
    main()
