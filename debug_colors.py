"""Debug: znajdź kolory stroków (szczególnie różowy) w nowych PDF-ach."""
import fitz
from collections import Counter

for name in ["Grochowska", "Kurka", "Polna"]:
    path = f"Mapy/PZT {name}-Model.pdf"
    print(f"\n=== {name} ===")
    doc = fitz.open(path)
    page = doc[0]
    print(f"Page: {page.rect}")
    draws = page.get_drawings()
    print(f"Total drawings: {len(draws)}")
    colors = Counter()
    for d in draws:
        c = d.get("color")
        w = d.get("width")
        if c is not None:
            rgb = tuple(round(v, 3) for v in c)
            colors[(rgb, round(w or 0, 2))] += 1
    # Show distinct colors
    print("Distinct (color, width) counts:")
    for (c, w), n in sorted(colors.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:4d} × stroke={c} width={w}")
