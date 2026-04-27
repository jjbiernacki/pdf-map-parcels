"""Szybki render dowolnej mapy z adnotacjami GT/FP/FN — BEZ analyze()."""
import sys
import fitz
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from analyze_cv import load_ocr_cache, extract_paths


# hard-coded results from run_all_maps (aby uniknąć pełnego pipeline'u)
RESULTS = {
    "PZT Grochowska-Model": {
        "gt": {"421","430","440","439/3","448","465","479/4","479/2","429/2",
               "502","523","522","568","581","592","594/1","605/2","614/1"},
        "crossed": {"439/3","479/4","479/2","502","523","592","594/1","605/2",
                    "569","614"},  # 8 TP + 2 FP
    },
    "PZT Kurka-Model": {
        "gt": {"443","465","456/12","456/11","456/1","456/2","456/4","456/5"},
        "crossed": {"443","465","456/12","456/1","456/2"},
    },
    "PZT Polna-Model": {
        "gt": {"536","550/6","547/5","547/10","550/3","550/4","550/5",
               "558/2","557"},
        "crossed": {"547/10","547/5","550/3","550/4","550/5","550/6","558/2","81"},
    },
}


def render(name: str, crop_bbox=None, out_suffix=""):
    spec = RESULTS[name]
    gt = spec["gt"]; crossed = spec["crossed"]
    pdf = f"Mapy/{name}.pdf"
    cache = f"/tmp/mapy_ocr/{name}.pkl"
    doc = fitz.open(pdf); page = doc[0]

    scale = 3
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)

    labels = load_ocr_cache(Path(cache), ocr_scale=8)
    tp = gt & crossed
    fp = crossed - gt
    fn = gt - crossed
    for l in labels:
        tag, color = None, None
        if l.text in tp:
            tag, color = "TP", "green"
        elif l.text in fn:
            tag, color = "FN", "blue"
        elif l.text in fp:
            tag, color = "FP", "red"
        else:
            continue
        x, y = l.x*scale, l.y*scale
        r = 14
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=4)
        draw.text((x+r+3, y-r-2), f"{tag}:{l.text}", fill=color)

    if crop_bbox:
        img = img.crop([int(v*scale) for v in crop_bbox])

    out = f"/tmp/{name.lower().replace(' ','_').replace('/','_')}{out_suffix}.png"
    img.save(out)
    print(f"saved {out} size={img.size}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "PZT Grochowska-Model"
    render(name)
