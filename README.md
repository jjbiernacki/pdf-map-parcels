# pdf-map-parcels

> Detect which land parcels (działki) are crossed by a route line on a
> vectorized geodetic PDF map.

A pipeline that takes a Polish PZT map (PDF with green parcel boundaries
and a red dashed road-corridor line) and returns the **exact list of parcel
numbers the red route passes through**.

Includes:

- **CLI** — `python analyze_hybrid.py <map.pdf>`
- **Web demo** — drag-drop UI with live progress, map preview, and a
  copy-friendly text field of the resulting parcel numbers
  ([`webapp/`](webapp/))
- **Hugging Face Spaces deploy** — Dockerized, free CPU tier
  ([`huggingface/`](huggingface/))

## The problem

Geodetic maps in Poland (PZT — *projekt zagospodarowania terenu*) come as
PDFs with three vector layers:

| Color | Meaning |
| ----- | ------- |
| 🟢 green `(0, 0.584, 0)` | parcel boundaries (numbered polygons) |
| 🔴 red `(1, 0, 0)`, dashed | road corridor / route |
| ⚫ black | survey pins, decorations |

Parcel numbers (e.g. `296/3`, `48/1`) are drawn as **vector glyph paths**,
not text — `page.get_text()` returns an empty string. They have to be
recovered with OCR.

The task: given the PDF, output every parcel number whose polygon is
intersected by the red route.

## How it works

`analyze_hybrid.py` runs a hybrid vector + raster pipeline:

1. Extract green / red drawings from the PDF (PyMuPDF).
2. Build green polygons via segment closure heuristics
   (T-junctions, endpoint extensions, parallel-pair closures, route-end
   buffers, page hull).
3. Polygonize with Shapely — typically 100–400 polygons per map.
4. Classify each polygon as **crossed** by 6 stacked rules:
   - boundary intersected ≥ 2 times by the route,
   - route runs in the polygon's interior,
   - boundary segments crossed by the route sum ≥ 100 pt,
   - boundary runs *along* the route within a 5 pt buffer,
   - tangent touch + concave shape (notch where route entered/exited),
   - polygon abuts the global route endpoint and is elongated / oriented
     opposite the route.
5. OCR parcel labels with **EasyOCR** (multi-pass: scale 8 + scale 12,
   chunk widths 1500/2500/4000, repair common artefacts like `7260`→`260`).
6. Match each label point to a polygon → the labels in crossed polygons
   are the answer.

Edge cases (corridor sub-parcels, route fully inside one parcel, OCR misses
near the route) have explicit fallbacks.

## Accuracy

End-to-end on 11 ground-truth maps (`Mapy/` + `numery działek.txt`):
**100% match** — every expected parcel found, no false positives.
See `tests/` for the test suite and `run_all_maps.py` for the harness.

## Quick start

```bash
git clone https://github.com/jjbiernacki/pdf-map-parcels.git
cd pdf-map-parcels

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI
python analyze_hybrid.py "Mapy/03 PZT granice.pdf"

# Run against all sample maps (uses ground truth in Mapy/)
python run_all_maps.py
```

First run downloads EasyOCR models (~110 MB) to `~/.EasyOCR/`.

## Web demo

```bash
pip install -r webapp/requirements.txt
python webapp/app.py
# open http://localhost:8000
```

Drop a PDF, watch the 9-step progress bar, get the parcel list and a
rendered map with highlighted labels.

See [`webapp/README.md`](webapp/README.md) for architecture details.

## Deploy

- **Hugging Face Spaces** (recommended, free, CPU 16 GB) —
  [`huggingface/DEPLOY.md`](huggingface/DEPLOY.md)
- **Render.com** — `webapp/render.yaml` blueprint, free tier (512 MB,
  may be tight for EasyOCR)
- **Fly.io** — `fly launch --dockerfile webapp/Dockerfile`

> ⚠️ **Vercel won't work**: serverless functions cap at 50 MB and 10–60 s,
> EasyOCR + torch is ~1.5 GB and a typical analysis takes 30–90 s.

## Project layout

```
analyze_hybrid.py     # main pipeline (vector + raster + OCR + classification)
analyze_cv.py         # OCR loader, label parsing
analyze_ray.py        # ray-casting backup classifier
analyze.py            # earlier vector-only attempt (kept for reference)
run_all_maps.py       # batch over Mapy/ + ground-truth diff
compare_gt.py         # tiny diff helper
render_*.py           # debug renderers (annotated PNGs)
tests/                # pytest: end-to-end accuracy tests
Mapy/                 # 11 sample PDFs + ground-truth parcel numbers
webapp/               # Flask + SSE web demo
huggingface/          # Docker + README for HF Spaces deploy
CEL.md                # original task brief (Polish)
PROBY.md              # research log: what didn't work and why (Polish)
```

## Stack

PyMuPDF · Shapely · NumPy · scikit-image · OpenCV · EasyOCR · Flask

## License

MIT — see [LICENSE](LICENSE).
