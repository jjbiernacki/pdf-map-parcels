"""Web demo: upload mapy PDF i wypisz wybrane numery działek.

Architektura:
- Flask + SSE (Server-Sent Events) dla strumieniowania postępu
- Job state w pamięci procesu (single-instance demo)
- Reużywa istniejący pipeline z run_all_maps.build_ocr_cache + analyze_hybrid.analyze
- Renderuje PDF + nakładkę kolorową dla wybranych etykiet jako PNG (PIL)

Uruchomienie lokalne:
    python webapp/app.py
    # otwórz http://localhost:8000

Deploy: patrz webapp/README.md (Hugging Face Spaces / Render / Fly.io).
"""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import (
    Flask, Response, jsonify, render_template, request, send_file, stream_with_context,
)

# --- repo importy --------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

import analyze_hybrid
import run_all_maps
from analyze_cv import load_ocr_cache

# --- konfiguracja --------------------------------------------------------
MAX_UPLOAD_MB = 25
JOB_TTL_SECONDS = 60 * 30  # 30 min w pamięci

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# --- progres pipeline'u --------------------------------------------------
# Każdy krok ma znany udział w całości (sumują się do 100%).
# Etykiety widoczne dla użytkownika.
PIPELINE_STEPS = [
    ("upload",     "Wczytuję PDF",                     2),
    ("paths",      "Wyciągam ścieżki wektorowe",       3),
    ("ocr",        "OCR etykiet (EasyOCR — najdłuższy etap)", 60),
    ("segments",   "Buduję segmenty zielonych granic", 4),
    ("closures",   "Zamykam otwarte granice działek",  6),
    ("polygonize", "Składam wielokąty działek",         5),
    ("classify",   "Klasyfikuję działki przecięte trasą", 8),
    ("match",      "Dopasowuję etykiety do działek",    5),
    ("render",     "Renderuję mapę z zaznaczeniami",    7),
]
STEP_BY_KEY = {k: (label, w) for k, label, w in PIPELINE_STEPS}


# Mapowanie znanych komunikatów log z analyze_hybrid → klucz kroku.
# Pattern matching jest celowo prosty.
LOG_TO_STEP = [
    (re.compile(r"^paths:"),                          "paths"),
    (re.compile(r"^green segments:"),                 "segments"),
    (re.compile(r"^endpoint closures:"),              "closures"),
    (re.compile(r"^endpoint extensions:"),            "closures"),
    (re.compile(r"^parallel endpoint closures:"),     "closures"),
    (re.compile(r"^T-junction extensions:"),          "closures"),
    (re.compile(r"^pin crosslines:"),                 "closures"),
    (re.compile(r"^route-end closures:"),             "closures"),
    (re.compile(r"^route buffer frame:"),             "closures"),
    (re.compile(r"^polygons:"),                       "polygonize"),
    (re.compile(r"^polygons after area filter:"),     "polygonize"),
    (re.compile(r"^crossed green border segments:"),  "classify"),
    (re.compile(r"^road corridor polygon:"),          "classify"),
    (re.compile(r"^crossed polygons:"),               "classify"),
    (re.compile(r"^OCR labels"),                      "match"),
    (re.compile(r"^crossed=\d+ borderline=\d+"),      "match"),
]


@dataclass
class Job:
    id: str
    pdf_path: Path
    pdf_name: str
    created: float = field(default_factory=time.time)
    events: queue.Queue = field(default_factory=queue.Queue)
    done: bool = False
    error: str | None = None
    result: dict | None = None  # {crossed: [...], borderline: [...], image_path: ...}


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _gc_jobs() -> None:
    """Usuń wygasłe joby + ich pliki."""
    now = time.time()
    with JOBS_LOCK:
        stale = [jid for jid, j in JOBS.items() if now - j.created > JOB_TTL_SECONDS]
        for jid in stale:
            j = JOBS.pop(jid)
            for p in (j.pdf_path, Path(j.pdf_path.parent / f"{jid}.png")):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass


# ------------------------------------------------------------------------ #
# logging handler tłumaczący log analyze_hybrid → eventy SSE
# ------------------------------------------------------------------------ #
class JobProgressHandler(logging.Handler):
    def __init__(self, job: Job):
        super().__init__(level=logging.INFO)
        self.job = job
        self.completed_steps: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for pat, key in LOG_TO_STEP:
            if pat.match(msg):
                push_step(self.job, key, detail=msg, complete=True)
                return


# ------------------------------------------------------------------------ #
# event helpers
# ------------------------------------------------------------------------ #
def push_event(job: Job, kind: str, **payload) -> None:
    job.events.put({"type": kind, **payload})


def push_step(job: Job, key: str, *, detail: str = "", complete: bool = False) -> None:
    """Wyślij update postępu — oblicz % po wagach kroków."""
    label, weight = STEP_BY_KEY.get(key, (key, 0))
    push_event(job, "step", step=key, label=label, detail=detail, complete=complete)
    # progres skumulowany — suma wag kroków zakończonych + bieżący
    if complete:
        push_event(job, "progress", percent=_compute_percent(job, key, complete=True))
    else:
        push_event(job, "progress", percent=_compute_percent(job, key, complete=False))


# ślad ukończonych kroków per-job (handler aktualizuje przez completed_steps,
# ale dla push_step "ręcznych" trzymamy prosty zbiór w obiekcie Joba)
_JOB_DONE_KEYS: dict[str, set[str]] = {}
_JOB_DONE_LOCK = threading.Lock()


def _compute_percent(job: Job, current_key: str, complete: bool) -> int:
    with _JOB_DONE_LOCK:
        done = _JOB_DONE_KEYS.setdefault(job.id, set())
        if complete:
            done.add(current_key)
        total = sum(w for _, _, w in PIPELINE_STEPS)
        accumulated = sum(w for k, _, w in [(k, l, w) for k, l, w in PIPELINE_STEPS] if k in done)
        if not complete:
            # dodaj połowę bieżącego kroku jako "w trakcie"
            cur_w = STEP_BY_KEY.get(current_key, ("", 0))[1]
            accumulated += cur_w * 0.4
    return min(99 if not complete else 100, int(round(accumulated / total * 100)))


# ------------------------------------------------------------------------ #
# pipeline w wątku
# ------------------------------------------------------------------------ #
def _run_pipeline(job: Job) -> None:
    try:
        push_step(job, "upload", complete=True)

        # 1. OCR cache (build, niezależnie od analyze)
        push_step(job, "ocr", detail="Inicjalizuję OCR…")
        ocr_dir = Path(tempfile.gettempdir()) / "mapy_ocr_web"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        ocr_cache = ocr_dir / f"{job.id}.pkl"
        t0 = time.time()
        run_all_maps.build_ocr_cache(job.pdf_path, ocr_cache)
        push_event(job, "ocr_done", seconds=round(time.time() - t0, 1))
        push_step(job, "ocr", complete=True, detail=f"OCR zakończony ({time.time()-t0:.1f}s)")

        # 2. analyze() z handlerem postępu
        log = logging.getLogger("analyze_hybrid")
        log.setLevel(logging.INFO)
        handler = JobProgressHandler(job)
        log.addHandler(handler)
        try:
            result = analyze_hybrid.analyze(str(job.pdf_path), ocr_cache=str(ocr_cache))
        finally:
            log.removeHandler(handler)

        # 3. render PNG
        push_step(job, "render", detail="Rysuję mapę z zaznaczeniami…")
        png_path = job.pdf_path.parent / f"{job.id}.png"
        render_result_png(job.pdf_path, ocr_cache, set(result.crossed),
                          set(result.borderline), png_path)
        push_step(job, "render", complete=True)

        job.result = {
            "crossed": result.crossed,
            "borderline": result.borderline,
            "image_url": f"/result/{job.id}/image",
        }
        push_event(job, "done",
                   crossed=result.crossed, borderline=result.borderline,
                   image_url=job.result["image_url"])
    except Exception as e:
        job.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        push_event(job, "error", message=job.error)
    finally:
        job.done = True
        # zamknięcie strumienia eventów
        push_event(job, "close")


# ------------------------------------------------------------------------ #
# render mapy z nakładką
# ------------------------------------------------------------------------ #
def render_result_png(pdf_path: Path, ocr_cache: Path,
                      crossed: set[str], borderline: set[str],
                      out_path: Path, scale: int = 3) -> None:
    """Zapisuje render PDFa + okręgi/etykiety dla wybranych działek."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    labels = load_ocr_cache(ocr_cache, ocr_scale=8)
    # kolory: wybrane = zielony jaskrawy, sporne = pomarańczowy
    PICK = (16, 185, 129, 220)        # emerald-500
    PICK_FILL = (16, 185, 129, 60)
    BORD = (245, 158, 11, 220)        # amber-500
    BORD_FILL = (245, 158, 11, 50)

    for l in labels:
        text = l.text
        if text in crossed:
            color, fill = PICK, PICK_FILL
        elif text in borderline:
            color, fill = BORD, BORD_FILL
        else:
            continue
        x, y = l.x * scale, l.y * scale
        r = 22
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4, fill=fill)
        # etykieta tekstowa obok kółka
        tx, ty = x + r + 4, y - r - 2
        # background under text for readability
        bbox = draw.textbbox((tx, ty), text, font=font)
        pad = 3
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                       fill=(255, 255, 255, 220))
        draw.text((tx, ty), text, fill=color[:3] + (255,), font=font)

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    composed.save(out_path, "PNG", optimize=True)


# ------------------------------------------------------------------------ #
# routes
# ------------------------------------------------------------------------ #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def start_analyze():
    _gc_jobs()
    f = request.files.get("pdf")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Wymagany plik PDF (.pdf)"}), 400

    job_id = uuid.uuid4().hex
    upload_dir = Path(tempfile.gettempdir()) / "mapy_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / f"{job_id}.pdf"
    f.save(pdf_path)

    job = Job(id=job_id, pdf_path=pdf_path, pdf_name=f.filename)
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_pipeline, args=(job,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/events/<job_id>")
def events(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return Response("unknown job", status=404)

    @stream_with_context
    def stream():
        # wyślij listę kroków na start
        yield _sse({"type": "init",
                    "steps": [{"key": k, "label": l} for k, l, _ in PIPELINE_STEPS]})
        while True:
            try:
                ev = job.events.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield _sse(ev)
            if ev.get("type") == "close":
                break

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/result/<job_id>/image")
def result_image(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or not job.done or job.error:
        return Response("not ready", status=404)
    png = job.pdf_path.parent / f"{job_id}.png"
    if not png.exists():
        return Response("no image", status=404)
    return send_file(png, mimetype="image/png")


@app.route("/healthz")
def healthz():
    return "ok"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
