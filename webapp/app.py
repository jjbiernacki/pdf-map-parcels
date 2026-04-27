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
import math
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
# Pięć kroków user-friendly w kolejności RZECZYWISTEGO wykonania.
# Wagi proporcjonalne do realnego czasu (OCR dominuje, ~75% wallclock).
# Sumują się do 100.
PIPELINE_STEPS = [
    ("upload",   "Wczytuję plik",                  1),
    ("ocr",      "Rozpoznaję numery działek",     75),
    ("analyze",  "Analizuję rysunek mapy",        10),
    ("classify", "Wyznaczam działki na trasie",    7),
    ("render",   "Generuję podgląd mapy",          7),
]
STEP_BY_KEY = {k: (label, w) for k, label, w in PIPELINE_STEPS}
STEP_ORDER = [k for k, _, _ in PIPELINE_STEPS]


# Mapowanie log z analyze_hybrid na user-friendly kroki.
# - "analyze"  = ścieżki wektorowe + segmenty zielonych + zamykanie luk + składanie wielokątów
# - "classify" = klasyfikacja działek + dopasowanie etykiet (ostatnia faza analyze())
LOG_TO_STEP = [
    (re.compile(r"^paths:"),                          "analyze"),
    (re.compile(r"^green segments:"),                 "analyze"),
    (re.compile(r"^endpoint closures:"),              "analyze"),
    (re.compile(r"^endpoint extensions"),             "analyze"),
    (re.compile(r"^parallel endpoint closures:"),     "analyze"),
    (re.compile(r"^T-junction extensions:"),          "analyze"),
    (re.compile(r"^pin crosslines:"),                 "analyze"),
    (re.compile(r"^route-end closures:"),             "analyze"),
    (re.compile(r"^route buffer frame:"),             "analyze"),
    (re.compile(r"^polygons:"),                       "analyze"),
    (re.compile(r"^polygons after area filter:"),     "analyze"),
    (re.compile(r"^crossed green border segments:"),  "classify"),
    (re.compile(r"^road corridor polygon:"),          "classify"),
    (re.compile(r"^crossed polygons:"),               "classify"),
    (re.compile(r"^OCR labels"),                      "classify"),
    (re.compile(r"^crossed=\d+ borderline=\d+"),      "classify"),
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
    cancelled: threading.Event = field(default_factory=threading.Event)
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
    """Mapuje log analyze_hybrid → eventy SSE.

    Działa w trybie "milestone": gdy widzimy log linię przypisaną do kroku
    PÓŹNIEJSZEGO niż obecny, zamykamy wszystkie wcześniejsze kroki i
    aktywujemy nowy. Wewnątrz tego samego kroku tylko aktualizujemy detail
    (drobne ruchy paska realizuje osobny tick w pipeline).
    """

    def __init__(self, job: Job, start_step: str = "analyze"):
        super().__init__(level=logging.INFO)
        self.job = job
        self.current_idx = STEP_ORDER.index(start_step)

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for pat, key in LOG_TO_STEP:
            if pat.match(msg):
                idx = STEP_ORDER.index(key)
                if idx > self.current_idx:
                    # zamknij wszystkie wcześniejsze etapy do nowego włącznie
                    for i in range(self.current_idx, idx):
                        push_step(self.job, STEP_ORDER[i], complete=True)
                    push_step(self.job, key, detail=msg)
                    self.current_idx = idx
                else:
                    push_step(self.job, key, detail=msg)
                return


# ------------------------------------------------------------------------ #
# event helpers
# ------------------------------------------------------------------------ #
def push_event(job: Job, kind: str, **payload) -> None:
    job.events.put({"type": kind, **payload})


def push_step(job: Job, key: str, *, detail: str = "",
              complete: bool = False, emit_progress: bool = True) -> None:
    """Wyślij update etapu i (opcjonalnie) postęp paska.

    emit_progress=False używamy gdy START krokowi ma towarzyszyć ticker —
    inaczej dostajemy szarpnięcie (push_step pcha 30% z 0.4-frakcji, potem
    ticker zaczyna od 0% i pasek się cofa).
    """
    label, weight = STEP_BY_KEY.get(key, (key, 0))
    push_event(job, "step", step=key, label=label, detail=detail, complete=complete)
    if emit_progress:
        push_event(job, "progress", percent=_compute_percent(job, key, complete=complete))


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
def _start_smooth_ticker(job: Job, step_key: str, *, expected_seconds: float):
    """Wątek-ticker który płynnie pcha pasek do PRZÓD przez czas trwania
    blokującego kroku (np. OCR ~60-90s).

    Krzywa: asymptotyczna (1 - exp(-t/tau)), gdzie tau dobrane tak, że po
    `expected_seconds` jesteśmy na ~80% allocacji kroku. Cap 95% — ostatnie
    5% domknie `complete=True` po realnym zakończeniu.
    """
    stop = threading.Event()

    label, weight = STEP_BY_KEY[step_key]
    # zakres % przeznaczony na ten krok
    total = sum(w for _, _, w in PIPELINE_STEPS)
    prior = sum(w for k, _, w in PIPELINE_STEPS if STEP_ORDER.index(k) < STEP_ORDER.index(step_key))
    pct_start = prior / total * 100
    pct_end = (prior + weight) / total * 100
    pct_cap = pct_start + (pct_end - pct_start) * 0.95

    tau = max(8.0, expected_seconds * 0.55)  # stała czasowa krzywej
    t0 = time.time()

    def loop():
        while not stop.is_set():
            if job.cancelled.is_set():
                return
            elapsed = time.time() - t0
            frac = 1.0 - math.exp(-elapsed / tau)
            pct = pct_start + (pct_cap - pct_start) * frac
            push_event(job, "progress", percent=int(round(pct)))
            if stop.wait(1.5):
                return

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return stop, th


class _Cancelled(Exception):
    """Sygnał wyjścia z pipeline po klepnięciu Stop przez usera."""


def _check_cancel(job: Job) -> None:
    if job.cancelled.is_set():
        raise _Cancelled


def _run_pipeline(job: Job) -> None:
    try:
        _check_cancel(job)
        push_step(job, "upload", complete=True)

        # 1. OCR — najdłuższy etap, ticker płynnie podnosi % w tle.
        # Pierwszy strzał na świeżym workerze: ~120-150s (model load).
        # Kolejne: ~30-60s. Dajemy 100s jako średnią — krzywa exp i tak nie
        # dobije do 100% (cap 95% allocacji). OCR jest jednym blokującym
        # callem do EasyOCR — nie da się go bezpiecznie przerwać z
        # zewnątrz, więc po cancel czekamy aż się skończy i WYCHODZIMY
        # przed analyze/render. (W praktyce z perspektywy usera UI już
        # zamknięte; CPU dopali się w tle ≤ minuta.)
        push_step(job, "ocr", detail="Czytam numery działek z mapy…",
                  emit_progress=False)
        ocr_dir = Path(tempfile.gettempdir()) / "mapy_ocr_web"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        ocr_cache = ocr_dir / f"{job.id}.pkl"
        t0 = time.time()
        ticker_stop, ticker_th = _start_smooth_ticker(
            job, "ocr", expected_seconds=100.0)
        try:
            run_all_maps.build_ocr_cache(job.pdf_path, ocr_cache)
        finally:
            ticker_stop.set()
            ticker_th.join(timeout=2)
        _check_cancel(job)
        ocr_secs = time.time() - t0
        push_step(job, "ocr", complete=True,
                  detail=f"Rozpoznano numery ({ocr_secs:.0f} s)")

        # 2. analyze() — handler mapuje log na "analyze" / "classify"
        log = logging.getLogger("analyze_hybrid")
        log.setLevel(logging.INFO)
        push_step(job, "analyze", detail="Wczytuję geometrię mapy…")
        handler = JobProgressHandler(job, start_step="analyze")
        log.addHandler(handler)
        try:
            result = analyze_hybrid.analyze(str(job.pdf_path), ocr_cache=str(ocr_cache))
        finally:
            log.removeHandler(handler)
        _check_cancel(job)
        # po zakończeniu analyze: classify jest ostatnim aktywowanym etapem,
        # zamykamy go (handler aktywuje go ale nie zamyka)
        push_step(job, "classify", complete=True,
                  detail=f"Znaleziono {len(result.crossed)} dz.")

        # 3. render PNG
        _check_cancel(job)
        push_step(job, "render", detail="Rysuję mapę…")
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
    except _Cancelled:
        push_event(job, "cancelled")
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


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """User klika Stop. Markujemy job jako anulowany — pipeline wyjdzie
    przy najbliższym checkpoint'cie. OCR (jeden blokujący call) dokończy się
    w tle, ale nic z tego nie wynika dla użytkownika."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    if not job.done:
        job.cancelled.set()
    return jsonify({"ok": True})


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
