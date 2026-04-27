# Hugging Face Space — Docker SDK, port 7860 wymagany przez platformę.
# Build context = root repo (huggingface README + Dockerfile leżą w roocie Space'a,
# kod aplikacji w webapp/ + pliki analyze_*.py w roocie repo).

FROM python:3.11-slim

# system-deps dla pymupdf, opencv, torch (pulled przez easyocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces oczekuje że proces nasłuchuje jako user 'user' (uid 1000), nie root.
# Tworzymy go i ustawiamy jego HOME — easyocr pobiera modele do ~/.EasyOCR/.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /home/user/app

COPY --chown=user webapp/requirements.txt /home/user/app/requirements.txt

USER user
RUN pip install --user --no-cache-dir -r /home/user/app/requirements.txt

# Pre-warm EasyOCR — pobiera ~110 MB modeli do /home/user/.EasyOCR/.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# Kopia całego repo (analyze_hybrid.py, analyze_cv.py, analyze_ray.py,
# analyze.py, run_all_maps.py i webapp/).
COPY --chown=user . /home/user/app

EXPOSE 7860

CMD ["gunicorn", "--chdir", "webapp", "app:app", \
     "--workers", "1", "--threads", "4", \
     "--timeout", "600", \
     "--bind", "0.0.0.0:7860"]
