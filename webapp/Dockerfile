# Dockerfile do deploy na Hugging Face Spaces, Fly.io, Render-Docker, Railway.
# Uwaga: EasyOCR przy pierwszym uruchomieniu pobiera model (~64 MB craft +
# ~47 MB en) — dlatego pre-warmujemy go w build-stage żeby nie dziacgać
# na pierwszego usera.

FROM python:3.11-slim

# system-deps niezbędne dla pymupdf, opencv, easyocr/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webapp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-warm EasyOCR (download modeli) — robione w build, żeby kontener
# startował szybko, bez timeoutów na pierwszym requestcie.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# Reszta kodu repozytorium
COPY . /app

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["gunicorn", "--chdir", "webapp", "app:app", \
     "--workers", "1", "--threads", "4", \
     "--timeout", "600", \
     "--bind", "0.0.0.0:7860"]
