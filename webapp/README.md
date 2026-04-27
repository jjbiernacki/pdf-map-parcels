# Analiza mapy — webowy demo

Webowy interfejs do uploadu mapy PDF i wypisania numerów działek przeciętych
przez czerwoną trasę. Reużywa pipeline `analyze_hybrid.py` z głównego repo.

## Funkcje

- Drag-and-drop upload jednego PDF
- Strumieniowy progres (SSE) — pokazuje aktualny krok i % całości
- Lista wybranych numerów w polu tekstowym (przycisk **Kopiuj**)
- Render mapy z zaznaczonymi etykietami (wybrane = zielone kółka, sporne = pomarańczowe)
- Czytelny, neutralny design (jasny + ciemny motyw automatycznie)

## Uruchomienie lokalne

```bash
# w katalogu głównym repo:
python -m venv .venv && source .venv/bin/activate
pip install -r webapp/requirements.txt
python webapp/app.py
# otwórz http://localhost:8000
```

Pierwszy upload będzie wolniejszy — EasyOCR pobiera modele (~110 MB) do `~/.EasyOCR/`.

## Deploy

> **Vercel nie zadziała** — funkcje serverless mają limit 50 MB i 10–60 s
> timeout, a EasyOCR + torch + opencv to ~1.5 GB image, jeden request OCR
> trwa 30–90 s. Heroku nie ma już darmowego planu.
> Polecane: **Hugging Face Spaces** (najlepiej, free, ML-friendly), Render, Fly.io.

### Hugging Face Spaces (zalecane, najprościej)

1. Załóż konto na [huggingface.co](https://huggingface.co).
2. Stwórz nowy Space → SDK = **Docker** → wybierz hardware *CPU basic (free)*.
3. Sklonuj repo Space'a, skopiuj zawartość tego repo do środka, upewnij się że
   `Dockerfile` jest w roocie Space'a (możesz zrobić `cp webapp/Dockerfile ./Dockerfile`).
4. `git push` — HF sam zbuduje obraz i odpali apkę pod adresem
   `https://huggingface.co/spaces/<user>/<space>`.

### Render.com

1. Wrzuć repo na GitHuba.
2. Render → **New +** → **Blueprint** → wskaż repo (Render odczyta `webapp/render.yaml`).
3. Build trwa ~5–8 min (pre-warm EasyOCR). Free plan: 512 MB RAM, instance usypia
   po 15 min braku ruchu i ~50 s startu po wybudzeniu.

> Free plan Render może być ciasny pamięciowo (EasyOCR + torch potrafi przekroczyć
> 512 MB na większych mapach). Jeśli zobaczysz `out of memory`, zmień plan na
> **Starter** ($7/mc) albo użyj HF Spaces.

### Fly.io

```bash
fly launch --dockerfile webapp/Dockerfile --no-deploy
fly scale memory 1024     # zaleca się 1 GB pod EasyOCR
fly deploy
```

### Lokalnie przez Docker

```bash
docker build -f webapp/Dockerfile -t analiza-mapy .
docker run --rm -p 7860:7860 analiza-mapy
# otwórz http://localhost:7860
```

## Architektura

- **Backend**: Flask + SSE. Każdy upload tworzy `Job` w pamięci procesu,
  wątek roboczy puszcza pipeline i pcha eventy do kolejki SSE.
- **Postęp**: `analyze_hybrid` ma logger `analyze_hybrid` z deterministycznymi
  komunikatami INFO; własny `JobProgressHandler` mapuje je na predefiniowane
  kroki UI z wagami sumującymi się do 100%.
- **Render mapy**: PyMuPDF rasteryzuje stronę w skali 3×, PIL nakłada okręgi
  i podpisy na pozycjach OCR-owych etykiet z `crossed`.
- **OCR cache**: per-job w `/tmp/mapy_ocr_web/`, garbage-collected razem z jobem
  po 30 min.

## Limity i znane ograniczenia

- Tylko strona 1 PDFa (zgodnie z pipeline'em).
- Single-process, single-instance — `Job` jest in-memory. Pod load balancerem
  z >1 worker SSE nie znajdzie joba na innym workerze. Trzymaj `--workers 1`.
- EasyOCR jest wąskim gardłem (60–80% czasu). Nie ma lepszej opcji bez GPU.
- Limit upload: 25 MB (wystarczy na każdą mapę z `Mapy/`).
