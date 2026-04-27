# Deploy na Hugging Face Spaces — krok po kroku

## 1. Załóż Space

1. Wejdź na <https://huggingface.co/new-space>.
2. **Owner**: twój user / org.
3. **Space name**: np. `analiza-mapy` (URL będzie `https://huggingface.co/spaces/<user>/analiza-mapy`).
4. **License**: MIT (albo dowolnie).
5. **Space SDK**: wybierz **Docker** → **Blank**.
6. **Hardware**: *CPU basic · 2 vCPU · 16 GB RAM · free*.
7. **Public** (żeby udostępnić jako demo).
8. Kliknij **Create Space**.

HF utworzy pusty git repo i pokaże komendy do klonowania.

## 2. Wrzuć kod

```bash
# w katalogu projektu (tym, w którym jesteś teraz)
SPACE=https://huggingface.co/spaces/<TWOJ_USER>/analiza-mapy
git clone $SPACE /tmp/hf-space
cd /tmp/hf-space

# skopiuj cały kod aplikacji
rsync -av --exclude='.git' \
          --exclude='.venv' \
          --exclude='__pycache__' \
          --exclude='.pytest_cache' \
          --exclude='Mapy/' \
          --exclude='*.png' \
          --exclude='*.pkl' \
          "/Users/jerzybiernacki/Documents/Claude/Projects/Analiza mapy/" ./

# Dockerfile + README muszą być w ROOCIE Space'a (HF nie wykryje ich w huggingface/)
cp huggingface/Dockerfile ./Dockerfile
cp huggingface/README.md  ./README.md
cp huggingface/.gitattributes ./.gitattributes

# commit + push
git add .
git commit -m "Initial deploy"
git push
```

> Jeśli HF zapyta o login, użyj `huggingface-cli login` (token z
> <https://huggingface.co/settings/tokens>, scope `write`).

## 3. Patrz jak się buduje

- W Space'ie zakładka **Logs** pokazuje build (5–8 min — instalacja torcha,
  pre-warm EasyOCR).
- Po zakończeniu Space pokazuje aplikację pod
  `https://huggingface.co/spaces/<user>/<space>`.

## 4. Pierwsze uruchomienie

- Kontener startuje w ~5–10 s.
- Pierwszy upload PDFa: ~30–60 s (EasyOCR ładuje modele do pamięci).
- Kolejne uploady: 15–40 s (zależnie od rozmiaru mapy).

## 5. Co testowo wrzucić

Małe mapy z `Mapy/` (10–25 KB każda) wystarczą. Dla testu możesz załączyć
np. `Mapy/03 PZT granice.pdf` (48 KB).

## Częste problemy

- **Build fail "out of memory"** — HF free dostaje 16 GB RAM przy buildzie,
  to wystarczy. Jeśli mimo to się wywali, spróbuj `pip install` z `--no-cache-dir`
  (już jest w Dockerfile).
- **App restartuje co request** — sprawdź `Logs`, czy nie ma OOM przy ładowaniu
  EasyOCR. Free CPU ma 16 GB — powinno być z górką.
- **SSE nie strumieniuje, czeka na koniec** — HF route'uje przez Cloudflare,
  ale `text/event-stream` z nagłówkiem `X-Accel-Buffering: no` powinno działać.
  W kodzie jest ustawione.
- **Kontener nie startuje, "no such file or directory: webapp/app.py"** —
  sprawdź czy w Space'ie jest folder `webapp/` (rsync mógł go pominąć jeśli
  źle podałeś ścieżkę źródłową).

## Update'y

Każdy `git push` do Space'a wywołuje rebuild i redeploy automatycznie.
