---
title: Analiza Mapy
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Wgraj mapę PDF i zobacz numery działek przeciętych przez czerwoną trasę.
---

# Analiza mapy geodezyjnej

Webowy demo do wykrywania numerów działek przeciętych przez czerwoną trasę
na zwektoryzowanej mapie PDF (PZT/granice). Pipeline: PyMuPDF + EasyOCR +
Shapely.

Wybierz plik PDF — w trakcie analizy widać postęp z podziałem na kroki.
Po zakończeniu dostajesz:

- listę numerów działek do skopiowania,
- mapę z zaznaczonymi etykietami (zielone = wybrane, pomarańczowe = sporne).

Pierwszy upload trwa nieco dłużej (~30 s) — kontener pre-warmuje EasyOCR
podczas budowy obrazu, ale ładowanie modeli do pamięci procesu dzieje się
przy pierwszym requestcie.
