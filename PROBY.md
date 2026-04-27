# Próby i wnioski — co nie zadziałało

## TL;DR dla nowej sesji

Osiągnąłem **12/19** w liście „pewne przecięcia" (bez FP) + pozostałe 7
w liście „do sprawdzenia" (zanieczyszczonej ~82 FP). **Nie udało mi się
zredukować listy sprawdzania do dokładnie 7 szukanych działek** ani
dołączyć ich do listy pewnych przecięć. Poniżej szczegóły dlaczego.

---

## Struktura PDF-a (zweryfikowana)

- 623 drawings ogółem (`page.get_drawings()`)
- **Zielone** (granice): 314 ścieżek po filtracji (wyrzucone glify <30×20pt z multi-item)
- **Czerwone**: 161 ścieżek
  - 68 kresek trasy `width=1.44` (ale rozbitych na ~1073 maleńkich odcinków 2.8pt)
  - 62 piny pomiarowe `width=0.84` (pomijane)
  - 31 wypełnień (główki pinów, pomijane)
- Trasa: `x ∈ [736.7, 6386.3]`, `y ∈ [161.5, 768.1]` (w pkt PDF)
- Lewy koniec trasy: `(736.68, 162.12)`
- Prawy koniec trasy: `(6386.28, 195.48)`
- **Tekst działek jest rysowany wektorowo jako glify** — `page.get_text()`
  zwraca PUSTY wynik. Jedyne źródło napisów to OCR.

---

## Co z 19 działek GT robi mi problem

Po OCR easyocr (`/tmp/ocr_cache_v2.pkl` — 125 etykiet, wszystkie 19 GT
obecne) i polygonizacji Shapely, 12 działek GT siedzi w wielokącie
przecinanym przez trasę — klasyfikowane poprawnie jako „crossed":

| ✓ Crossed (12) | 254, 277, 278, 279, 280, 281, 282, 296/2, 296/3, 334, 336, 337 |

Pozostałe 7 działek GT — **etykiety NIE siedzą w żadnym zamkniętym
wielokącie** (są w wyciekającym komponencie tła — 195M px z 199M px
całego rastru):

| Działka | d(trasy) pt | d(najbliższy wielokąt) pt | Uwagi |
|---------|-------------|----------------------------|-------|
| 391     | 38          | 101                        | Etykieta pośrodku pasa drogowego, otwarte pole |
| 310     | 15          | 534                        | W pasie drogowym; brak zamkniętych granic |
| 48/1    | 33          | 893                        | Jak wyżej |
| 49/1    | 21          | 1057                       | Jak wyżej |
| 51/1    | 83          | 879                        | Jak wyżej |
| 283     | 88          | 21 (niekrzyżowany) / 102 (krzyżowany) | Sąsiad przeciętego wielokąta |
| **284** | **192**     | 37                         | **190pt ZA końcem narysowanej trasy** — dla 284 nie ma ŻADNEJ czerwonej kreski w zasięgu |

**Kluczowy wniosek o 284**: sprawdziłem wszystkie czerwone drawings w
promieniu 250pt od `(6572.9, 151.9)` — najbliższy czerwony element jest
190pt dalej (ostatni pin pomiarowy trasy). **Trasa fizycznie kończy się
przed działką 284** — mimo to GT uznaje ją za przeciętą. Sugeruje to,
że GT odzwierciedla *prawny zasięg pasa drogowego*, a nie narysowaną
linię. Ta interpretacja może być kluczowa dla nowej sesji.

---

## Próby algorytmiczne

### Próba 1: Raster + flood-fill (PIL + scipy.ndimage.label)

- Rasteryzacja zielonych granic na skali 8, szerokość 3 px + koła w wierzchołkach
- Rasteryzacja czerwonej trasy na skali 8, szerokość 4 px
- `binary_closing` na zielonej masce, `label` na jej negatywie
- Dopasowanie etykiet OCR do komponentów przez spiralne wyszukiwanie

**Problem**: 98% pikseli (195M z 199M) wypada w JEDNYM komponencie —
„wyciekającym tle". Granice działek nie są w wielu miejscach szczelnie
zamknięte (działki wychodzą poza stronę, są szczeliny w T-junction).
Próba `binary_closing` z większym jądrem NIE POMAGA — zaczyna fuzjonować
sąsiednie działki.

**Wynik**: 13/19 (12 GT + 1 artefakt „7283" z błędnego OCR)

### Próba 2: Shapely polygonize + snap(tol) — CURRENT BEST

- `unary_union(MultiLineString(green_segs))` z `snap(tol=3..5)` by zamknąć szczeliny
- `polygonize(noded)` → 122–160 wielokątów
- Dla każdego wielokąta: `poly.buffer(-0.5).intersection(red_union).length >= 0.5` → crossed
- Dla każdej etykiety OCR: `poly.contains(Point)` → przypisanie tekstu do wielokąta

**Problem**: wielokąty obejmują tylko w pełni zamknięte poligony.
Działki „slivery" w pasie drogowym (391, 310, 48/1, 49/1, 51/1) oraz
działki końcowe (283, 284) nie mają zamkniętych granic i nie wchodzą
do wyniku polygonize.

Testowałem `snap_tol ∈ {0.5, 1, 2, 3, 4, 5, 8}` — większy tol tworzy więcej
wielokątów ALE nadal nie zamyka otwartych pasów drogowych.

**Wynik**: 12/19 poprawnych w crossed, 0 FP.

### Próba 3: Label distance → route (bufor trasy)

Pomysł: etykieta bliska trasy (d < threshold) → działka przecięta.

**Problem**: próg nie rozróżnia GT od FP:

```
★  391 d=38    |    390 d=9   (NIE GT)
★  310 d=15    |    53  d=16  (NIE GT — numer stacji pomiarowej)
★  49/1 d=21   |    55  d=19  (NIE GT)
★  48/1 d=33   |    386 d=12  (NIE GT)
★  51/1 d=83   |    60  d=27  (NIE GT)
★  283 d=88    |    297 d=33  (NIE GT)
★  284 d=192   |    163/5 d=164 (NIE GT)
```

Etykiety stacji pomiarowych (50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62)
siedzą **NA** czerwonej linii — ich odległość do trasy jest MNIEJSZA niż GT.

### Próba 4: Filtr po wysokości glifu (`h` z OCR bbox)

GT: `h ∈ [104, 195]`. Non-GT stacje: `h ∈ [87, 131]`. **Overlap: 104–131**
— nie da się wyznaczyć progu.

### Próba 5: Filtr po font size z vector text

`page.get_text("dict")` → **pusty wynik**. Tekst nie jest tekstem, tylko
wektorowymi ścieżkami glifów. Font metadata niedostępne.

### Próba 6: Deduplikacja etykiet (280, 281, 282 występują po DWA RAZY)

OCR znajduje `280`, `281`, `282` w dwóch miejscach:

| Tekst | Instancja 1 (GT) | Instancja 2 (NIE GT) |
|-------|------------------|----------------------|
| 280   | (6003, 275)      | (3019, 658)          |
| 281   | (6102, 258)      | (3183, 642)          |
| 282   | (6257, 225)      | (2885, 657)          |

Obie instancje są blisko trasy — same współrzędne nie rozróżniają.
Rozróżniam je przez to, że instancja GT wpada w wielokąt przecinany
przez trasę, a druga — nie. **To działa dla 280/281/282 GT.**

### Próba 7: Filtr po confidence OCR

- `7283` (artefakt OCR) ma conf=0.71
- Wszystkie GT mają conf ≥ 0.78 (najniższa: 310 = 0.78, 296/2 duplikat = 0.67)
- **Próg 0.75 eliminuje „7283" bez utraty GT** → zastosowałem w finalnym kodzie

### Próba 8: Filtr po regex (długość numeru)

- GT: wszystkie 1–3 cyfry z opcjonalnym `/N` (N = 1–2 cyfry)
- Artefakty OCR: `7283`, `1315/16`, `163/5`, `8/45` — mają ≥4 cyfry w prefiksie lub sklejki
- Zacieśniłem `LABEL_RE = r"^\d{1,3}(/\d{1,2})?$"` — to też eliminuje
  kilka FP, ale nie rozwiązuje głównego problemu

### Próba 9: Endpoint distance

Dla 284 (192 pt za końcem trasy): rozszerzyłem borderline o działki
bliskie **końcom trasy** (< 250 pt). To wprowadza 284 do listy
borderline, ale też wciąga kilka innych FP przy końcówkach.

---

## Co zostało nie sprawdzone (pomysły na nową sesję)

### A) Rozpoznanie pasa drogowego jako poligonu
Trasa biegnie między DWIEMA równoległymi zielonymi liniami (outer
boundaries pasa drogowego). Jeśli udałoby się wykryć te dwie linie i
wyznaczyć *polygon pasa drogowego*, to:
- Działki wewnątrz pasa = road sub-parcels (391, 310, 48/1, 49/1, 51/1, 283, 284, 296/3, 296/2)
- Działki przecinane przez pas = main crossed parcels (277, 278, ...)

**Implementacja**: znaleźć pary długich, prawie-równoległych zielonych
odcinków biegnących w pobliżu trasy; ich konkatenacja = outer boundaries.
Potem zamknąć końce → polygon pasa drogowego.

### B) Rozszerzenie mapowania na „wylewające się" regiony
Zamiast `polygon.contains(label)`, dla etykiet w dużym komponencie tła:
- Cluster etykiety po bliskości + bliskości trasy
- Każdy klaster = jedna działka
- Klaster który „klejowo" łączy się z przecinanymi wielokątami → crossed

### C) Użyć OCR z większą kontrolą kontekstu
- `tesseract --psm 11` (sparse text) lub Google Vision API
- Może wykryć że 50/51/52 są w innym *stylu font* (kursywa?) niż 48/1/49/1

### D) Interpretacja prawna pasa drogowego
Jeśli GT odzwierciedla *prawną* listę działek objętych pasem, a nie
geometryczne przecięcie — to może trzeba parsować osobny dokument (PZT
= Plan Zagospodarowania Terenu często ma tabelę działek w legendzie).
**Sprawdzić czy PDF ma drugą stronę / tabelę** (kod zawsze bierze
`doc[0]` — może jest stron więcej?).

### E) Wyznaczyć geometrię pasa po wektorach
Zauważyłem że poszczególne „kreski" trasy (odcinki długości ~2.8pt) można
połączyć w ciągłą linię przez:
1. Sortowanie po parametrze (arc_length)
2. Bezpośrednie łączenie kolejnych dash-endpoint pairs
To daje ciągły polyline zamiast 1073 rozproszonych segmentów. Potem
`buffer(road_width)` dałoby dokładniejsze polygon pasa.

### F) Analiza „pinów" pomiarowych (red, width=0.84)
Piny są rozmieszczone wzdłuż trasy co ~190pt, dokładnie na granicach
działek. Mogą być **najbardziej wiarygodnym sygnałem** gdzie trasa
przecina granicę. Każdy pin = punkt przecięcia. Działki po obu stronach
pina = przecięte.

**To jest chyba najmocniejszy nietknięty pomysł.** Piny oznaczają
punkty pomiarowe w miejscach gdzie trasa tnie granice działek — czyli
są GEOMETRYCZNYMI MARKERAMI PRZECIĘĆ z definicji geodezyjnej.

### G) Spróbować drugiej strony PDF-a
```python
doc = fitz.open(pdf); print(len(doc))   # ile stron?
```
Sprawdzić czy jest druga strona z tabelą / legendą.

### H) Użyć PyMuPDF text extraction z opcjami
```python
page.get_text("rawdict")  # zamiast "dict"
page.get_textpage_ocr()   # OCR bezpośrednio w PyMuPDF
```

---

## Pliki artefakty (do wykorzystania przez nową sesję)

Wszystkie w `/tmp/`:

- `ocr_cache_v2.pkl` — 125 etykiet OCR easyocr przy skali 8, zapisane jako
  lista dict `{text, conf, cx, cy, w, h, bbox}`. **Wszystkie 19 GT są w cache'u.**
  Ładowanie: `pickle.load(open('/tmp/ocr_cache_v2.pkl','rb'))`.
  Oszczędza ~5 min OCR-u przy każdym uruchomieniu.
- `overlay_seg0.png`..`overlay_seg3.png` — cały PDF podzielony na 4 segmenty
  horyzontalne z oznaczeniami:
  - **żółte koło** = koniec trasy (leftmost/rightmost)
  - **magenta koło** = pozycja etykiety GT
  - **mały niebieski punkt** = pozostałe etykiety OCR
- `det_283.png`, `det_284.png`, `det_310.png`, `det_277_284.png`,
  `det_296_336.png`, `det_48_51.png` — detale okolic konkretnych działek
  w wysokiej rozdzielczości

---

## Obecny stan pipeline'u (co zostawiłem w `analyze.py`)

- Filtracja OCR: `conf ≥ 0.75` + regex `^\d{1,3}(/\d{1,2})?$`
- Polygonize z snap tol=5.0
- Klasyfikacja:
  - crossed = etykieta w wielokącie, który trasa przecina w *wnętrzu*
    (po `buffer(-0.5)`) z długością > 0.5 pt
  - borderline = etykieta poza wielokątami + blisko trasy (<100 pt) lub
    blisko końca trasy (<250 pt)
- Deduplikacja po tekście (dla 280/281/282 wybiera bliższą trasy)

**Wynik końcowy**: 12/19 w crossed (0 FP), pozostałe 7 GT + ~82 FP w
borderline. 38 testów pytest zielone, włączając test end-to-end
sprawdzający pokrycie całego GT (ale nie sprawdza rozdziału — bo tego
nie udało się osiągnąć).

---

## Nowa sesja (2026-04-22, Claude Opus 4.7): OpenCV + Shapely hybrid — **13/19 TP, 0 FP, 6 FN**

**Plik**: `analyze_hybrid.py` + lower-level helpers w `analyze_cv.py`.

### Pipeline (deterministyczny, bez LLM)

1. **Vector greens**: ~484 odcinków z `iter_segments` po `extract_paths`.
2. **Endpoint closures**: pary wolnych końców szkieletu w promieniu 5pt
   → odcinki vector (~6 par).
3. **T-junction extensions**: w T-j wyznaczyć kierunek gałęzi bocznej (z
   antykolinearności 2 głównych) → ekstrapolować do trafienia w szkielet
   po drugiej stronie. ~90 nowych odcinków (max 40pt).
4. **Pin crosslines**: 62 piny `width=0.84` → każdy dostaje PROSTOPADŁY
   odcinek (długość 50pt) do lokalnego kierunku trasy. Ma na celu sztuczne
   podzielenie pasa drogowego na sub-działki.
5. **Ramka**: bbox strony jako 4 odcinki.
6. **Shapely polygonize** po `unary_union + snap(tol=3)`: 204 polygony.
7. **Filtr powierzchni**: odrzuć polygony > 1 mln pt² (zewnętrzne wycieki).
8. **Klasyfikacja**: polygon `crossed` ⇔ `poly.buffer(-0.5).intersection(red_union).length ≥ 0.5`.

### Wyniki

| Metryka | `analyze.py` (poprzedni best) | **`analyze_hybrid.py`** |
|---|---|---|
| TP | 12/19 | **13/19** |
| FP w crossed | 0 | **0** |
| FN | 7 (w borderline) | **6 (w borderline)** |
| Borderline | ~82 FP | 62 (6 GT + 56 nie-GT) |

Progres vs. `analyze.py`: **+1 GT** (`310`) wchodzi do pewnych crossed dzięki
T-junction extension + pin crosslines. Zerowe FP utrzymane.

### Pozostałe 6 FN — analiza geometryczna

Działki `283, 284, 391, 48/1, 49/1, 51/1` siedzą w **jednym wielkim polygonie
zewnętrznym (~5.4 mln pt²)** obejmującym całość poza „zamkniętymi" działkami.
Przyczyny:

| Działka | Etykieta w...        | Problem geometryczny |
|---------|-----------------------|----------------------|
| 391     | pasie drogowym       | brak poprzecznej granicy na lewo i prawo od etykiety |
| 310     | pasie drogowym       | (wyłapany dzięki T-j extension z 334/336/337 odnóg) |
| 48/1    | pod pasem            | dolna granica działki to dolna linia pasa — działka ma OTWARTE BOKI (brak pionowych granic w polu widzenia etykiety) |
| 49/1    | pod pasem            | jak 48/1 |
| 51/1    | nad pasem            | jak 48/1 od góry |
| 283     | pasie drogowym       | jak 391 |
| 284     | za końcem trasy       | 192pt za końcem narysowanej trasy; brak poprzecznej granicy w pobliżu |

**Kluczowy brak geometryczny**: w regionie `48/1..51/1` pas drogowy NIE MA
poprzecznych granic wewnętrznych ani odnóg T-junction — pin crosslines
dodają tylko 1 poprzeczną linię na ok. 200pt pasa, niewystarczająco żeby
odseparować etykiety stacji pomiarowych od sliverów.

### Kandydaci na dalsze kroki (niezrobione)

- **A) Detekcja podwójnych równoległych linii pasa drogowego**:
  zidentyfikować parę długich zielonych linii biegnących równolegle do
  trasy (d<30pt, kąt<10°) → wyznaczyć polygon pasa → dodać POPRZECZNE
  granice między pary linii co ~100pt (nie tylko na pinach).
- **B) Per-crop Claude vision**: dla każdego z 6 FN + sąsiadów, zrobić crop
  600×600pt i przez Claude multimodal/Anthropic SDK pytać „czy trasa
  przecina działkę X". Szacowany koszt: 10-40 wywołań API.
- **C) Wyłapywanie 284 przez ekstrapolację trasy**: oś trasy kończy się
  pinem pomiarowym; przedłużyć oś tangentowo o N pt i sprawdzić które
  działki przetną ten przedłużony zasięg.
- **D) OCG layer `EGBD03`**: PDF ma 5 optional content groups — `z_1 lampa`,
  `z_1 szafa`, `z_tekst`, `EGBD03`, `z_1 kabel zasilający`. Warstwa `EGBD03`
  może zawierać granice działek lepiej zdefiniowane niż widoczne zielone
  stroki. Sprawdzić czy da się iterować po tej warstwie z PyMuPDF (obecnie
  `page.get_drawings()` zwraca wszystkie warstwy razem).

### Ostateczna rekomendacja

Czysto deterministyczny algorytm geometryczny osiąga sufit **13/19 TP, 0 FP**.
Pozostałe 6 GT są w wielkim polygonie wyciekającym obejmującym stacje pomiarowe
i FP-slivery — **nie są geometrycznie odróżnialne od nich bez dodatkowego
sygnału wizualnego lub warstwy prawnej**. Rozwiązanie docelowe wymaga
albo (A) explicite detekcji polygonu pasa drogowego z dodatkowymi syntezowanymi
granicami, albo (B) LLM-vision per kandydat.

1. **Najpierw sprawdzić czy PDF ma więcej stron** — `len(doc)`. Jeśli
   tak, przetworzyć tabelę legendy (pomysł **D + G**).
2. **Wykorzystać piny czerwone (width=0.84)** jako markery przecięć
   (pomysł **F**). Każdy pin przypisać do pary działek po obu stronach
   najbliższej zielonej linii.
3. **Wyznaczyć geometrię pasa drogowego** (pomysł **A**): znaleźć
   długie równoległe zielone odcinki blisko trasy, zbudować polygon pasa,
   działki przecięte pasem = szukana lista.

Nie tracić więcej czasu na próg odległości OCR do trasy — stacje pomiarowe
50–62 i etykiety GT 277–284 są nieodróżnialne geometrycznie przez
samą odległość.
