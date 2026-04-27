# Cel zadania

## Co trzeba zrobić

Napisać program w Pythonie, który analizuje mapę geodezyjną w formie PDF-a
(`03 PZT granice.pdf` w katalogu projektu) i wypisuje **dokładną listę
numerów działek, przez które przebiega czerwona linia** (pas drogowy).

## Wejście

- **PDF**: `/Users/jerzybiernacki/Documents/Claude/Projects/Analiza mapy/03 PZT granice.pdf`
  - Jest w 100% wektorowy (operatory `m/l/c/S/B`)
  - Trzy kluczowe kolory:
    - czerwony `(1, 0, 0)` = trasa (pas drogowy, linia kreskowana, `width=1.44`)
    - zielony `(0, 0.584, 0)` = granice działek
    - czarny = elementy dodatkowe (piny pomiarowe itp.)
  - Czerwona trasa składa się z ~1073 krótkich odcinków-kreseczek (długość ~2.8 pt każda)
  - Zielone granice: 314 ścieżek (311 linii granicznych + 114 glifów-etykiet)
  - Tekst etykiet działek jest **narysowany jako wektorowe ścieżki glifów** —
    `page.get_text()` nic nie zwraca; etykiety trzeba rozpoznać OCR-em

- **Ground truth**: `numery działek.txt` — 19 numerów działek:
  ```
  391, 336, 334, 337, 296/3, 296/2, 48/1, 49/1, 310,
  254, 277, 278, 279, 280, 281, 282, 283, 284, 51/1
  ```

## Wyjście

Program ma wypisać **dokładnie te 19 numerów** — wszystkie z pliku, żadnej innej.

Format CLI (przykład):
```
DZIAŁKI PRZECIĘTE (19):
  391, 336, 334, 337, 296/3, 296/2, 48/1, 49/1, 310,
  254, 277, 278, 279, 280, 281, 282, 283, 284, 51/1
DZIAŁKI DO SPRAWDZENIA (0):
  —
```

## Wymagania

1. **100% poprawności** — żadnych false positive, żadnych false negative
2. **Przypadki sporne** (linia czerwona mija granicę w tolerancji ~1 px) mają
   być oznaczone do ręcznej weryfikacji, a nie zgadywane
3. **Testy** — w szczególności test end-to-end porównujący wynik z
   `numery działek.txt`
4. **Iteracja do skutku** — cytat z oryginalnej instrukcji:
   > „Pracuj aż nie znajdziesz algorytmu który zwraca dokładnie wypisane
   > działki - wszystkie działki z pliku i żadnej innej."

## Środowisko

- Katalog projektu: `/Users/jerzybiernacki/Documents/Claude/Projects/Analiza mapy/`
- `venv` już istnieje w `.venv/` (Python 3.9)
- Zainstalowane: `pymupdf`, `numpy`, `scipy`, `Pillow`, `shapely>=2.0`,
  `easyocr`, `pytest`
- macOS, mps dostępne dla easyocr (gpu=False domyślnie dla powtarzalności)

## Kontrakt wyjścia

```python
@dataclass
class Result:
    crossed: list[str]      # numery działek przeciętych (posortowane wg kolejności na trasie)
    borderline: list[str]   # do weryfikacji ręcznej (w granicach ~1 px)
```

CLI:
```
python analyze.py "03 PZT granice.pdf" [--debug-png out.png] [-v]
```

## Kryterium akceptacji (docelowe)

- `set(result.crossed) == set_of_19_from_file`
- `result.borderline == []` (lub zawiera TYLKO niejednoznaczne przypadki,
  nie zawsze 7 z GT jak obecnie)
- Wszystkie testy zielone
