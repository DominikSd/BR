Ten katalog przechowuje male, kontrolowane fixture'y dla perception pipeline.

Zasada na obecnym etapie:
- fixture moze byc zwyklym plikiem JSON opisujacym klatke i jej metadane,
- nie wymagamy jeszcze prawdziwych screenshotow ani OCR,
- najwazniejsze jest stabilne porownywanie:
  - detekcji kandydatow,
  - filtrowania occupied/free,
  - wyboru najblizszej wolnej grupki,
  - reaction latency i agregatow sesji.

Jesli pozniej dojdzie prawdziwy template matching po obrazach:
- obok pliku obrazu moze lezec sidecar `.json`,
- `PerceptionFrameLoader` juz potrafi taki uklad obsluzyc.
