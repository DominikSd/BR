To jest kanoniczne miejsce na prawdziwe screenshoty uzywane przez pixel-based MVP perception.

Struktura:
- `raw/`
  Surowe sample frames do analizy pojedynczej i batchowej.

Na obecnym etapie skopiowano tu istniejace screenshoty z `screenshots/enemies`,
zeby perception mialo jedno, spójne miejsce na:
- sample frames do lokalnej analizy,
- fixture'y pod dopracowywanie template packow,
- porownywanie wynikow kolejnych iteracji heurystyk.

Dodatkowo:
- pelne screeny z realnej gry sa trzymane jako `live_spot_scene_*.png`,
- obok nich leza sidecary `.json`,
- sidecary moga nadpisywac:
  - `reference_point_xy`
  - `spawn_roi`
  - inne lekkie metadane pomocne przy analizie.

Obecny folder `screenshots/enemies` moze jeszcze istniec pomocniczo,
ale docelowo analiza perception powinna operowac na `assets/live/sample_frames/raw`.
