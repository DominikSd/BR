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
- sidecary moga tez zawierac `expected_perception`,
  czyli lekki kontrakt regresyjny dla realnych klatek referencyjnych:
  - ile targetow powinno byc widocznych,
  - ile powinno byc occupied/free,
  - czy target powinien zostac wybrany,
  - przyblizona pozycje wybranego targetu,
  - przyblizone pozycje occupied targetow.

Przykladowe pola `expected_perception`:
- `min_target_count`, `max_target_count`
- `min_free_target_count`, `max_free_target_count`
- `min_occupied_target_count`, `max_occupied_target_count`
- `expected_candidates`
- `occupied_target_screen_xy`
- `occupied_target_max_error_px`
- `selected_target_required`
- `selected_target_must_be_free`
- `selected_target_screen_xy`
- `selected_target_max_error_px`

`expected_candidates` sluzy do opisu konkretnych kandydatow bardziej semantycznie:
- `screen_xy`
- `max_error_px`
- `occupied`
- `selected`

Obecny folder `screenshots/enemies` moze jeszcze istniec pomocniczo,
ale docelowo analiza perception powinna operowac na `assets/live/sample_frames/raw`.
