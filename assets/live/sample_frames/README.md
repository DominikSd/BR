To jest kanoniczne miejsce na prawdziwe screenshoty uzywane przez pixel-based MVP perception.

Struktura:
- `raw/`
  Surowe sample frames do analizy pojedynczej i batchowej.
- `tuning/`
  Manifesty klatek do codziennego strojenia progow i template packow.
- `regression/`
  Manifesty klatek, ktore maja dawac porownywalny wynik quality i latency miedzy uruchomieniami.
- `holdout/`
  Manifesty scen odlozonych do sprawdzenia, czy tuning nie przeuczyl heurystyk.
- `hard_cases/`
  Manifesty trudniejszych scen z duzym szumem tla, occupied albo niejednoznacznym selected target.

Na obecnym etapie skopiowano tu istniejace screenshoty z `screenshots/enemies`,
zeby perception mialo jedno, spójne miejsce na:
- sample frames do lokalnej analizy,
- fixture'y pod dopracowywanie template packow,
- porownywanie wynikow kolejnych iteracji heurystyk.

Dodatkowo:
- pelne screeny z realnej gry sa trzymane jako `live_spot_scene_*.png`,
- obok nich leza sidecary `.json`,
- konkretne profile spotow/scen leza w `assets/live/scenes`,
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

Sidecar moze tez zawierac jawny `ground_truth`, ktory jest zrodlem prawdy dla strict benchmarku:

- `candidates`
  lista kandydatow referencyjnych
- `candidate_id`
- `screen_xy`
- `max_error_px`
- `occupied`
- `selected`
- opcjonalnie `bbox`

Benchmark splity nie duplikuja screenshotow. Zamiast tego kazdy split ma w swoim katalogu:

- `frames.json`

Manifest wskazuje na klatki w `raw/`, np.:

- `../raw/live_spot_scene_1.png`
- `../raw/live_spot_scene_2.png`

To pozwala:

- wersjonowac splity benchmarkowe bez kopiowania tych samych PNG,
- miec jeden kanoniczny zestaw raw frames,
- uruchamiac benchmarki `tuning/regression/holdout/hard_cases` tym samym pipeline'em.

`expected_candidates` sluzy do opisu konkretnych kandydatow bardziej semantycznie:
- `screen_xy`
- `max_error_px`
- `occupied`
- `selected`

Obecny folder `screenshots/enemies` moze jeszcze istniec pomocniczo,
ale docelowo analiza perception powinna operowac na `assets/live/sample_frames/raw`.

Jesli scena ma byc spot-aware:
- screenshot zostaje w `raw/`,
- profil sceny opisujacy polygon strefy trafia do `assets/live/scenes`,
- config live ustawia `scene_profile_path`,
- perception oznacza detections jako `in_scene_zone` lub `out_of_zone`.
