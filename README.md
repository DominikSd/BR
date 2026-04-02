# botlab

Stabilny, testowalny rdzen logiczny dla cyklicznego bota PvE, rozwijany najpierw jako architektura i przeplyw use-case, a dopiero potem jako adaptery srodowiskowe.

## Aktualna architektura

Repo uzywa jednej kanonicznej sciezki:

- `src/botlab/domain`
  logika domenowa: predictor, scheduler, decision engine, FSM, recovery, model swiata i targetowanie
- `src/botlab/application`
  DTO, porty i use-case flow: orchestrator cyklu, target acquisition, approach, interaction, engagement
- `src/botlab/adapters/simulation`
  symulacyjny runtime swiata, scenariusze, replay runner, combat/rest i composition root
- `src/botlab/adapters/telemetry`
  logger JSON i zapis do SQLite

W repo nie ma juz rownoleglych namespace'ow `botlab.core`, `botlab.simulation` ani `botlab.telemetry`.

## Przebieg cyklu

Aktualny flow symulacji:

1. przygotowanie obserwacji strefy spawnu
2. wykrycie zdarzenia w oknie obserwacji
3. acquire / keep target
4. approach do targetu
5. final revalidate przed interakcja
6. verify
7. combat
8. rest, jesli potrzeba
9. powrot do `WAIT_NEXT_CYCLE`

Target moze zostac utracony podczas ruchu albo tuz przed interakcja. W takim przypadku system natychmiast robi retarget albo konczy cykl jako `no_target_available`.

## Symulacja walki

Aktualna walka jest celowo prosta i deterministyczna:

- domyslny loop walki to:
  runda 1: `1 -> space`
  kolejne rundy: `3 -> space`
- nazwane plany walki sa ladowane z `config/combat_plans.yaml`
- profile walki mapujace sie na named plans sa ladowane z `config/combat_profiles.yaml`
- standardowy run moze miec domyslny profil z `combat.default_profile_name` w `config/default.yaml`
- sekwencja jest jawna w scenariuszu jako `combat_inputs`
- mozna wybrac profil walki przez `combat_profile_name` albo CLI `--combat-profile`
- mozna tez wybrac nazwany plan walki przez `combat_plan_name` albo CLI `--combat-plan`
- mozna tez opisac plan per runda przez `combat_plan_rounds`, np. `[['1', 'space'], ['2']]`
- to jest fundament pod przyszla konfiguracje "co klikac", w jakiej kolejnosci i w ktorej rundzie

## Najwazniejsze moduly

- `src/botlab/config.py`
  wczytywanie i walidacja konfiguracji YAML
- `src/botlab/types.py`
  wspolne typy domenowe i `TelemetryRecord`
- `src/botlab/main.py`
  CLI
- `src/botlab/application/orchestrator.py`
  kanoniczny use-case cyklu
- `src/botlab/application/targeting.py`
  acquire / approach / interaction / engagement services
- `src/botlab/adapters/simulation/runner.py`
  composition root dla symulacji
- `src/botlab/adapters/simulation/replay.py`
  powtarzalne replaye, presety scenariuszy i loader YAML dla scenario runnera

## Uruchomienie

Standardowy przebieg:

```bash
python -m botlab.main --config config/default.yaml --cycles 10 --anchor-spawn-ts 100.0 --anchor-cycle-id 0
```

albo:

```bash
python run.py
```

Replay z wbudowanego presetu:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset baseline_mixed_cycle
```

Kanoniczne demo farming loop:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset demo_farming_cycle
```

Ten preset wypisuje czytelny trace decyzji cyklu:
- ktory target zostal wybrany,
- ktore grupki odrzucono i dlaczego,
- kiedy nastapil retarget,
- kiedy zaczela sie walka,
- czy byl rest,
- jaki byl koncowy wynik cyklu.

Kanoniczna pelna sesja farmienia:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset demo_farming_session
```

Ten preset pokazuje w jednym przebiegu:
- kilka grupek w strefie respawnu,
- odrzucenie grupki zajetej przez kogos innego,
- przejecie targetu podczas approach,
- natychmiastowy retarget na najblizsza inna wolna grupke,
- combat,
- reward,
- rest do progow gotowosci,
- kolejny cykl z nowym respawnem grupek.

Bardziej pokazowy wariant z czytelnymi timestampami faz:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset demo_farming_showcase
```

Showcase wypisuje dodatkowo:
- przejscie do pozycji obserwacyjnej (`phase=staging`),
- oczekiwanie na spawn w punkcie obserwacji (`phase=wait`),
- dalej standardowy flow targetowania, retargetu, combat i rest.

Demo utraty cyklu przed targetowaniem:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset demo_observation_miss
```

Demo recovery po takim spoznieniu w kolejnym cyklu:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset demo_observation_reposition
```

Lista dostepnych presetow:

```bash
python -m botlab.main --list-scenario-presets
```

Wymuszenie trace decyzji dla dowolnego runa:

```bash
python -m botlab.main --config config/default.yaml --scenario-preset retarget_path --show-cycle-trace
```

Lista dostepnych planow walki:

```bash
python -m botlab.main --list-combat-plans
```

Lista dostepnych profili walki:

```bash
python -m botlab.main --list-combat-profiles
```

Standardowy run z nazwanym planem walki:

```bash
python -m botlab.main --config config/default.yaml --cycles 5 --combat-plan spam_1_space
```

Standardowy run z profilem walki:

```bash
python -m botlab.main --config config/default.yaml --cycles 5 --combat-profile fast_farmer
```

Na koncu przebiegu CLI wypisuje tez agregaty telemetry per `combat_plan_name` i `combat_profile_name`, z liczba sukcesow, restow i srednim koncowym HP.

Opcjonalny eksport raportu do JSON:

```bash
python -m botlab.main --config config/default.yaml --cycles 5 --export-report-json data/reports/latest.json
```

Replay z pliku YAML:

```bash
python -m botlab.main --config config/default.yaml --scenario-file scenarios/custom.yaml
```

## Live Vision i Engage MVP

Repo ma tez cienka warstwe `src/botlab/adapters/live`, ktora wykorzystuje ten sam rdzen logiki, ale podpina do niego:

- capture foreground albo dry-run,
- pixel-based perception i batch analysis,
- preview/debug window dla live vision,
- minimalny engage MVP,
- pixel-based verify dla `combat_indicator`.

### Co juz dziala

- analiza pojedynczej klatki i batcha screenshotow,
- overlay SVG, JSON i JSONL dla perception,
- metryki latency perception:
  - `detection latency`
  - `selection latency`
  - `total reaction latency`
- real-scene regression summary dla `live_spot_scene_*`,
- marker-first live vision:
  - czerwony marker
  - occupied swords
  - local mob confirmation
- minimalny engage MVP:
  - wybierz najblizszy wolny target
  - wykonaj probe engage
  - sklasyfikuj wynik jako:
    - `engaged`
    - `target_stolen`
    - `misclick`
    - `approach_stalled`
    - `approach_timeout`
    - `no_target_available`

### Batch tuning / analysis

Pojedyncza klatka:

```bash
python -m botlab.main --config config/live_dry_run.yaml --analyze-frame assets/live/sample_frames/raw/live_spot_scene_1.png --perception-output-dir data/perception_single
```

Batch katalogu screenshotow:

```bash
python -m botlab.main --config config/live_dry_run.yaml --analyze-batch-dir assets/live/sample_frames/raw --perception-output-dir data/perception_batch
```

Batch analysis dziala rekurencyjnie, wiec mozesz tez analizowac wieksza strukture katalogow, np.:

```bash
python -m botlab.main --config config/live_dry_run.yaml --analyze-batch-dir assets/live/sample_frames --perception-output-dir data/perception_batch
```

Batch wypisuje:

- wynik per frame,
- summary latency,
- accuracy summary dla klatek z `expected_perception`,
- `real_scene_regression` dla klatek `live_spot_scene_*`.

### Spot-aware scene profile

Scene-aware perception dla konkretnego spota jest konfigurowane przez:

- `live.scene_profile_path`

Aktualny profil referencyjny:

- `assets/live/scenes/single_spot_scene.json`

Profil sceny moze zdefiniowac:

- `scene_name`
- `reference_frame_path`
- `spawn_zone_polygon`
- opcjonalne `reference_point_xy`
- opcjonalne `sub_rois`
- opcjonalne `exclusion_polygons`

W MVP detections sa oznaczane jako `in_scene_zone`, a target selection bierze tylko wolne targety wewnatrz `spawn_zone_polygon`.

Kalibracja sceny do live capture dziala tak:

- `reference_frame_path` wskazuje screenshot referencyjny, z ktorego pochodza polygon i punkt odniesienia,
- loader sceny odczytuje rozmiar tej klatki,
- przy analizie live frame polygon, `reference_point_xy` i `sub_rois` sa skalowane do aktualnego rozmiaru klatki,
- opcjonalne `live.scene_calibration_offset_xy` pozwala dodac reczny offset, jesli capture wymaga drobnej korekty.

To pozwala utrzymac jeden profil sceny dla benchmarku, preview i realnego capture, bez wprowadzania osobnej warstwy geometrii.

Scene-aware benchmark / perception:

```bash
python -m botlab.main --config config/live_real_mvp.yaml --benchmark-split regression --strict-pixel-benchmark --perception-output-dir data/perception_scene_regression
```

albo analiza pojedynczej klatki dla spota:

```bash
python -m botlab.main --config config/live_real_mvp.yaml --analyze-frame assets/live/sample_frames/raw/live_spot_scene_1.png --perception-output-dir data/perception_scene_single
```

### Strict pixel-based benchmark

Benchmark dataset jest podzielony na splity:

- `tuning`
- `regression`
- `holdout`
- `hard_cases`

Kazdy split ma manifest `frames.json`, ktory wskazuje na screenshoty z `assets/live/sample_frames/raw`.

Strict benchmark:

```bash
python -m botlab.main --config config/live_dry_run.yaml --benchmark-split regression --strict-pixel-benchmark --perception-output-dir data/perception_regression
```

Ten tryb:

- wymaga prawdziwego obrazu rasterowego,
- blokuje metadata-only fallback,
- liczy benchmark quality i latency,
- zapisuje per-frame JSON/overlay oraz zbiorczy `perception_session_summary.json`.

W summary pojawia sie dodatkowo:

- `perception_benchmark_summary`
- `target_recall`
- `target_precision`
- `occupied_classification_accuracy`
- `selected_target_accuracy`
- `selected_target_in_zone_accuracy`
- `out_of_zone_rejection_count`
- `false_positive_reduction_after_zone_filtering`
- `false_positive`
- `false_negative`

To jest preferowany tryb do porownywania kolejnych zmian w heurystykach vision.

### Strojenie wykrywania mobow

Najwazniejsze parametry detection do dokrecania siedza teraz w jednym miejscu, w `live` configu:

- `perception_confidence_threshold`
- `confirmation_confidence_threshold`
- `occupied_confidence_threshold`
- `template_match_stride_px`
- `template_rotations_deg`
- `merge_distance_px`
- `candidate_confirmation_frames`
- `occupied_confirmation_frames`
- `candidate_loss_frames`

Po kazdej analizie w `perception_session_summary.json` dostajesz:

- `tuning_parameters`
- `worst_frames`
- per-frame `diagnostics`

W `diagnostics` dla kazdej klatki zobaczysz m.in.:

- `raw_hit_summary`
- `low_confidence_hits`
- `duplicate_merges`
- `occupied_rejections`
- `out_of_zone_rejections`
- `unstable_rejections`
- `final_candidates`
- `selection_reason`

To jest teraz glowny material do recznego strojenia template packow i progow.

### Raport najgorszych klatek

Po batch runie summary zapisuje tez `worst_frames`.

To jest skrocona lista najgorszych przypadkow, z priorytetem na:

- bledny `selected target`
- false negatives
- false positives
- bledne `occupied/free`
- wysoka latency

CLI wypisuje je tez jako:

```bash
perception_worst_frame=...
```

To jest preferowany punkt startowy do recznego przegladania overlayow.

### Porownywanie dwoch runow detection

Mozesz porownac dwa zapisane `perception_session_summary.json`:

```bash
python -m botlab.main --compare-perception-summaries data/run_a/perception_session_summary.json data/run_b/perception_session_summary.json
```

CLI wypisze roznice dla:

- `target_recall`
- `target_precision`
- `occupied_classification_accuracy`
- `selected_target_accuracy`
- false positives / false negatives
- srednich latency

To ma byc prosty, praktyczny diff do porownywania wersji heurystyk, nie duzy system eksperymentow.

### Live Preview

```bash
python -m botlab.main --config config/live_dry_run.yaml --live-preview
```

Preview pokazuje:

- aktualna klatke,
- spawn ROI,
- scene polygon / strefe spota, jesli aktywny jest `scene_profile_path`,
- status okna gry i window guard,
- kandydatow,
- occupied/free,
- detections out-of-zone,
- selected target,
- podstawowe latency vision.

### Live capture skupiony na oknie gry

Live capture jest teraz pilnowany przez prosty window guard:

- szuka okna po `live.window_title`,
- liczy region capture wzgledem okna gry,
- respektuje `live.foreground_only`,
- blokuje realny input, jesli foreground nie zgadza sie z oknem gry.

Praktyczna uwaga:

- `live.capture_region: [0, 0, 0, 0]` oznacza "bierz cale okno gry",
- to jest preferowany tryb, jesli nie chcesz recznie trzymac stalego cropa dla jednej rozdzielczosci.

W artefaktach debugowych zobaczysz:

- jakie okno zostalo dopasowane,
- jaki jest bbox okna,
- jaki jest finalny `capture_bbox`,
- czy foreground zgadza sie z gra,
- czy input zostal zablokowany przez `window_guard`.

### Pixel-based state/resource detection

Live path korzysta teraz z prostych, pikselowych detektorow UI:

- `hp_bar_roi`
  odczyt wypelnienia czerwonego paska HP
- `condition_bar_roi`
  odczyt wypelnienia zielonego paska kondycji
- `combat_indicator_roi`
  binarny combat indicator
- `reward_roi`
  prosta widocznosc rewardu

Fallback do `frame.metadata` nadal istnieje dla dry-run i starszych fixture'ow, ale gdy klatka ma prawdziwy obraz rasterowy, wykrywanie stanu i zasobow idzie po pikselach.

### Stabilny odczyt zasobow podczas restu

Podczas odpoczynku live path nie opiera sie juz na pojedynczym odczycie paska. Zamiast tego:

- pobiera kilka kolejnych probek z `hp_bar_roi` i `condition_bar_roi`,
- agreguje je medianą,
- liczy prosty `resource_confidence`,
- zapisuje warningi, jesli odczyt jest niestabilny albo podejrzany.

Najwazniejsze pola konfiguracyjne:

- `live.rest_resource_sample_count`
- `live.rest_resource_sample_interval_s`
- `live.rest_resource_min_confidence`
- `live.rest_resource_max_ticks`
- `live.rest_resource_growth_min_delta`
- `live.rest_resource_warning_spread_threshold`
- `live.rest_resource_stall_warning_ticks`

W praktyce rest tick zapisuje teraz:

- kolejne probki zasobow,
- zagregowany `hp_ratio` i `condition_ratio`,
- `resource_confidence`,
- `resource_warnings`,
- decyzje:
  - `rest_continue`
  - `rest_stop_threshold_reached`
  - `rest_stalled_or_uncertain`

Jesli overlay albo warningi sugeruja, ze odczyt jest niestabilny, najpierw warto:

1. sprawdzic czy `hp_bar_roi` i `condition_bar_roi` nadal pokrywaja prawdziwy HUD,
2. zapisac nowe screenshoty referencyjne,
3. dopiero potem stroic progi i liczbe probek.

### Live Engage Observe

```bash
python -m botlab.main --config config/live_dry_run.yaml --live-engage-observe
```

Tryb `live engage observe` pokazuje w osobnym oknie debugowym pelny pion:

- capture,
- perception,
- selected target,
- punkt kliku,
- wynik verify po probie engage.

To jest tryb pomostowy miedzy samym preview vision a `live-engage-mvp`:

- korzysta z tego samego live stacku,
- nie buduje osobnego pipeline'u,
- pozwala na zywo zobaczyc `detect -> select -> engage -> verify`.

Wazne:

- `live-engage-observe` wymusza `dry_run` input dla bezpieczenstwa,
- moze korzystac z realnego capture,
- ale nie powinien wysylac realnych klikniec do gry,
- do rzeczywistej proby PPM sluzy `live-engage-mvp`.

### Engage MVP

Dry-run engage MVP:

```bash
python -m botlab.main --config config/live_dry_run.yaml --live-engage-mvp --cycles 3
```

Kontrolowane profile dry-run do testowania klasyfikacji wyniku engage:

- `single_spot_mvp`
  bazowy przypadek `engaged`
- `engage_target_stolen`
  target po kliku staje sie zajety przez kogos innego
- `engage_target_stolen_noisy`
  jak wyzej, ale verify zawiera tez dodatkowe kandydaty i lekki szum perception
- `engage_misclick`
  klik nie prowadzi ani do engage, ani do zajecia targetu
- `engage_misclick_partial`
  misclick przy czesciowo niejednoznacznym verify i dodatkowych kandydatach w tle
- `engage_approach_stalled`
  probe engage zatrzymuje sie na etapie approach i konczy jako stall
- `engage_timeout`
  probe engage konczymy jako timeout

Aby uruchomic inny profil, ustaw `live.dry_run_profile` w konfiguracji live.

Tryb realny:

```bash
python -m botlab.main --config path/to/live.yaml --live-engage-mvp --cycles 3
```

Minimalny gotowy config pod realny preview/engage:

```bash
python -m botlab.main --config config/live_real_mvp.yaml --live-preview
```

albo:

```bash
python -m botlab.main --config config/live_real_mvp.yaml --live-engage-observe
```

albo:

```bash
python -m botlab.main --config config/live_real_mvp.yaml --live-engage-mvp --cycles 3
```

Na tym etapie realna sciezka kliku jest przygotowana tylko dla PPM na Windows i nadal powinna byc traktowana jako ostrozny MVP.

### Real key input path

Realny input jest rozdzielony na dwa bezpieczniki konfiguracyjne:

- `live.enable_real_input`
- `live.enable_real_clicks`
- `live.enable_real_keys`

Domyslnie wszystkie sa ustawione na `false`, nawet w `config/live_real_mvp.yaml`.

To daje 3 praktyczne tryby:

- preview / observe bez realnych wejsc,
- realny capture z logowaniem inputu, ale bez wysylania klawiszy,
- ostrozny realny MVP po jawnej zmianie configu.

W praktyce:

- `enable_real_input=false` blokuje wszystkie realne wejscia niezaleznie od pozostalych flag,
- `enable_real_clicks=true` pozwala na realny PPM,
- `enable_real_keys=true` pozwala na realne `press_key` i `press_sequence`.

W logach i artefaktach input dostajesz tez status wykonania, np.:

- `dry_run`
- `real_clicks_disabled`
- `real_keys_disabled`
- `real_click_sent`
- `real_key_sent`

### Quality gate przed engage

Dla `live-engage-mvp` mozna teraz dodatkowo ograniczyc zbyt agresywne klikanie przez:

- `live.engage_min_target_confidence`
- `live.engage_min_seen_frames`

Target moze zostac odrzucony jeszcze przed PPM, jesli:

- ma zbyt niska pewnosc,
- nie byl wystarczajaco stabilny przez kolejne klatki,
- jest poza strefa sceny,
- albo staje sie nieosiagalny.

W takim przypadku wynik engage trafia do `no_target_available`, ale z czytelnym `reason`, np. `engage_quality_gate_not_stable`.

To jest szczegolnie przydatne na jednej maszynie i jednym spocie, gdzie chcemy najpierw ograniczyc klikniecia w chwilowe albo podejrzane detekcje.

Artefakty engage trafiaja do:

- `data/live_debug/engage/engage_results.jsonl`
- `data/live_debug/engage/engage_session_summary.json`
- `data/live_debug/engage/engage_XXX/engage_result.json`
- `data/live_debug/engage/engage_XXX/engage_overlay.svg`

### Ograniczenia warstwy live na tym etapie

- to nadal heurystyczne live vision MVP, nie finalna wizja produkcyjna,
- brak OCR i brak ciezkiego ML/CV,
- full combat/reward/rest loop nie jest jeszcze celem warstwy live,
- real input execution jest nadal ostroznym MVP i ogranicza sie do PPM + prostych klawiszy na Windows,
- pixel-based verify dotyczy na razie glownie `combat_indicator` i prostego `reward_roi`,
- najlepszy feedback do strojenia daje obecnie batch na realnych scenach referencyjnych.

### Organizacja scen live

W `assets/live/sample_frames` mozna juz rozdzielac material na:

- `raw`
- `tuning`
- `regression`
- `holdout`

Batch loader czyta te katalogi rekurencyjnie, wiec nie trzeba splaszczac wszystkiego do jednego poziomu.

Dodatkowo profile konkretnych spotow leza w:

- `assets/live/scenes`

Minimalny workflow dla nowego spota:

1. dodaj referencyjny screenshot do `assets/live/sample_frames/raw`
2. dodaj sidecar `.json` z `reference_point_xy`, `spawn_roi` i opcjonalnym `ground_truth`
3. dodaj profil sceny w `assets/live/scenes/*.json`
4. ustaw `live.scene_profile_path` w configu live
5. uruchom `--analyze-frame` albo `--benchmark-split regression --strict-pixel-benchmark`

Workflow regresji dla tego etapu:

1. zmieniasz template pack, scene profile albo progi
2. odpalasz strict benchmark:

```bash
python -m botlab.main --config config/live_real_mvp.yaml --benchmark-split regression --strict-pixel-benchmark --perception-output-dir data/perception_scene_regression
```

3. porownujesz:
   - `target_recall`
   - `target_precision`
   - `occupied_classification_accuracy`
   - `selected_target_accuracy`
   - `selected_target_in_zone_accuracy`
   - `out_of_zone_rejection_count`
   - detection / selection / total reaction latency
4. zostawiasz tylko te zmiany, ktore poprawiaja jakosc bez niekontrolowanego wzrostu opoznien

Domyslna konfiguracja zapisuje telemetry do:

- `data/telemetry/botlab.sqlite3`
- `logs/botlab.log`

## Testy

Najwazniejsze grupy testow:

- `tests/test_orchestrator.py`
- `tests/test_orchestrator_scenarios.py`
- `tests/test_simulation_runner.py`
- `tests/test_scenario_replay.py`
- `tests/test_telemetry.py`
- `tests/test_architecture_regression.py`

Standardowe uruchomienie:

```bash
pytest -q
```
