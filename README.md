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
- minimalny engage MVP.

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

Batch wypisuje:

- wynik per frame,
- summary latency,
- accuracy summary dla klatek z `expected_perception`,
- `real_scene_regression` dla klatek `live_spot_scene_*`.

### Live Preview

```bash
python -m botlab.main --config config/live_dry_run.yaml --live-preview
```

Preview pokazuje:

- aktualna klatke,
- spawn ROI,
- kandydatow,
- occupied/free,
- selected target,
- podstawowe latency vision.

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

Na tym etapie realna sciezka kliku jest przygotowana tylko dla PPM na Windows i nadal powinna byc traktowana jako ostrozny MVP.

Artefakty engage trafiaja do:

- `data/live_debug/engage/engage_results.jsonl`
- `data/live_debug/engage/engage_session_summary.json`
- `data/live_debug/engage/engage_XXX/engage_result.json`
- `data/live_debug/engage/engage_XXX/engage_overlay.svg`

### Ograniczenia warstwy live na tym etapie

- to nadal heurystyczne live vision MVP, nie finalna wizja produkcyjna,
- brak OCR i brak ciezkiego ML/CV,
- full combat/reward/rest loop nie jest jeszcze celem warstwy live,
- real input execution poza PPM nie jest jeszcze domkniete,
- najlepszy feedback do strojenia daje obecnie batch na realnych scenach referencyjnych.

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
