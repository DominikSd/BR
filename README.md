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

- domyslna sekwencja wejsc to `1 -> space`
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

Lista dostepnych presetow:

```bash
python -m botlab.main --list-scenario-presets
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

Replay z pliku YAML:

```bash
python -m botlab.main --config config/default.yaml --scenario-file scenarios/custom.yaml
```

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
