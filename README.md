# botlab

Minimalny, modularny symulator lokalnego rdzenia cyklicznego systemu PvE w Pythonie.

## 1. Czym projekt jest teraz

Projekt realizuje samodzielny przebieg symulacyjny z następującymi warstwami:
- konfiguracja z YAML (`config/default.yaml`),
- scheduler/predictor dla cyklicznych zdarzeń (interval + drift + okna czasowe),
- FSM decydujący o fazach cyklu (IDLE -> PREPARE -> READY -> ATTEMPT -> VERIFY -> COMBAT/REST/RECOVER),
- prosty generator zdarzeń i świata symulacji (spawny, timingi, symulowane walki),
- telemetryka do logu i SQLite (wpisy cyklu, prób, przejść stanów),
- warstwa odzysku przy timeoutach i wyjątkach.

## 2. Moduły projektu

- `src/botlab/config.py` - wczytanie/validacja konfiguracji aplikacji, cyklu, combat, telemetry, vision.
- `src/botlab/types.py` - klasy domenowe (`BotState`, `CyclePrediction`, `Observation`, `Decision`, `CombatSnapshot`, `TelemetryRecord`).
- `src/botlab/domain/predictor.py` - `SpawnPredictor`, odkłada anchor, obserwacje i estymacja interwału.
- `src/botlab/domain/scheduler.py` - `CycleScheduler`, stany czasowe, predykcje okien przygotowania i gotowości.
- `src/botlab/domain/decision_engine.py` - `DecisionEngine`, algorytm podejmowania decyzji dla FSM.
- `src/botlab/domain/fsm.py` - `CycleFSM`, przetwarzanie decyzji i historia przejść.
- `src/botlab/domain/recovery.py` - `RecoveryManager`, wykrywanie stuck stanu i plan recovery.
- `src/botlab/adapters/simulation/runner.py` - `SimulationRunner`, orchestracja cykli, telemetria i logika przebiegu.
- `src/botlab/adapters/simulation/spawner.py` - `SimulatedSpawner`, scenariusze cykli, generacja obserwacji i wyników verify.
- `src/botlab/adapters/simulation/world.py` - `SimulatedWorld`, timeline cyklu (prepare/ready/attempt/verify/rest) dla świata.
- `src/botlab/adapters/simulation/battle.py` - symulacja walki i odpocznienia.
- `src/botlab/adapters/telemetry/logger.py` - logger JSON + plik + opcjonalnie konsola.
- `src/botlab/adapters/telemetry/storage.py` - `SQLiteTelemetryStorage`, schema init i przeszukiwanie.
- `src/botlab/adapters/telemetry/schema.py` - definicje tabel SQLite (`cycles`, `state_transitions`, `attempts`).

## 3. Jak działa przebieg symulacyjny

1. `main.py` uruchamia `SimulationRunner.from_settings()`.
2. `CycleScheduler` bootstrapuje predykcję na `anchor_spawn_ts` i `anchor_cycle_id`.
3. Dla każdego cyklu `run_cycles()`:
   - wylicza predykcję z `SpawnPredictor`,
   - generuje `SpawnEvent` w `SimulatedSpawner`,
   - tworzy `CycleTrace` w `SimulatedWorld` (przygotowanie/ready/attempt/verify...),
   - kolejno `FSM.tick()` w kluczowych momentach (prepare, ready, attempt, verify itd.),
   - zapisuje rekordy stanu/atempt/cykl do SQLite i do logów,
   - przy błędach lub timeoutach weryfikacji: wykonuje `RecoveryManager` i dodatkowe przejścia.
4. Po cyklu raportuje `SimulationReport` z liczbą sukcesów, porażek, timeoutów itp.

## 4. Komponenty rdzenia

### config
`botlab.config.Settings` i podsekcje to jednoźródłowa konfiguracja: `app`, `cycle`, `combat`, `telemetry`, `vision`.

### types
`BotState`, `CyclePrediction`, `Observation`, `Decision`, `CombatSnapshot`, `TelemetryRecord`.

### predictor
`SpawnPredictor` wykorzystuje historyczne spawny do uśrednienia interwału i przewidywania czasu następnego spawn.

### scheduler
`CycleScheduler` wykorzystuje predykcje do określenia stanu czasowego: `WAIT_NEXT_CYCLE`, `PREPARE_WINDOW`, `READY_WINDOW`.

### decision engine
`DecisionEngine` logika wyboru następnego stanu w FSM zależnie od:
- obecny stan,
- temporal_state (okna),
- obserwacja sygnału,
- wynik weryfikacji,
- snapshot walki (hp),
- timeouty.

### FSM
`CycleFSM` przyjmuje `Decision`, przechowuje aktualny stan, wysyła historię przejść i stan końcowy.

### recovery
`RecoveryManager` wykrywa stuck w `PREPARE_WINDOW`, `READY_WINDOW`, `ATTEMPT`, `VERIFY`, `COMBAT`, `REST`, `RECOVER` i buduje plan wyjścia do `RECOVER` + `WAIT_NEXT_CYCLE`.

### simulation runner
`SimulationRunner` integruje scheduler, FSM, spawner, world, battle, rest, recovery, logging i storage w jednym przebiegu cykli.

### telemetry logger
`configure_telemetry_logger()` (plik + opcjonalnie konsola) i `log_telemetry_record()` (JSON manifest) w `botlab.adapters.telemetry.logger`.

### SQLite storage
`SQLiteTelemetryStorage` inicjalizuje schemat i zapisuje w tabelach:
- `cycles`,
- `state_transitions`,
- `attempts`.

## 5. Uruchomienie z CLI

```bash
python -m botlab.main --cycles 20 --config config/default.yaml --anchor-spawn-ts 100.0 --anchor-cycle-id 0 --console-log
```

albo

```bash
python -m botlab.main
```

Domyślnie używa `config/default.yaml`:
- `data/telemetry/botlab.sqlite3`
- `logs/botlab.log`

## 6. Uruchomienie testów

```bash
pytest -q
```

lub

```bash
python -m pytest -q
```

Uwaga: testsy sprawdzają import pakietu i istniejące ścieżki oraz poprawność konfiguracji.

## 7. Aktualny następny etap rozwoju

Priorytety kolejnego kroku:
- dodać wieloźródłowe przypadki testowe dla `SpawnPredictor` i `CycleScheduler` w różnych scenariuszach driftem,
- pełna implementacja modelu walki w `SimulatedBattle` (efektywny decline HP, prawdziwe stany zagrożenia),
- lepsza adaptacja predictora do niestacjonarnego interwału oraz outlierów,
- debugowalny interfejs eksportu wyników symulacji z SQLite w CSV/JSON,
- integracja plug-inów obserwacji (np. vision/kamera/zbior danych rzeczywistego sygnału),
- dodatnie symulacji wielowątkowej/async dla równoległych agentów.
