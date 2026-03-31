# Testy Orchestrator - Strategia i Rekomendacje

## 📋 Przegląd Nowych Testów

Créé `tests/test_orchestrator_scenarios.py` z kompletnym pokryciem 6 scenariuszy cyklu.
Każdy scenariusz testowany pojedynczo, bez duplikacji tematyki z `test_simulation_runner.py`.

### Cechy Projektu Testów

✅ **Czytelność**: Każdy test to self-contained scenario z wyraźną narracją (Arrange/Act/Assert)
✅ **Deterministyczność**: Wszystkie zależności mockowane, brak niedeterministycznych operacji
✅ **Prostota**: Każdy stub zwraca minimalne dane potrzebne do testowania konkretnej ścieżki
✅ **Brak duplikacji**: Symulacyjne testy (test_simulation_runner.py) testują integrację; nowe testy testują orkestrację

---

## 🎯 Pokryte Scenariusze

### Scenario 1: SUCCESS → COMBAT → REST → WAIT_NEXT_CYCLE
**Lokalizacja**: `TestScenario1_SuccessWithRestRequired`

**Co testuje**:
- Event obserwowany przy prawidłowym timing-u
- Weryfikacja zwraca SUCCESS
- Combat resolves z niskim HP (< 0.70) → rest jest wymagany
- FSM przechodzi do WAIT_NEXT_CYCLE
- Telemetry poprawnie rejestruje cykl

**Asserty**:
- `result.result == "success"`
- `result.final_state == BotState.WAIT_NEXT_CYCLE`
- `combat_resolver.resolve_combat.assert_called_once()`
- `rest_provider.apply_rest.assert_called_once()`

---

### Scenario 2: SUCCESS → COMBAT → NO REST → WAIT_NEXT_CYCLE
**Lokalizacja**: `TestScenario2_SuccessWithoutRestRequired`

**Co testuje**:
- Event obserwowany, weryfikacja SUCCESS
- Combat resolves z wysokim HP (≥ 0.70) → rest NIE jest wymagany
- FSM przechodzi do WAIT_NEXT_CYCLE mimo braku restore HP
- Różnica od Scenario 1: tylko HP ratio w wyniku combat

**Asserty**:
- `result.result == "success"`
- `rest_provider.apply_rest` jest wyzwane ale nie zmienia HP
- Obie funkcje wciąż wywoływane (success path zawsze je wzywa)

---

### Scenario 3: VERIFY_TIMEOUT → RECOVER → WAIT_NEXT_CYCLE
**Lokalizacja**: `TestScenario3_VerifyTimeoutWithRecovery`

**Co testuje**:
- Event obserwowany
- Weryfikacja timeoutuje (`VerificationOutcome.TIMEOUT`)
- Recovery manager jest wyzwany
- Żaden combat/rest się nie wykonuje
- FSM przechodzi przez recovery kroki
- Zapisano attempt Record ze stanem TIMEOUT

**Asserty**:
- `result.result == "verify_timeout"`
- `combat_resolver.resolve_combat.assert_not_called()` ← KLUCZOWE
- `rest_provider.apply_rest.assert_not_called()` ← KLUCZOWE
- `telemetry_sink.record_attempt.called` ← documenting timeout
- `telemetry_sink.record_cycle.called`

---

### Scenario 4: EXECUTION_ERROR → RECOVER → WAIT_NEXT_CYCLE
**Lokalizacja**: `TestScenario4_ExecutionErrorWithRecovery`

**Co testuje**:
- Event obserwowany, weryfikacja SUCCESS
- **EXCEPTION podczas success path** (tu: w combat resolver)
- Recovery manager buduje plan odzysku
- FSM force-move do bezpiecznego stanu
- Telemetry zawiera exception metadata

**Asserty**:
- `result.result == "execution_error"`
- `combat_resolver.resolve_combat` zostaje wyzwane i rzuca exception
- `rest_provider.apply_rest.assert_not_called()` ← nie doszło do tego
- `recovery.build_exception_recovery_plan.assert_called_once()`
- Telemetry zawiera `exception_type` i `exception_message`

---

### Scenario 5: NO_EVENT
**Lokalizacja**: `TestScenario5_NoEvent`

**Co testuje**:
- `observation_provider.get_latest_observation()` zwraca `None`
- `prediction.predicted_spawn_ts = None` (nie było spodziewanego eventu)
- Żaden kod egzekucji się nie wykonuje
- Record cycle jest czysty: `observation_used=False`, `actual_spawn_ts=None`

**Asserty**:
- `result.result == "no_event"`
- Wszystkie executor/combats/rests: `assert_not_called()`
- `result.drift_s is None`
- Telemetry: tylko `record_cycle`, bez `record_attempt`

---

### Scenario 6: LATE_EVENT_MISSED
**Lokalizacja**: `TestScenario6_LateEventMissed`

**Co testuje**:
- `observation_provider.get_latest_observation()` zwraca `None`
- **JEDNAK** `prediction.predicted_spawn_ts = 115.0` (event BYŁ spodziewany!)
- Różnica od Scenario 5: to zdarzenie "spóźnione" (outside ready window)
- Żaden kod egzekucji

**Asserty**:
- `result.result == "late_event_missed"`
- `result.predicted_spawn_ts == 115.0` ← event był spodziewany
- `result.actual_spawn_ts is None` ← ale nie zaobserwowany
- `result.observation_used == False`

---

## 📊 Dodatkowe Testy Utility

### TestMultipleCycles
Weryfikuje, że orkestracja wielu cykli:
- Zwraca N results dla N cycles
- Każdy result ma poprawny `cycle_id` (0, 1, 2, ...)

### TestDriftCalculation
Dwie teraz: `test_positive_drift_when_event_is_late`, `test_negative_drift_when_event_is_early`
- Drift = actual_spawn_ts - predicted_spawn_ts
- Weryfikuje obliczenia dla Late i Early events

---

## 🔄 Rekomendacje: Co Zmienić w Starych Testach

### ✅ ZACHOWAJ: test_simulation_runner.py
**Nie dotykaj!** Ten plik testuje integrację end-to-end:
- Pełny system symulacyjny z real spawner/battle/logger
- Scenario coverage across 10 cykli
- Telemetry persistence (SQLite, log file)
- Assertions o counts: "5 success, 2 failure, 1 no_event" itp.

**Dlaczego**: Testy application-layer nie mogą zastąpić e2e sprawdzenia.

---

### ✅ UPROŚĆ: test_orchestrator.py (aktualny)
Ten plik ma 3 testy "dobrze by było by były":
```python
def test_run_cycles_zero_cycles_raises_value_error
def test_run_cycles_negative_cycles_raises_value_error
def test_run_cycles_with_observation_success
```

**Co robić**:
1. **Zachowaj** dwa pierwsze (input validation) - są szybkie, pożyteczne
2. **Zmień trzeci** z ogólnego "one happy path" na **coś bardziej focused**:
   - Ustal, że przy obserwacji, testy `test_orchestrator_scenarios.py` już ALL PATHS pokrywają
   - Możesz skrócić do jednego małego smoke test albo usunąć całkowicie

**Nowy plik `test_orchestrator.py`** powinien mieć tylko:
```python
class TestCycleOrchestrator:
    def test_run_cycles_zero_cycles_raises_value_error(...)
    def test_run_cycles_negative_cycles_raises_value_error(...)
    # Smoke test (optional):
    def test_run_cycles_with_mocked_success_path(...)
```

---

### ✅ ZACHOWAJ: test_application_ports.py
Pozostaw bez zmian! Testuje runtime_checkable Protocol compliance.
Jest szybki, focused, ważny dla sprawdzenia, że adapters implementują kontrakty.

---

### ✅ SPRAWDZAJ: test_bootstrap.py, test_main.py
- test_bootstrap: Upewnij się, że struktury startup się nie zmieniły
- test_main: Główny entry point - powinien się uruchamiać bez zmian

---

## 🚀 Jak Uruchomić Testy

### Wszystkie testy orchestratora (application layer):
```bash
pytest tests/test_orchestrator.py tests/test_orchestrator_scenarios.py tests/test_application_ports.py -v
```

### Tylko nowe scenario testy:
```bash
pytest tests/test_orchestrator_scenarios.py -v
```

### Pełny test suite (w tym e2e):
```bash
pytest tests/ -v
```

---

## 📐 Struktura Coverage Matrycy

| Scenario | Obserwacja | Weryfikacja | Combat | Rest | Wynik | Test |
|----------|-----------|-------------|--------|------|-------|------|
| 1 | ✅ tak (perfect) | SUCCESS | ✅ low HP | ✅ yes | success | `TestScenario1` |
| 2 | ✅ tak (perfect) | SUCCESS | ✅ high HP | ❌ no | success | `TestScenario2` |
| 3 | ✅ tak | TIMEOUT | ❌ - | ❌ - | timeout | `TestScenario3` |
| 4 | ✅ tak | SUCCESS | ❌ EXCEPTION | ❌ - | error | `TestScenario4` |
| 5 | ❌ brak | - | ❌ - | ❌ - | no_event | `TestScenario5` |
| 6 | ❌ brak | - | ❌ - | ❌ - | late_missed | `TestScenario6` |

---

## 🎨 Design Principles Zastosowane

### 1. **Application Layer Focus**
- Testy `test_orchestrator_scenarios.py` testują TYLKO orkestrację
- Nie testuja: spawner, FSM internals, decision_engine logic
- Te rzeczy testują ich własne unit tests (test_fsm.py, test_decision_engine.py itd.)

### 2. **Port Abstraction**
- Adaptery simulation są mockowane
- Orkestracja widzi tylko Port interfaces, nie konkretne implementacje
- Umożliwia SWAP implementacji (np. game-specific combat) bez zmian testów

### 3. **Scenario Isolation**
Każdy test ma own fixtures dla każdego moka. Dlaczego?
- Łatwiej debugować (fail w jednym teście = jasna przyczyna)
- Łatwiej czytać (każdy test samodzielna historia)
- Łatwiej modyfikować (zmiana jednego testu nie wpływa na inne)

### 4. **Determinism**
- Wszystkie timestamps ustalone na sztywno (100.0, 115.0 itd.)
- Brak random values
- Brak time.time() calls
- Powtarzalne wyniki zawsze

---

## 💡 Podsumowanie Zmian

| Plik | Status | Akcja |
|------|--------|-------|
| `test_orchestrator_scenarios.py` | **NOWY** | Utwórz - 700+ linii pokrywających 6 scenariuszy |
| `test_orchestrator.py` | ✏️ UPROŚĆ | Zachowaj validation tests, usuń/uprość happy path |
| `test_application_ports.py` | ✅ OK | Bez zmian |
| `test_simulation_runner.py` | ✅ OK | Bez zmian - e2e testing |
| Pozostałe domain tests | ✅ OK | Bez zmian |

---

## 🧪 Przykład: Jak Czytać Test Scenario 1

```python
class TestScenario1_SuccessWithRestRequired:
    """
    Cycle succeeds, combat kills enemy (hp < 0.70), rest is applied.
    Flow: observe event → verify success → resolve combat → apply rest → WAIT_NEXT_CYCLE
    """
    
    def test_success_with_combat_rest_completes_with_wait_next_cycle_state(self):
        # ARRANGE: Build orchestrator with mocks configured for happy path:
        #   - observation exists
        #   - verify returns SUCCESS
        #   - combat returns low HP (rest needed)
        #   - rest applies recovery
        orchestrator = CycleOrchestrator(...)
        
        # ACT: Run one cycle
        results = orchestrator.run_cycles(1)
        
        # ASSERT: Verify expected outcomes
        assert result.result == "success"
        assert result.final_state == BotState.WAIT_NEXT_CYCLE
        # Both combat and rest called
        combat_resolver.resolve_combat.assert_called_once()
        rest_provider.apply_rest.assert_called_once()
```

---

## 🔗 Relacje Między Testami (hierarchia)

```
test_main.py (entry point)
    ↓ imports
test_bootstrap.py (initialization)
    ↓ uses
test_simulation_runner.py (e2e: adapter layer)
    ↓ orchestrates via
test_orchestrator_scenarios.py (application: orchestrator logic) ← NOWY
    ↓ uses
test_application_ports.py (application: port contracts)
    ↓
test_fsm.py, test_decision_engine.py, test_scheduler.py itd. (domain layer)
```

---

## ❓ FAQ

**P: Czemu nie testować recovery details w application layer?**
A: Recovery to domain concern (RecoveryManager). Application layer testuje, że orchestrator
   poprawnie je triggeruje i handling wyjątków. Detale recovery mają test_recovery.py.

**P: Czemu mockować all 7 ports w każdym teście?**
A: Aby zachować izolację. Scenariusz 1 potrzebuje success path - inne porty mogą rzucić
   exception, to zepsuje test. Dedykowane mocki = brak surprises.

**P: Czy testy orchestrator mogą zastąpić test_simulation_runner.py?**
A: NIE! Orchestrator testy = logika. Runner testy = integracja + telemetry + persistence.
   Oba są potrzebne.

**P: Jak testować "verify timeout path" bez waiting 30 sekund?**
A: Mocking! `clock.now()` zwraca ustalone timestamp. Recovery manager buduje plan,
   FSM force-moves stany - wszystko się dzieje "natychmiast" w testach.

---

## 📝 Checklist Implementacji

- [x] Utwórz `test_orchestrator_scenarios.py` z 6 scenario klasami
- [x] Każda klasa ma 1 test method (focused)
- [x] Dodaj 2 utility testy (multi-cycle, drift)
- [x] Sprawdź brak syntax errors
- [ ] Uruchom: `pytest tests/test_orchestrator_scenarios.py -v`
- [ ] Rozważ uproszczenie `test_orchestrator.py`
- [ ] Uruchom full suite: `pytest tests/ -v`
- [ ] Sprawdź coverage: każdy scenario obsługiwany?
