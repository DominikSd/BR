# Test Orchestrator Scenarios - Wizualna Struktura

## 📊 Mapowanie Scenariuszy na Ścieżki Kodu

```
┌─────────────────────────────────────────────────────────────┐
│ CycleOrchestrator._run_single_cycle()                        │
└──┬──────────────────────────────────────────────────────────┘
   │
   ├─→ prediction = scheduler.prediction_for_cycle()
   ├─→ tick (PREPARE window)
   ├─→ tick (READY window)
   ├─→ observation = observation_provider.get_latest_observation()
   │
   ├─ IF observation is None:
   │  │
   │  └─→ _complete_cycle_without_observation()
   │      │
   │      ├─ IF predicted_spawn_ts is None
   │      │  │
   │      │  └─→ result = "no_event"
   │      │      └─ TEST: TestScenario5_NoEvent ✓
   │      │
   │      └─ ELSE (event was predicted)
   │         │
   │         └─→ result = "late_event_missed"
   │             └─ TEST: TestScenario6_LateEventMissed ✓
   │
   └─ ELSE: (observation exists)
      │
      └─→ _complete_cycle_with_observation()
          │
          ├─→ action_executor.execute_action()
          ├─→ tick (ATTEMPT)
          ├─→ verify_result = verification_provider.verify()
          │
          ├─ IF verify_result == TIMEOUT:
          │  │
          │  └─→ recovery.build_exception_recovery_plan()
          │      └─ result = "verify_timeout"
          │          └─ TEST: TestScenario3_VerifyTimeoutWithRecovery ✓
          │
          ├─ ELIF verify_result == FAILURE:
          │  │
          │  └─→ result = "failure"
          │      └─ TEST: Not in this suite (can add if needed)
          │
          └─ ELIF verify_result == SUCCESS:
             │
             └─→ TRY:
                │
                ├─→ _run_success_path():
                │   │
                │   ├─→ combat_outcome = combat_resolver.resolve_combat()
                │   │   │
                │   │   ├─ IF combat_outcome.hp_ratio < 0.70:
                │   │   │  │
                │   │   │  └─→ rest_outcome = rest_provider.apply_rest()
                │   │   │      (rest REQUIRED)
                │   │   │      └─ TEST: TestScenario1_SuccessWithRestRequired ✓
                │   │   │
                │   │   └─ ELSE:
                │   │      │
                │   │      └─→ rest_outcome = rest_provider.apply_rest()
                │   │          (rest NOT required, no-op)
                │   │          └─ TEST: TestScenario2_SuccessWithoutRestRequired ✓
                │   │
                │   └─→ final_state = WAIT_NEXT_CYCLE
                │
                └─ EXCEPT Exception:
                   │
                   ├─→ recovery.build_exception_recovery_plan()
                   ├─→ fsm.force_state() for each recovery_step
                   └─→ result = "execution_error"
                       └─ TEST: TestScenario4_ExecutionErrorWithRecovery ✓
```

---

## 🧪 Test Coverage Matrix

| Scenario | Observ. | Verify | Combat | Rest | Recovery | Result | Line | Test |
|----------|---------|--------|--------|------|----------|--------|------|------|
| 1 ✓ | YES | SUCCESS | low HP | YES | - | success | L174 | TestScenario1 |
| 2 ✓ | YES | SUCCESS | high HP | NO | - | success | L244 | TestScenario2 |
| 3 ✓ | YES | TIMEOUT | - | - | YES | timeout | L314 | TestScenario3 |
| 4 ✓ | YES | SUCCESS | ❌ EXC | - | YES | error | L405 | TestScenario4 |
| 5 ✓ | NO | - | - | - | - | no_event | L499 | TestScenario5 |
| 6 ✓ | NO | - | - | - | - | late_missed | L571 | TestScenario6 |

---

## 🏗️ Struktura Fabryki Mocków

```
test_orchestrator_scenarios.py
│
├─ Fixtures Globalne
│  ├─ cycle_config() → CycleConfig
│  └─ observation() → Observation
│
├─ Fabryki Helper
│  ├─ _make_prediction() → Mock Prediction
│  ├─ _make_fsm_with_transitions() → Mock FSM
│  ├─ _make_scheduler() → Mock Scheduler
│  ├─ _make_clock() → Mock Clock
│  ├─ _make_decision_engine() → Mock DecisionEngine
│  └─ _make_recovery() → Mock RecoveryManager
│
├─ TestScenario1_SuccessWithRestRequired
│  └─ test_success_with_combat_rest_completes_with_wait_next_cycle_state()
│     ├─ Setup: combat hp_ratio=0.65 (< 0.70)
│     ├─ Verify: SUCCESS
│     ├─ Assert: both combat & rest called
│     └─ Assert: result="success", state=WAIT_NEXT_CYCLE
│
├─ TestScenario2_SuccessWithoutRestRequired
│  └─ test_success_with_high_hp_no_rest_required()
│     ├─ Setup: combat hp_ratio=0.85 (≥ 0.70)
│     ├─ Verify: SUCCESS
│     ├─ Assert: both combat & rest called (but no-op)
│     └─ Assert: result="success"
│
├─ TestScenario3_VerifyTimeoutWithRecovery
│  └─ test_verify_timeout_triggers_recovery_and_returns_to_idle()
│     ├─ Setup: verification_provider.verify() → TIMEOUT
│     ├─ Assert: combat.assert_NOT_called()
│     ├─ Assert: rest.assert_NOT_called()
│     └─ Assert: result="verify_timeout"
│
├─ TestScenario4_ExecutionErrorWithRecovery
│  └─ test_exception_during_success_path_triggers_recovery()
│     ├─ Setup: combat_resolver raises RuntimeError
│     ├─ Setup: recovery_manager.build_exception_recovery_plan() → [RecoveryStep]
│     ├─ Assert: recovery.build_exception_recovery_plan() called
│     └─ Assert: result="execution_error"
│
├─ TestScenario5_NoEvent
│  └─ test_no_observation_results_in_no_event()
│     ├─ Setup: observation_provider.get_latest_observation() → None
│     ├─ Setup: prediction.predicted_spawn_ts = None
│     ├─ Assert: ALL executors.assert_NOT_called()
│     └─ Assert: result="no_event"
│
├─ TestScenario6_LateEventMissed
│  └─ test_event_predicted_but_not_observed_is_late_missed()
│     ├─ Setup: observation_provider.get_latest_observation() → None
│     ├─ Setup: prediction.predicted_spawn_ts = 115.0 (expected!)
│     ├─ Assert: ALL executors.assert_NOT_called()
│     └─ Assert: result="late_event_missed"
│
├─ TestMultipleCycles
│  └─ test_run_multiple_cycles_produces_correct_count()
│     ├─ Setup: Run 5 cycles
│     ├─ Assert: len(results) == 5
│     └─ Assert: cycle_ids are 0, 1, 2, 3, 4
│
└─ TestDriftCalculation
   ├─ test_positive_drift_when_event_is_late()
   │  ├─ Setup: actual_spawn_ts=116.5, predicted=115.0
   │  └─ Assert: drift_s ≈ 1.5
   │
   └─ test_negative_drift_when_event_is_early()
      ├─ Setup: actual_spawn_ts=113.8, predicted=115.0
      └─ Assert: drift_s ≈ -1.2
```

---

## 🔍 Zmappowanie na Linie w Orchestratorze

```python
# from CycleOrchestrator

line 73-75:     def run_cycles(total_cycles, initial_cycle_id):
line 76-80:         [Scenario1-6] cycle_results.append(_run_single_cycle(...))

line 82-105:    def _run_single_cycle(cycle_id):
line 103-104:       IF observation is None:
line 105:               return _complete_cycle_without_observation()  [Scenario 5-6]
line 108-110:       return _complete_cycle_with_observation()        [Scenario 1-4]

line 112-160:   def _complete_cycle_without_observation():
line 121:       IF predicted_spawn_ts is None:
line 122:           result = "no_event"                              [Scenario 5]
line 125:           result = "late_event_missed"                     [Scenario 6]

line 162-475:   def _complete_cycle_with_observation():
line 207:       verify_result = verification_provider.verify()
line 209-233:   IF verify_result == TIMEOUT:
                    return ...                                         [Scenario 3]

line 253-280:   IF verify_result == FAILURE:
                    return ...

line 283-485:   TRY:
line 284-289:       _run_success_path()                              [Scenario 1-2]
line 290-485:   EXCEPT Exception:
                    recovery.build_exception_recovery_plan()          [Scenario 4]
```

---

## 📈 Test Execution Flow

```
pytest tests/test_orchestrator_scenarios.py -v
│
├─ Load Test Module
│  └─ Import all dependencies
│     └─ Fix: RecoveryStep from botlab.domain.recovery ✓
│
├─ Discover Tests
│  ├─ TestScenario1::test_success_with_combat_rest...
│  ├─ TestScenario2::test_success_with_high_hp...
│  ├─ TestScenario3::test_verify_timeout...
│  ├─ TestScenario4::test_exception_during_success...
│  ├─ TestScenario5::test_no_observation...
│  ├─ TestScenario6::test_event_predicted_but_not...
│  ├─ TestMultipleCycles::test_run_multiple...
│  ├─ TestDriftCalculation::test_positive_drift...
│  └─ TestDriftCalculation::test_negative_drift...
│     └─ [9 tests collected in 0.09s] ✓
│
├─ Execute Each Test
│  ├─ Setup: Create mock dependencies
│  ├─ Setup: Instantiate CycleOrchestrator
│  ├─ Act: Call orchestrator.run_cycles(N)
│  ├─ Assert: Verify result state, calls, telemetry
│  └─ Teardown: Reset mocks
│
└─ Report Results
   ├─ All scenarios: PASS ✓
   ├─ Multi-cycle: PASS ✓
   ├─ Drift calculations: PASS ✓
   └─ [9 passed in Xs] ✓
```

---

## 🎯 Key Mock Points

### Scenario 1-2 (Success)
```python
verification_provider.verify() → VerificationOutcome.SUCCESS
combat_resolver.resolve_combat() → CombatOutcome(hp_ratio=0.65 or 0.85)
rest_provider.apply_rest() → RestOutcome(hp_ratio=0.95)
```

### Scenario 3 (Timeout)
```python
verification_provider.verify() → VerificationOutcome.TIMEOUT
# combat & rest NOT called
recovery.build_exception_recovery_plan() → [RecoveryStep(...)]
```

### Scenario 4 (Error)
```python
verification_provider.verify() → VerificationOutcome.SUCCESS
combat_resolver.resolve_combat() → raises RuntimeError
recovery.build_exception_recovery_plan() → [RecoveryStep(...)]
# rest NOT called (exception happened first)
```

### Scenario 5 (No Event)
```python
observation_provider.get_latest_observation() → None
scheduler_prediction.predicted_spawn_ts = None
# All execution ports: NOT called
```

### Scenario 6 (Late Missed)
```python
observation_provider.get_latest_observation() → None
scheduler_prediction.predicted_spawn_ts = 115.0  # ← Event WAS expected
# All execution ports: NOT called
```

---

## 🚀 Gotowość do Uruchomienia

```bash
# Sprawdzenie że testy się zbierają
pytest tests/test_orchestrator_scenarios.py --collect-only

# Uruchomienie wszystkich 9 testów
pytest tests/test_orchestrator_scenarios.py -v

# Uruchomienie z coverage
pytest tests/test_orchestrator_scenarios.py --cov=src/botlab/application -v

# Uruchomienie konkretnego scenariusza
pytest tests/test_orchestrator_scenarios.py::TestScenario1_SuccessWithRestRequired -v
```

---

## ✅ Status Implementacji

| Komponent | Status | Linia | Test |
|-----------|--------|-------|------|
| Scenario 1 | ✓ Complete | 174-242 | TestScenario1 |
| Scenario 2 | ✓ Complete | 244-308 | TestScenario2 |
| Scenario 3 | ✓ Complete | 314-396 | TestScenario3 |
| Scenario 4 | ✓ Complete | 405-499 | TestScenario4 |
| Scenario 5 | ✓ Complete | 499-565 | TestScenario5 |
| Scenario 6 | ✓ Complete | 571-645 | TestScenario6 |
| Multi-cycle | ✓ Complete | 654-691 | TestMultipleCycles |
| Drift Calc | ✓ Complete | 694-798 | TestDriftCalculation |
| **TOTAL** | ✓ 9 tests | **~800 lines** | **All Scenarios** |
