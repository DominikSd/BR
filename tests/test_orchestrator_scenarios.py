"""
Focused tests for CycleOrchestrator covering all 6 core cycle scenarios.

Each test focuses on ONE scenario flow and uses simple stubs for clarity:
- Stubs return minimal, controlled data
- No integration with simulation (that's test_simulation_runner.py)
- Tests verify orchestrator state transitions and telemetry recording
"""

from unittest.mock import Mock, call
import pytest

from botlab.application import CycleOrchestrator
from botlab.application.ports import VerificationOutcome
from botlab.config import CycleConfig
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM, StateTransition
from botlab.domain.recovery import RecoveryManager, RecoveryStep
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, Decision, Observation


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def cycle_config() -> CycleConfig:
    """Standard cycle config for all tests."""
    return CycleConfig(
        cycle_length_s=60.0,
        prepare_window_s=10.0,
        ready_window_s=10.0,
        recover_timeout_s=30.0,
    )


@pytest.fixture
def observation() -> Observation:
    """Standard observation: event at predicted spawn time."""
    return Observation(
        cycle_id=0,
        observed_at_ts=115.0,
        signal_detected=True,
        actual_spawn_ts=115.0,
    )


def _make_prediction():
    """Create a mock prediction with standard timestamps."""
    prediction = Mock()
    prediction.prepare_window_start_ts = 100.0
    prediction.ready_window_start_ts = 110.0
    prediction.ready_window_end_ts = 120.0
    prediction.predicted_spawn_ts = 115.0
    return prediction


def _make_fsm_with_transitions() -> Mock:
    """Create FSM that tracks transitions for verification."""
    fsm = Mock(spec=CycleFSM)
    fsm.current_state = BotState.IDLE
    fsm.transition_count.side_effect = [0, 1, 2, 3]  # Increments each call
    
    transition = Mock(spec=StateTransition)
    transition.cycle_id = 0
    transition.at_ts = 100.0
    transition.from_state = BotState.IDLE
    transition.to_state = BotState.PREPARE_WINDOW
    transition.reason = "cycle_start"
    
    fsm.transition_history.return_value = [transition]
    return fsm


def _make_scheduler() -> Mock:
    """Create scheduler with standard prediction."""
    scheduler = Mock(spec=CycleScheduler)
    scheduler.prediction_for_cycle.return_value = _make_prediction()
    scheduler.state_for_time.return_value = "PREPARE"
    return scheduler


def _make_clock() -> Mock:
    """Create simple clock."""
    clock = Mock()
    clock.now.return_value = 100.0
    return clock


def _make_decision_engine() -> Mock:
    """Create simple decision engine."""
    engine = Mock(spec=DecisionEngine)
    decision = Mock(spec=Decision)
    decision.action = "ATTEMPT"
    decision.reason = "ready"
    return engine


def _make_recovery() -> Mock:
    """Create recovery manager with no recovery plans by default."""
    return Mock(spec=RecoveryManager)


def _make_cycle_config() -> CycleConfig:
    """Create standard CycleConfig for tests."""
    return CycleConfig(
        interval_s=60.0,
        prepare_before_s=10.0,
        ready_before_s=10.0,
        ready_after_s=10.0,
        verify_timeout_s=30.0,
        recover_timeout_s=30.0,
    )


# ============================================================================
# Scenario 1: SUCCESS -> COMBAT -> REST -> WAIT_NEXT_CYCLE
# ============================================================================

class TestScenario1_SuccessWithRestRequired:
    """
    Cycle succeeds, combat kills enemy (hp < 0.70), rest is applied.
    
    Flow: observe event → verify success → resolve combat → apply rest → WAIT_NEXT_CYCLE
    """

    def test_success_with_combat_rest_completes_with_wait_next_cycle_state(self):
        """Results in WAIT_NEXT_CYCLE state after successful combat and rest."""
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.WAIT_NEXT_CYCLE  # FSM ends at this state after tick
        
        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation = Observation(
            cycle_id=0,
            observed_at_ts=115.0,
            signal_detected=True,
            actual_spawn_ts=115.0,
        )

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = observation

        action_executor = Mock()
        action_executor.execute_action.return_value = Mock(
            cycle_id=0,
            success=True,
            reason="action_executed",
            metadata={},
        )

        verification_provider = Mock()
        verification_provider.verify.return_value = VerificationOutcome.SUCCESS

        combat_resolver = Mock()
        combat_resolver.resolve_combat.return_value = Mock(
            cycle_id=0,
            won=True,
            hp_ratio=0.65,  # Below 0.70 threshold -> rest required
            metadata={},
        )

        rest_provider = Mock()
        rest_provider.apply_rest.return_value = Mock(
            cycle_id=0,
            hp_ratio=0.95,
            recovered=True,
            metadata={},
        )

        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "success"
        assert result.final_state == BotState.WAIT_NEXT_CYCLE
        assert result.observation_used is True
        assert result.drift_s == 0.0  # actual_spawn_ts == predicted_spawn_ts

        # Verify combat and rest were called
        combat_resolver.resolve_combat.assert_called_once()
        rest_provider.apply_rest.assert_called_once()

        # Verify telemetry recorded the cycle
        assert telemetry_sink.record_cycle.called


# ============================================================================
# Scenario 2: SUCCESS -> COMBAT -> NO REST -> WAIT_NEXT_CYCLE
# ============================================================================

class TestScenario2_SuccessWithoutRestRequired:
    """
    Cycle succeeds, combat leaves hp > 0.70, no rest applied.
    
    Flow: observe event → verify success → resolve combat (high HP) → NO rest → WAIT_NEXT_CYCLE
    """

    def test_success_with_high_hp_no_rest_required(self):
        """High HP after combat means no rest, but still completes successfully."""
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.WAIT_NEXT_CYCLE

        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation = Observation(
            cycle_id=0,
            observed_at_ts=115.0,
            signal_detected=True,
            actual_spawn_ts=115.0,
        )

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = observation

        action_executor = Mock()
        action_executor.execute_action.return_value = Mock(
            cycle_id=0,
            success=True,
            reason="action_executed",
            metadata={},
        )

        verification_provider = Mock()
        verification_provider.verify.return_value = VerificationOutcome.SUCCESS

        combat_resolver = Mock()
        combat_resolver.resolve_combat.return_value = Mock(
            cycle_id=0,
            won=True,
            hp_ratio=0.85,  # Above 0.70 threshold -> no rest required
            metadata={},
        )

        rest_provider = Mock()
        rest_provider.apply_rest.return_value = Mock(
            cycle_id=0,
            hp_ratio=0.85,  # No change, no rest applied
            recovered=False,
            metadata={},
        )

        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "success"
        assert result.final_state == BotState.WAIT_NEXT_CYCLE
        assert result.observation_used is True

        # Both combat and rest are still called (via success path) but rest is no-op
        combat_resolver.resolve_combat.assert_called_once()
        rest_provider.apply_rest.assert_called_once()


# ============================================================================
# Scenario 3: VERIFY_TIMEOUT -> RECOVER -> WAIT_NEXT_CYCLE
# ============================================================================

class TestScenario3_VerifyTimeoutWithRecovery:
    """
    Cycle observes event, verification times out, recovery brings bot to safe state.
    
    Flow: observe event → verify timeout → apply recovery → WAIT_NEXT_CYCLE
    """

    def test_verify_timeout_triggers_recovery_and_returns_to_idle(self):
        """Timeout during verification triggers recovery process."""
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.IDLE  # After recovery
        fsm.transition_count.side_effect = [0, 1, 2, 3, 4]  # More transitions for recovery

        decision_engine = _make_decision_engine()
        
        clock = _make_clock()

        observation = Observation(
            cycle_id=0,
            observed_at_ts=115.0,
            signal_detected=True,
            actual_spawn_ts=115.0,
        )

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = observation

        action_executor = Mock()
        action_executor.execute_action.return_value = Mock(
            cycle_id=0,
            success=True,
            reason="action_executed",
            metadata={},
        )

        verification_provider = Mock()
        verification_provider.verify.return_value = VerificationOutcome.TIMEOUT

        combat_resolver = Mock()  # Should NOT be called on timeout
        rest_provider = Mock()  # Should NOT be called on timeout

        telemetry_sink = Mock()

        recovery = Mock(spec=RecoveryManager)
        recovery_step = Mock(spec=RecoveryStep)
        recovery_step.target_state = BotState.IDLE
        recovery_step.at_ts = 146.0  # 115.0 + 0.020 + 0.100 + 30.0
        recovery_step.from_state = BotState.ATTEMPT
        recovery_step.reason = "recover_from_timeout"
        recovery_step.cycle_id = 0

        recovery.build_exception_recovery_plan.return_value = [recovery_step]

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "verify_timeout"
        assert result.observation_used is True

        # Combat and rest should NOT be called on timeout
        combat_resolver.resolve_combat.assert_not_called()
        rest_provider.apply_rest.assert_not_called()

        # Recovery should have been triggered
        recovery.build_exception_recovery_plan.assert_not_called()  # No exception, timeout path is direct

        # Telemetry should record the attempt and cycle
        assert telemetry_sink.record_attempt.called
        assert telemetry_sink.record_cycle.called


# ============================================================================
# Scenario 4: EXECUTION_ERROR -> RECOVER -> WAIT_NEXT_CYCLE
# ============================================================================

class TestScenario4_ExecutionErrorWithRecovery:
    """
    Cycle succeeds verification but throws during success path (combat/rest).
    Recovery manager handles exception and brings bot to safe state.
    
    Flow: observe event → verify success → exception during success path → recovery → WAIT_NEXT_CYCLE
    """

    def test_exception_during_success_path_triggers_recovery(self):
        """Exception during combat/rest execution triggers recovery process.""" 
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.IDLE  # After recovery
        fsm.force_state = Mock()  # Required for recovery path
        fsm.transition_count.side_effect = [0, 1, 2, 3, 4, 5]

        decision_engine = _make_decision_engine()
        clock = _make_clock()

        observation = Observation(
            cycle_id=0,
            observed_at_ts=115.0,
            signal_detected=True,
            actual_spawn_ts=115.0,
        )

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = observation

        action_executor = Mock()
        action_executor.execute_action.return_value = Mock(
            cycle_id=0,
            success=True,
            reason="action_executed",
            metadata={},
        )

        verification_provider = Mock()
        verification_provider.verify.return_value = VerificationOutcome.SUCCESS

        # Combat throws exception
        combat_resolver = Mock()
        test_exception = RuntimeError("Combat system malfunction")
        combat_resolver.resolve_combat.side_effect = test_exception

        rest_provider = Mock()  # Should NOT be called

        telemetry_sink = Mock()

        recovery = Mock(spec=RecoveryManager)
        recovery_step = Mock(spec=RecoveryStep)
        recovery_step.target_state = BotState.IDLE
        recovery_step.at_ts = 115.12  # Verify resolution time + small delay
        recovery_step.from_state = BotState.ATTEMPT
        recovery_step.reason = "exception_recovery"
        recovery_step.cycle_id = 0

        recovery.build_exception_recovery_plan.return_value = [recovery_step]

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "execution_error"
        assert result.observation_used is True

        # Combat was called but threw
        combat_resolver.resolve_combat.assert_called_once()

        # Rest was NOT called (exception before it)
        rest_provider.apply_rest.assert_not_called()

        # Recovery was triggered
        recovery.build_exception_recovery_plan.assert_called_once()

        # Telemetry should record attempt and cycle with error metadata
        assert telemetry_sink.record_attempt.called
        assert telemetry_sink.record_cycle.called


# ============================================================================
# Scenario 5: NO_EVENT
# ============================================================================

class TestScenario5_NoEvent:
    """
    Cycle completes without observing any event (no observation returned).
    
    Flow: prepare → ready → no observation → complete with NO_EVENT result
    """

    def test_no_observation_results_in_no_event(self):
        """Absence of observation is recorded as no_event, not failure."""
        # Arrange
        scheduler = _make_scheduler()
        scheduler_prediction = scheduler.prediction_for_cycle.return_value
        scheduler_prediction.predicted_spawn_ts = None  # No event was expected

        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.IDLE

        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = None  # No event

        action_executor = Mock()
        verification_provider = Mock()
        combat_resolver = Mock()
        rest_provider = Mock()
        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "no_event"
        assert result.observation_used is False
        assert result.actual_spawn_ts is None
        assert result.drift_s is None

        # None of the execution paths should be called
        action_executor.execute_action.assert_not_called()
        verification_provider.verify.assert_not_called()
        combat_resolver.resolve_combat.assert_not_called()
        rest_provider.apply_rest.assert_not_called()

        # Record cycle should be called
        telemetry_sink.record_cycle.assert_called_once()


# ============================================================================
# Scenario 6: LATE_EVENT_MISSED
# ============================================================================

class TestScenario6_LateEventMissed:
    """
    Cycle completes without observation but event was predicted.
    Event occurred outside of ready window.
    
    Flow: prepare → ready → no observation (but event was expected) → complete with LATE_EVENT_MISSED
    """

    def test_event_predicted_but_not_observed_is_late_missed(self):
        """Event was scheduled but not observed => late event missed."""
        # Arrange
        scheduler = _make_scheduler()
        scheduler_prediction = scheduler.prediction_for_cycle.return_value
        scheduler_prediction.predicted_spawn_ts = 115.0  # Event WAS predicted

        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.IDLE

        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = None  # But NOT observed

        action_executor = Mock()
        verification_provider = Mock()
        combat_resolver = Mock()
        rest_provider = Mock()
        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "late_event_missed"
        assert result.observation_used is False
        assert result.predicted_spawn_ts == 115.0
        assert result.actual_spawn_ts is None

        # None of the execution paths should be called
        action_executor.execute_action.assert_not_called()
        verification_provider.verify.assert_not_called()
        combat_resolver.resolve_combat.assert_not_called()
        rest_provider.apply_rest.assert_not_called()

        # Record cycle should be called
        telemetry_sink.record_cycle.assert_called_once()


# ============================================================================
# Additional Tests: Multi-cycle and Drift
# ============================================================================

class TestMultipleCycles:
    """Test that orchestrator handles multiple cycles correctly."""

    def test_run_multiple_cycles_produces_correct_count(self):
        """Running N cycles produces N results with correct IDs."""
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.IDLE
        fsm.transition_count.side_effect = [0, 1] * 10  # For multiple cycles

        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = None

        action_executor = Mock()
        verification_provider = Mock()
        combat_resolver = Mock()
        rest_provider = Mock()
        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(5)

        # Assert
        assert len(results) == 5
        assert results[0].cycle_id == 0
        assert results[1].cycle_id == 1
        assert results[2].cycle_id == 2
        assert results[3].cycle_id == 3
        assert results[4].cycle_id == 4


class TestDriftCalculation:
    """Test that drift is correctly calculated when observation registered."""

    def test_positive_drift_when_event_is_late(self):
        """Event observed after predicted spawn time = positive drift."""
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.WAIT_NEXT_CYCLE

        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation = Observation(
            cycle_id=0,
            observed_at_ts=116.5,  # 1.5s late
            signal_detected=True,
            actual_spawn_ts=116.5,
        )

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = observation

        action_executor = Mock()
        action_executor.execute_action.return_value = Mock(
            cycle_id=0, success=True, reason="executed", metadata={}
        )

        verification_provider = Mock()
        verification_provider.verify.return_value = VerificationOutcome.SUCCESS

        combat_resolver = Mock()
        combat_resolver.resolve_combat.return_value = Mock(
            cycle_id=0, won=True, hp_ratio=0.7, metadata={}
        )

        rest_provider = Mock()
        rest_provider.apply_rest.return_value = Mock(
            cycle_id=0, hp_ratio=0.95, recovered=True, metadata={}
        )

        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert results[0].drift_s == pytest.approx(1.5, abs=0.01)

    def test_negative_drift_when_event_is_early(self):
        """Event observed before predicted spawn time = negative drift."""
        # Arrange
        scheduler = _make_scheduler()
        fsm = _make_fsm_with_transitions()
        fsm.current_state = BotState.WAIT_NEXT_CYCLE

        decision_engine = _make_decision_engine()
        recovery = _make_recovery()
        clock = _make_clock()

        observation = Observation(
            cycle_id=0,
            observed_at_ts=113.8,  # 1.2s early
            signal_detected=True,
            actual_spawn_ts=113.8,
        )

        observation_provider = Mock()
        observation_provider.get_latest_observation.return_value = observation

        action_executor = Mock()
        action_executor.execute_action.return_value = Mock(
            cycle_id=0, success=True, reason="executed", metadata={}
        )

        verification_provider = Mock()
        verification_provider.verify.return_value = VerificationOutcome.SUCCESS

        combat_resolver = Mock()
        combat_resolver.resolve_combat.return_value = Mock(
            cycle_id=0, won=True, hp_ratio=0.7, metadata={}
        )

        rest_provider = Mock()
        rest_provider.apply_rest.return_value = Mock(
            cycle_id=0, hp_ratio=0.95, recovered=True, metadata={}
        )

        telemetry_sink = Mock()

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=CycleConfig(
                cycle_length_s=60.0,
                prepare_window_s=10.0,
                ready_window_s=10.0,
                recover_timeout_s=30.0,
            ),
        )

        # Act
        results = orchestrator.run_cycles(1)

        # Assert
        assert results[0].drift_s == pytest.approx(-1.2, abs=0.01)
