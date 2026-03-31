from unittest.mock import Mock

import pytest

from botlab.application import CycleOrchestrator
from botlab.application.ports import VerificationOutcome
from botlab.config import CycleConfig
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, Observation


class TestCycleOrchestrator:
    @pytest.fixture
    def mock_scheduler(self) -> Mock:
        scheduler = Mock(spec=CycleScheduler)
        prediction = Mock()
        prediction.prepare_window_start_ts = 100.0
        prediction.ready_window_start_ts = 110.0
        prediction.ready_window_end_ts = 120.0
        prediction.predicted_spawn_ts = 115.0
        scheduler.prediction_for_cycle.return_value = prediction
        return scheduler

    @pytest.fixture
    def mock_decision_engine(self) -> Mock:
        return Mock(spec=DecisionEngine)

    @pytest.fixture
    def mock_fsm(self) -> Mock:
        fsm = Mock(spec=CycleFSM)
        fsm.current_state = BotState.IDLE
        fsm.transition_count.return_value = 0
        fsm.transition_history.return_value = []
        return fsm

    @pytest.fixture
    def mock_recovery(self) -> Mock:
        return Mock(spec=RecoveryManager)

    @pytest.fixture
    def mock_clock(self) -> Mock:
        clock = Mock()
        clock.now.return_value = 100.0
        return clock

    @pytest.fixture
    def mock_observation_provider(self) -> Mock:
        provider = Mock()
        observation = Observation(
            cycle_id=0,
            observed_at_ts=115.0,
            signal_detected=True,
            actual_spawn_ts=115.0,
        )
        provider.get_latest_observation.return_value = observation
        return provider

    @pytest.fixture
    def mock_action_executor(self) -> Mock:
        executor = Mock()
        result = Mock()
        result.success = True
        result.metadata = {}
        executor.execute_action.return_value = result
        return executor

    @pytest.fixture
    def mock_verification_provider(self) -> Mock:
        provider = Mock()
        provider.verify.return_value = VerificationOutcome.SUCCESS
        return provider

    @pytest.fixture
    def mock_combat_resolver(self) -> Mock:
        resolver = Mock()
        outcome = Mock()
        outcome.SUCCESS = "success"
        resolver.resolve_combat.return_value = outcome
        return resolver

    @pytest.fixture
    def mock_rest_provider(self) -> Mock:
        provider = Mock()
        outcome = Mock()
        outcome.COMPLETED = "completed"
        provider.apply_rest.return_value = outcome
        return provider

    @pytest.fixture
    def mock_telemetry_sink(self) -> Mock:
        return Mock()

    @pytest.fixture
    def cycle_config(self) -> CycleConfig:
        return CycleConfig(
            cycle_length_s=60.0,
            prepare_window_s=10.0,
            ready_window_s=10.0,
            recover_timeout_s=30.0,
        )

    @pytest.fixture
    def orchestrator(
        self,
        mock_scheduler,
        mock_decision_engine,
        mock_fsm,
        mock_recovery,
        mock_clock,
        mock_observation_provider,
        mock_action_executor,
        mock_verification_provider,
        mock_combat_resolver,
        mock_rest_provider,
        mock_telemetry_sink,
        cycle_config,
    ) -> CycleOrchestrator:
        return CycleOrchestrator(
            scheduler=mock_scheduler,
            decision_engine=mock_decision_engine,
            fsm=mock_fsm,
            recovery=mock_recovery,
            clock=mock_clock,
            observation_provider=mock_observation_provider,
            action_executor=mock_action_executor,
            verification_provider=mock_verification_provider,
            combat_resolver=mock_combat_resolver,
            rest_provider=mock_rest_provider,
            telemetry_sink=mock_telemetry_sink,
            cycle_config=cycle_config,
        )

    def test_run_cycles_zero_cycles_raises_value_error(self, orchestrator: CycleOrchestrator) -> None:
        with pytest.raises(ValueError, match="total_cycles musi być większe od 0"):
            orchestrator.run_cycles(0)

    def test_run_cycles_negative_cycles_raises_value_error(self, orchestrator: CycleOrchestrator) -> None:
        with pytest.raises(ValueError, match="total_cycles musi być większe od 0"):
            orchestrator.run_cycles(-1)

    def test_run_cycles_with_observation_success(self, orchestrator: CycleOrchestrator) -> None:
        results = orchestrator.run_cycles(1)
        
        assert len(results) == 1
        result = results[0]
        assert result.cycle_id == 0
        assert result.result == "success"
        assert result.observation_used is True