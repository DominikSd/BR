from __future__ import annotations

from pathlib import Path

import pytest

from botlab.config import ConfigError, load_config, load_default_config
from botlab.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from botlab.types import (
    BotState,
    CombatSnapshot,
    CyclePrediction,
    Decision,
    Observation,
    TelemetryRecord,
)


def test_load_default_config_returns_settings() -> None:
    settings = load_default_config()

    assert settings.source_path == DEFAULT_CONFIG_PATH.resolve()
    assert settings.app.name == "botlab"
    assert settings.app.mode == "simulation"

    assert settings.cycle.interval_s == 45.0
    assert settings.cycle.prepare_before_s == 5.0
    assert settings.cycle.ready_before_s == 1.0
    assert settings.cycle.ready_after_s == 1.0
    assert settings.cycle.verify_timeout_s == 0.5
    assert settings.cycle.recover_timeout_s == 2.0

    assert settings.combat.low_hp_threshold == 0.35
    assert settings.combat.rest_start_threshold == 0.50
    assert settings.combat.rest_stop_threshold == 0.90
    assert settings.combat.default_profile_name == "basic_farmer"

    assert settings.telemetry.sqlite_path == (
        PROJECT_ROOT / "data/telemetry/botlab.sqlite3"
    ).resolve()
    assert settings.telemetry.log_path == (PROJECT_ROOT / "logs/botlab.log").resolve()
    assert settings.telemetry.log_level == "INFO"

    assert settings.vision.enabled is False
    assert settings.live.dry_run is False
    assert settings.live.enable_real_input is False
    assert settings.live.enable_real_clicks is False
    assert settings.live.enable_real_keys is False
    assert settings.live.stall_timeout_s == 1.0
    assert settings.live.perception_confidence_threshold == 0.75
    assert settings.live.occupied_confidence_threshold == 0.75
    assert settings.live.merge_distance_px == 28
    assert settings.live.sample_frames_directory == (
        PROJECT_ROOT / "assets/live/sample_frames/raw"
    ).resolve()
    assert settings.live.benchmark_dataset_directory == (
        PROJECT_ROOT / "assets/live/sample_frames"
    ).resolve()
    assert settings.live.scene_profile_path is None
    assert settings.live.scene_calibration_offset_xy == (0, 0)
    assert settings.live.mobs_template_directory == (
        PROJECT_ROOT / "assets/live/templates/mobs"
    ).resolve()
    assert settings.live.occupied_template_directory == (
        PROJECT_ROOT / "assets/live/templates/occupied"
    ).resolve()
    assert settings.live.template_match_stride_px == 4
    assert settings.live.template_rotations_deg == (0, 90, 180, 270)
    assert settings.live.marker_min_red == 170
    assert settings.live.marker_confidence_threshold == 0.55
    assert settings.live.rescue_upper_scan_confidence_threshold == 0.74
    assert settings.live.rescue_upper_scan_stride_px == 8
    assert settings.live.rescue_pseudo_marker_size_px == 8
    assert settings.live.rescue_pseudo_marker_offset_y_px == 10
    assert settings.live.combat_indicator_min_red == 170
    assert settings.live.combat_indicator_min_ratio == 0.01
    assert settings.live.hp_bar_min_red == 150
    assert settings.live.hp_bar_min_fill_ratio == 0.01
    assert settings.live.condition_bar_min_green == 120
    assert settings.live.condition_bar_min_fill_ratio == 0.01
    assert settings.live.reward_min_red == 170
    assert settings.live.reward_min_ratio == 0.01
    assert settings.live.swords_min_green == 120
    assert settings.live.swords_confidence_threshold == 0.25
    assert settings.live.occupied_template_match_min_green_ratio == 0.01
    assert settings.live.occupied_local_roi_width_px == 64
    assert settings.live.confirmation_roi_width_px == 88
    assert settings.live.confirmation_alignment_weight == 0.25
    assert settings.live.confirmation_foreground_weight == 0.10
    assert settings.live.confirmation_max_horizontal_offset_px == 56
    assert settings.live.confirmation_min_vertical_offset_px == 12
    assert settings.live.confirmation_max_vertical_offset_px == 180
    assert settings.live.candidate_confirmation_frames == 1
    assert settings.live.candidate_loss_frames == 2
    assert settings.live.engage_verify_delay_s == 0.20
    assert settings.live.engage_click_offset_y_px == 0
    assert settings.live.engage_target_match_max_distance_px == 72
    assert settings.live.engage_min_target_confidence == 0.70
    assert settings.live.engage_min_seen_frames == 1
    assert settings.live.preview_fast_mode is False
    assert settings.live.preview_skip_fallback_confirmation is False
    assert settings.live.preview_render_aux_boxes is True
    assert settings.live.preview_crop_to_spawn_roi is False
    assert settings.live.preview_crop_padding_px == 0
    assert settings.live.preview_analyze_every_nth_frame == 1
    assert settings.live.rest_resource_sample_count == 3
    assert settings.live.rest_resource_sample_interval_s == 0.10
    assert settings.live.rest_resource_min_confidence == 0.60
    assert settings.live.rest_resource_max_ticks == 6
    assert settings.live.rest_resource_growth_min_delta == 0.01
    assert settings.live.rest_resource_warning_spread_threshold == 0.10
    assert settings.live.rest_resource_stall_warning_ticks == 2
    assert settings.live.preview_refresh_interval_ms == 120
    assert settings.live.preview_max_width_px == 1600
    assert settings.live.preview_max_height_px == 900


def test_live_config_profile_can_be_loaded_from_yaml() -> None:
    settings = load_config(PROJECT_ROOT / "config" / "live_dry_run.yaml")

    assert settings.app.mode == "live"
    assert settings.live.dry_run is True
    assert settings.live.enable_real_input is False
    assert settings.live.enable_real_clicks is False
    assert settings.live.enable_real_keys is False
    assert settings.live.capture_region == (0, 0, 1280, 720)
    assert settings.live.spawn_roi == (320, 140, 640, 320)
    assert settings.live.dry_run_profile == "single_spot_mvp"
    assert settings.live.perception_confidence_threshold == 0.75
    assert settings.live.sample_frames_directory.name == "raw"
    assert settings.live.benchmark_dataset_directory.name == "sample_frames"
    assert settings.live.scene_profile_path is None
    assert settings.live.scene_calibration_offset_xy == (0, 0)
    assert settings.live.marker_red_green_delta == 35
    assert settings.live.combat_indicator_red_green_delta == 35
    assert settings.live.hp_bar_red_green_delta == 40
    assert settings.live.condition_bar_green_red_delta == 15
    assert settings.live.reward_min_green == 130
    assert settings.live.swords_green_red_delta == 20
    assert settings.live.swords_min_blob_pixels == 2
    assert settings.live.occupied_template_match_min_green_ratio == 0.01
    assert settings.live.occupied_local_roi_offset_y_px == -42
    assert settings.live.confirmation_confidence_threshold == 0.60
    assert settings.live.confirmation_alignment_weight == 0.25
    assert settings.live.confirmation_foreground_weight == 0.10
    assert settings.live.confirmation_max_horizontal_offset_px == 56
    assert settings.live.engage_verify_delay_s == 0.20
    assert settings.live.engage_target_match_max_distance_px == 72
    assert settings.live.engage_min_target_confidence == 0.70
    assert settings.live.engage_min_seen_frames == 1
    assert settings.live.preview_fast_mode is False
    assert settings.live.preview_skip_fallback_confirmation is False
    assert settings.live.preview_render_aux_boxes is True
    assert settings.live.preview_crop_to_spawn_roi is False
    assert settings.live.preview_crop_padding_px == 0
    assert settings.live.preview_analyze_every_nth_frame == 1
    assert settings.live.rest_resource_sample_count == 3
    assert settings.live.rest_resource_sample_interval_s == 0.10
    assert settings.live.rest_resource_min_confidence == 0.60
    assert settings.live.rest_resource_max_ticks == 6
    assert settings.live.rest_resource_growth_min_delta == 0.01
    assert settings.live.rest_resource_warning_spread_threshold == 0.10
    assert settings.live.rest_resource_stall_warning_ticks == 2
    assert settings.live.preview_refresh_interval_ms == 120


def test_live_real_mvp_config_can_be_loaded_from_yaml() -> None:
    settings = load_config(PROJECT_ROOT / "config" / "live_real_mvp.yaml")

    assert settings.app.mode == "live"
    assert settings.live.dry_run is False
    assert settings.live.enable_real_input is False
    assert settings.live.enable_real_clicks is False
    assert settings.live.enable_real_keys is False
    assert settings.live.debug_directory.name == "live_real_debug"
    assert settings.live.benchmark_dataset_directory.name == "sample_frames"
    assert settings.live.capture_region == (0, 0, 0, 0)
    assert settings.live.spawn_roi == (80, 40, 1450, 860)
    assert settings.live.scene_profile_path is not None
    assert settings.live.scene_profile_path.name == "single_spot_scene.json"
    assert settings.live.scene_zone_overlay_visible is False
    assert settings.live.scene_calibration_offset_xy == (0, 0)
    assert settings.live.scene_reference_anchor_mode == "frame_center"
    assert settings.live.scene_reference_anchor_xy == (960, 520)
    assert settings.live.marker_color_mode == "yellow"
    assert settings.live.marker_min_red == 150
    assert settings.live.marker_min_green == 110
    assert settings.live.marker_red_green_delta == 20
    assert settings.live.marker_green_blue_delta == 8
    assert settings.live.marker_red_blue_delta == 15
    assert settings.live.marker_red_green_balance_delta == 70
    assert settings.live.marker_min_blob_pixels == 4
    assert settings.live.marker_max_blob_pixels == 180
    assert settings.live.marker_min_width_px == 4
    assert settings.live.marker_max_width_px == 28
    assert settings.live.marker_min_height_px == 4
    assert settings.live.marker_max_height_px == 28
    assert settings.live.marker_min_fill_density == 0.04
    assert settings.live.marker_max_fill_density == 0.95
    assert settings.live.marker_dark_core_max_rgb == 135
    assert settings.live.marker_min_dark_core_ratio == 0.00
    assert settings.live.marker_confidence_threshold == 0.30
    assert settings.live.rescue_upper_scan_confidence_threshold == 0.74
    assert settings.live.rescue_upper_scan_stride_px == 8
    assert settings.live.rescue_pseudo_marker_size_px == 8
    assert settings.live.rescue_pseudo_marker_offset_y_px == 10
    assert settings.live.player_veto_enabled is True
    assert settings.live.player_veto_roi_width_px == 190
    assert settings.live.player_veto_roi_height_px == 70
    assert settings.live.player_veto_roi_offset_y_px == -28
    assert settings.live.player_veto_green_min_green == 105
    assert settings.live.player_veto_green_red_delta == 8
    assert settings.live.player_veto_green_blue_delta == 4
    assert settings.live.player_veto_min_pixels == 8
    assert settings.live.player_veto_min_width_px == 12
    assert settings.live.player_veto_max_height_px == 24
    assert settings.live.ice_mob_signature_enabled is True
    assert settings.live.ice_mob_min_blue == 135
    assert settings.live.ice_mob_min_green == 105
    assert settings.live.ice_mob_min_brightness == 118
    assert settings.live.ice_mob_blue_red_tolerance == 16
    assert settings.live.ice_mob_min_pixels == 42
    assert settings.live.ice_mob_min_ratio == 0.10
    assert settings.live.ice_mob_focus_width_ratio == 0.62
    assert settings.live.ice_mob_focus_height_ratio == 0.68
    assert settings.live.ice_mob_max_dark_ratio == 0.34
    assert settings.live.ice_mob_max_brown_ratio == 0.18
    assert settings.live.combat_indicator_min_ratio == 0.01
    assert settings.live.reward_min_ratio == 0.01
    assert settings.live.template_match_stride_px == 8
    assert settings.live.enable_fallback_confirmation is False
    assert settings.live.confirmation_anchor_search_enabled is True
    assert settings.live.confirmation_anchor_only is True
    assert settings.live.engage_verify_delay_s == 0.20
    assert settings.live.engage_min_target_confidence == 0.80
    assert settings.live.engage_min_seen_frames == 2
    assert settings.live.max_seed_hits_for_confirmation == 6
    assert settings.live.preview_fast_mode is True
    assert settings.live.preview_skip_fallback_confirmation is True
    assert settings.live.preview_render_aux_boxes is False
    assert settings.live.preview_crop_to_spawn_roi is True
    assert settings.live.preview_crop_padding_px == 16
    assert settings.live.preview_analyze_every_nth_frame == 2
    assert settings.live.preview_refresh_interval_ms == 150
    assert settings.live.preview_max_width_px == 1280
    assert settings.live.preview_max_height_px == 720
    assert settings.live.rest_resource_sample_count == 5
    assert settings.live.rest_resource_sample_interval_s == 0.10
    assert settings.live.rest_resource_min_confidence == 0.65
    assert settings.live.rest_resource_max_ticks == 8
    assert settings.live.rest_resource_growth_min_delta == 0.01
    assert settings.live.rest_resource_warning_spread_threshold == 0.08
    assert settings.live.rest_resource_stall_warning_ticks == 2


def test_cycle_prediction_window_methods() -> None:
    prediction = CyclePrediction(
        cycle_id=1,
        predicted_spawn_ts=100.0,
        interval_s=45.0,
        prepare_window_start_ts=95.0,
        ready_window_start_ts=99.0,
        ready_window_end_ts=101.0,
        based_on_observation_count=3,
    )

    assert prediction.is_in_prepare_window(96.0) is True
    assert prediction.is_in_prepare_window(99.0) is False

    assert prediction.is_in_ready_window(99.0) is True
    assert prediction.is_in_ready_window(100.0) is True
    assert prediction.is_in_ready_window(101.0) is True
    assert prediction.is_in_ready_window(101.1) is False


def test_domain_models_can_be_instantiated() -> None:
    observation = Observation(
        cycle_id=7,
        observed_at_ts=123.456,
        signal_detected=True,
        actual_spawn_ts=123.400,
        source="simulation",
        confidence=0.98,
        metadata={"frame": 17},
    )

    decision = Decision(
        cycle_id=7,
        state=BotState.READY_WINDOW,
        next_state=BotState.ATTEMPT,
        action="attempt_interaction",
        reason="signal_detected_in_ready_window",
        decided_at_ts=123.500,
        metadata={"confidence": 0.98},
    )

    snapshot = CombatSnapshot(
        hp_ratio=0.72,
        turn_index=2,
        enemy_count=1,
        strategy="default",
        in_combat=True,
        combat_started_ts=124.0,
        combat_finished_ts=None,
        metadata={"target": "dummy"},
    )

    telemetry = TelemetryRecord(
        cycle_id=7,
        event_ts=123.600,
        state=BotState.VERIFY,
        expected_spawn_ts=123.400,
        actual_spawn_ts=123.410,
        drift_s=0.010,
        state_enter=BotState.ATTEMPT,
        state_exit=BotState.VERIFY,
        reason="attempt_completed",
        reaction_ms=45.0,
        verification_ms=120.0,
        result="pending",
        final_state=BotState.VERIFY,
        metadata={"attempt_id": 1},
    )

    assert observation.signal_detected is True
    assert decision.next_state is BotState.ATTEMPT
    assert snapshot.in_combat is True
    assert telemetry.state is BotState.VERIFY


def test_missing_required_section_raises_config_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 5.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Brak wymaganych sekcji konfiguracji"):
        load_config(invalid_config)


def test_invalid_cycle_values_raise_config_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid_cycle.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 50.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "combat:",
                "  low_hp_threshold: 0.35",
                "  rest_start_threshold: 0.50",
                "  rest_stop_threshold: 0.90",
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="prepare_before_s"):
        load_config(invalid_config)


def test_invalid_log_level_raises_config_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid_log_level.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 5.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "combat:",
                "  low_hp_threshold: 0.35",
                "  rest_start_threshold: 0.50",
                "  rest_stop_threshold: 0.90",
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "TRACE"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Nieprawidlowy poziom logowania"):
        load_config(invalid_config)


def test_combat_default_profile_name_can_be_loaded_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "combat-profile.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 5.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "combat:",
                "  low_hp_threshold: 0.35",
                "  rest_start_threshold: 0.50",
                "  rest_stop_threshold: 0.90",
                '  default_profile_name: "fast_farmer"',
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path)

    assert settings.combat.default_profile_name == "fast_farmer"
