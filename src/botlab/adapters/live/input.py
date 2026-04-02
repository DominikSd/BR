from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from botlab.adapters.live.models import LiveTargetDetection


MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_KEYUP = 0x0002


@dataclass(slots=True, frozen=True)
class LiveInputEvent:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)


class LiveInputDriver:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        dry_run: bool,
        enable_real_clicks: bool = False,
        enable_real_keys: bool = False,
        screen_offset_xy: tuple[int, int] = (0, 0),
    ) -> None:
        self._logger = logger
        self._dry_run = dry_run
        self._enable_real_clicks = enable_real_clicks
        self._enable_real_keys = enable_real_keys
        self._screen_offset_xy = screen_offset_xy
        self._events: list[LiveInputEvent] = []

    @property
    def events(self) -> tuple[LiveInputEvent, ...]:
        return tuple(self._events)

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def right_click_target(self, target: LiveTargetDetection) -> None:
        click_x = int(target.screen_x)
        click_y = int(target.screen_y)
        absolute_x = self._screen_offset_xy[0] + click_x
        absolute_y = self._screen_offset_xy[1] + click_y
        execution_status = "dry_run"
        if not self._dry_run and self._enable_real_clicks:
            execution_status = self._perform_right_click(
                absolute_x=absolute_x,
                absolute_y=absolute_y,
            )
        elif not self._dry_run:
            execution_status = "real_clicks_disabled"
        self._record(
            action="right_click_target",
            payload={
                "target_id": target.target_id,
                "screen_x": click_x,
                "screen_y": click_y,
                "absolute_x": absolute_x,
                "absolute_y": absolute_y,
                "dry_run": self._dry_run,
                "enable_real_clicks": self._enable_real_clicks,
                "execution_status": execution_status,
            },
        )

    def press_key(self, key: str) -> None:
        execution_status = "dry_run"
        if not self._dry_run and self._enable_real_keys:
            execution_status = self._perform_key_press(key=key)
        elif not self._dry_run:
            execution_status = "real_keys_disabled"
        self._record(
            action="press_key",
            payload={
                "key": key,
                "dry_run": self._dry_run,
                "enable_real_keys": self._enable_real_keys,
                "execution_status": execution_status,
            },
        )

    def press_sequence(self, keys: tuple[str, ...]) -> None:
        execution_statuses: list[str] = []
        if not self._dry_run and self._enable_real_keys:
            for key in keys:
                execution_statuses.append(self._perform_key_press(key=key))
                time.sleep(0.03)
        elif not self._dry_run:
            execution_statuses = ["real_keys_disabled" for _ in keys]
        else:
            execution_statuses = ["dry_run" for _ in keys]
        self._record(
            action="press_sequence",
            payload={
                "keys": list(keys),
                "dry_run": self._dry_run,
                "enable_real_keys": self._enable_real_keys,
                "execution_statuses": execution_statuses,
            },
        )

    def _perform_right_click(self, *, absolute_x: int, absolute_y: int) -> str:
        if sys.platform != "win32":
            self._logger.warning(
                "live_input real right click is unavailable on non-Windows platform."
            )
            return "real_click_unavailable_non_windows"
        try:
            user32 = ctypes.windll.user32
            user32.SetCursorPos(int(absolute_x), int(absolute_y))
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        except Exception as exc:  # pragma: no cover - defensive path for local GUI failures
            self._logger.exception("live_input real right click failed: %s", exc)
            return "real_click_failed"
        return "real_click_sent"

    def _perform_key_press(self, *, key: str) -> str:
        if sys.platform != "win32":
            self._logger.warning(
                "live_input real key press is unavailable on non-Windows platform."
            )
            return "real_key_unavailable_non_windows"
        virtual_key_code = self._resolve_virtual_key_code(key)
        if virtual_key_code is None:
            self._logger.warning("live_input unknown key for real press: %s", key)
            return "real_key_unknown"
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(virtual_key_code, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(virtual_key_code, 0, KEYEVENTF_KEYUP, 0)
        except Exception as exc:  # pragma: no cover - defensive path for local GUI failures
            self._logger.exception("live_input real key press failed: %s", exc)
            return "real_key_failed"
        return "real_key_sent"

    def _resolve_virtual_key_code(self, key: str) -> int | None:
        normalized = key.strip().lower()
        if not normalized:
            return None
        named_keys = {
            "esc": 0x1B,
            "escape": 0x1B,
            "space": 0x20,
            "r": 0x52,
            "1": 0x31,
            "2": 0x32,
            "3": 0x33,
            "4": 0x34,
            "5": 0x35,
            "6": 0x36,
            "7": 0x37,
            "8": 0x38,
            "9": 0x39,
            "0": 0x30,
        }
        if normalized in named_keys:
            return named_keys[normalized]
        if len(normalized) == 1 and normalized.isalpha():
            return ord(normalized.upper())
        return None

    def _record(self, *, action: str, payload: dict[str, Any]) -> None:
        event = LiveInputEvent(action=action, payload=payload)
        self._events.append(event)
        self._logger.info(
            "live_input action=%s payload=%s",
            action,
            payload,
        )
