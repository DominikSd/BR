from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from botlab.adapters.live.models import LiveTargetDetection


MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


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
        screen_offset_xy: tuple[int, int] = (0, 0),
    ) -> None:
        self._logger = logger
        self._dry_run = dry_run
        self._screen_offset_xy = screen_offset_xy
        self._events: list[LiveInputEvent] = []

    @property
    def events(self) -> tuple[LiveInputEvent, ...]:
        return tuple(self._events)

    def right_click_target(self, target: LiveTargetDetection) -> None:
        click_x = int(target.screen_x)
        click_y = int(target.screen_y)
        absolute_x = self._screen_offset_xy[0] + click_x
        absolute_y = self._screen_offset_xy[1] + click_y
        execution_status = "dry_run"
        if not self._dry_run:
            execution_status = self._perform_right_click(
                absolute_x=absolute_x,
                absolute_y=absolute_y,
            )
        self._record(
            action="right_click_target",
            payload={
                "target_id": target.target_id,
                "screen_x": click_x,
                "screen_y": click_y,
                "absolute_x": absolute_x,
                "absolute_y": absolute_y,
                "dry_run": self._dry_run,
                "execution_status": execution_status,
            },
        )

    def press_key(self, key: str) -> None:
        self._record(
            action="press_key",
            payload={
                "key": key,
                "dry_run": self._dry_run,
            },
        )

    def press_sequence(self, keys: tuple[str, ...]) -> None:
        self._record(
            action="press_sequence",
            payload={
                "keys": list(keys),
                "dry_run": self._dry_run,
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

    def _record(self, *, action: str, payload: dict[str, Any]) -> None:
        event = LiveInputEvent(action=action, payload=payload)
        self._events.append(event)
        self._logger.info(
            "live_input action=%s payload=%s",
            action,
            payload,
        )
