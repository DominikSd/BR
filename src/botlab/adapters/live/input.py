from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from botlab.adapters.live.models import LiveTargetDetection


@dataclass(slots=True, frozen=True)
class LiveInputEvent:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)


class LiveInputDriver:
    def __init__(self, *, logger: logging.Logger, dry_run: bool) -> None:
        self._logger = logger
        self._dry_run = dry_run
        self._events: list[LiveInputEvent] = []

    @property
    def events(self) -> tuple[LiveInputEvent, ...]:
        return tuple(self._events)

    def right_click_target(self, target: LiveTargetDetection) -> None:
        self._record(
            action="right_click_target",
            payload={
                "target_id": target.target_id,
                "screen_x": target.screen_x,
                "screen_y": target.screen_y,
                "dry_run": self._dry_run,
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

    def _record(self, *, action: str, payload: dict[str, Any]) -> None:
        event = LiveInputEvent(action=action, payload=payload)
        self._events.append(event)
        self._logger.info(
            "live_input action=%s payload=%s",
            action,
            payload,
        )
