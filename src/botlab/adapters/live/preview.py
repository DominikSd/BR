from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from botlab.adapters.live.capture import create_capture
from botlab.adapters.live.models import LiveFrame, LiveSessionState
from botlab.adapters.live.perception import PerceptionAnalyzer, PerceptionFrameResult
from botlab.adapters.telemetry.logger import configure_telemetry_logger
from botlab.config import Settings

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except Exception:  # pragma: no cover - optional dependency path
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageTk = None


Clock = Callable[[], float]


@dataclass(slots=True, frozen=True)
class LiveVisionPreviewState:
    frame_source: str
    frame_width: int
    frame_height: int
    selected_target_id: str | None
    candidate_count: int
    free_target_count: int
    occupied_target_count: int
    detection_latency_ms: float
    selection_latency_ms: float
    total_reaction_latency_ms: float
    headline_lines: tuple[str, ...]


def build_live_preview_state(
    *,
    frame: LiveFrame,
    result: PerceptionFrameResult,
) -> LiveVisionPreviewState:
    selected_target = result.selected_target
    headline_lines = (
        f"source={frame.source}",
        f"targets={len(result.detections)} free={len(result.free_detections)} occupied={len(result.occupied_detections)}",
        f"selected={result.selected_target_id or 'None'}",
        f"detection={result.timings.detection_latency_ms:.1f}ms selection={result.timings.selection_latency_ms:.1f}ms reaction={result.timings.total_reaction_latency_ms:.1f}ms",
        f"candidate_hits={result.candidate_hit_count} merged={result.merged_hit_count}",
        (
            "selected_info="
            f"{selected_target.target_id} dist={selected_target.distance:.1f} conf={selected_target.confidence:.2f}"
            if selected_target is not None
            else "selected_info=None"
        ),
    )
    return LiveVisionPreviewState(
        frame_source=frame.source,
        frame_width=frame.width,
        frame_height=frame.height,
        selected_target_id=result.selected_target_id,
        candidate_count=len(result.detections),
        free_target_count=len(result.free_detections),
        occupied_target_count=len(result.occupied_detections),
        detection_latency_ms=result.timings.detection_latency_ms,
        selection_latency_ms=result.timings.selection_latency_ms,
        total_reaction_latency_ms=result.timings.total_reaction_latency_ms,
        headline_lines=headline_lines,
    )


class LiveVisionPreviewRenderer:
    def __init__(
        self,
        *,
        max_width_px: int,
        max_height_px: int,
    ) -> None:
        self._max_width_px = max_width_px
        self._max_height_px = max_height_px

    def render(
        self,
        *,
        frame: LiveFrame,
        result: PerceptionFrameResult,
        state: LiveVisionPreviewState,
    ):
        if Image is None or ImageDraw is None or ImageFont is None:
            raise RuntimeError("Preview vision wymaga Pillow.")
        if frame.image is not None:
            canvas = frame.image.convert("RGB").copy()
        else:
            canvas = Image.new("RGB", (frame.width, frame.height), color=(17, 24, 39))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        roi = result.roi
        draw.rectangle(
            (
                int(roi["x"]),
                int(roi["y"]),
                int(roi["x"]) + int(roi["width"]),
                int(roi["y"]) + int(roi["height"]),
            ),
            outline=(96, 165, 250),
            width=3,
        )
        draw.text((int(roi["x"]) + 4, int(roi["y"]) + 4), "spawn_roi", fill=(147, 197, 253), font=font)

        reference_x, reference_y = result.reference_point_xy
        draw.line((reference_x - 10, reference_y, reference_x + 10, reference_y), fill=(253, 224, 71), width=2)
        draw.line((reference_x, reference_y - 10, reference_x, reference_y + 10), fill=(253, 224, 71), width=2)

        for hit in result.raw_hits:
            if hit.label == "red_marker":
                color = (239, 68, 68)
            elif hit.label == "occupied_swords":
                color = (249, 115, 22)
            else:
                color = (56, 189, 248)
            draw.rectangle(
                (hit.x, hit.y, hit.x + hit.width, hit.y + hit.height),
                outline=color,
                width=1,
            )

        for detection in result.detections:
            bbox = detection.bbox or (
                max(0, detection.screen_x - 18),
                max(0, detection.screen_y - 24),
                36,
                48,
            )
            selected = detection.target_id == result.selected_target_id
            color = (239, 68, 68) if detection.occupied else (34, 197, 94)
            width = 4 if selected else 2
            draw.rectangle(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                outline=color,
                width=width,
            )
            draw.ellipse(
                (detection.screen_x - 4, detection.screen_y - 4, detection.screen_x + 4, detection.screen_y + 4),
                fill=color,
            )
            marker_bbox = detection.metadata.get("marker_bbox")
            if isinstance(marker_bbox, list) and len(marker_bbox) == 4:
                draw.rectangle(
                    (
                        marker_bbox[0],
                        marker_bbox[1],
                        marker_bbox[0] + marker_bbox[2],
                        marker_bbox[1] + marker_bbox[3],
                    ),
                    outline=(239, 68, 68),
                    width=2,
                )
            occupied_roi = detection.metadata.get("occupied_roi")
            if isinstance(occupied_roi, list) and len(occupied_roi) == 4:
                draw.rectangle(
                    (
                        occupied_roi[0],
                        occupied_roi[1],
                        occupied_roi[0] + occupied_roi[2],
                        occupied_roi[1] + occupied_roi[3],
                    ),
                    outline=(249, 115, 22),
                    width=1,
                )
            confirmation_roi = detection.metadata.get("confirmation_roi")
            if isinstance(confirmation_roi, list) and len(confirmation_roi) == 4:
                draw.rectangle(
                    (
                        confirmation_roi[0],
                        confirmation_roi[1],
                        confirmation_roi[0] + confirmation_roi[2],
                        confirmation_roi[1] + confirmation_roi[3],
                    ),
                    outline=(56, 189, 248),
                    width=1,
                )
            label = "occupied" if detection.occupied else "free"
            text_y = max(4, bbox[1] - 14)
            draw.text(
                (bbox[0], text_y),
                f"{detection.target_id} {label} conf={detection.confidence:.2f}",
                fill=(249, 250, 251),
                font=font,
            )
            if selected:
                draw.text(
                    (bbox[0], bbox[1] + bbox[3] + 4),
                    "selected",
                    fill=(253, 224, 71),
                    font=font,
                )

        self._draw_headline_box(draw=draw, font=font, state=state)
        return self._resize_to_fit(canvas)

    def _draw_headline_box(self, *, draw, font, state: LiveVisionPreviewState) -> None:
        line_height = 16
        padding = 8
        box_width = 520
        box_height = padding * 2 + len(state.headline_lines) * line_height
        draw.rectangle((8, 8, 8 + box_width, 8 + box_height), fill=(0, 0, 0))
        for index, line in enumerate(state.headline_lines):
            draw.text(
                (16, 16 + index * line_height),
                line,
                fill=(249, 250, 251),
                font=font,
            )

    def _resize_to_fit(self, image):
        width, height = image.size
        scale = min(
            1.0,
            self._max_width_px / max(1, width),
            self._max_height_px / max(1, height),
        )
        if scale >= 1.0:
            return image
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return image.resize((resized_width, resized_height))


class LiveVisionPreview:
    def __init__(
        self,
        *,
        settings: Settings,
        logger_name: str = "botlab.live.preview",
        enable_console: bool = True,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._logger = configure_telemetry_logger(
            telemetry_config=settings.telemetry,
            logger_name=logger_name,
            enable_console=enable_console,
        )
        self._clock = clock or time.time
        self._capture = create_capture(settings.live)
        self._analyzer = PerceptionAnalyzer(settings.live)
        self._renderer = LiveVisionPreviewRenderer(
            max_width_px=settings.live.preview_max_width_px,
            max_height_px=settings.live.preview_max_height_px,
        )
        self._session_state = LiveSessionState()
        self._tick_index = 0

    def render_next_frame(self):
        frame = self._capture.capture_frame(
            cycle_id=self._tick_index + 1,
            phase="observation",
            default_ts=float(self._clock()),
            session_state=self._session_state,
        )
        result = self._analyzer.analyze_frame(
            frame,
            cycle_id=self._tick_index + 1,
            phase="preview",
        )
        state = build_live_preview_state(frame=frame, result=result)
        image = self._renderer.render(frame=frame, result=result, state=state)
        self._tick_index += 1
        self._logger.info(
            "live_preview frame=%s targets=%s free=%s occupied=%s selected=%s reaction_ms=%.3f",
            self._tick_index,
            state.candidate_count,
            state.free_target_count,
            state.occupied_target_count,
            state.selected_target_id,
            state.total_reaction_latency_ms,
        )
        return state, image

    def run(self) -> int:
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - GUI environment specific
            raise RuntimeError("Preview vision wymaga tkinter.") from exc
        if ImageTk is None:
            raise RuntimeError("Preview vision wymaga Pillow.ImageTk.")

        root = tk.Tk()
        root.title("botlab live vision preview")
        root.configure(bg="#111827")

        image_label = tk.Label(root, bg="#111827")
        image_label.pack()

        status_label = tk.Label(
            root,
            bg="#111827",
            fg="#f9fafb",
            justify="left",
            anchor="w",
            font=("Consolas", 10),
        )
        status_label.pack(fill="x", padx=8, pady=8)

        running = {"value": True}

        def stop_preview(event=None):
            running["value"] = False
            root.destroy()

        root.bind("<Escape>", stop_preview)
        root.bind("q", stop_preview)
        root.bind("Q", stop_preview)

        def tick():
            if not running["value"]:
                return
            state, image = self.render_next_frame()
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            status_label.configure(text="\n".join(state.headline_lines))
            root.after(self._settings.live.preview_refresh_interval_ms, tick)

        tick()
        root.mainloop()
        return 0
