from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from botlab.adapters.live.capture import create_capture
from botlab.adapters.live.models import LiveEngageResult, LiveFrame, LiveSessionState
from botlab.adapters.live.perception import PerceptionAnalyzer, PerceptionFrameResult
from botlab.adapters.live.runner import LiveRunner
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


class LivePreviewSnapshotWriter:
    def __init__(self, *, debug_directory: Path) -> None:
        self._output_directory = Path(debug_directory) / "preview_snapshot"
        self._image_path = self._output_directory / "latest_preview.png"
        self._state_path = self._output_directory / "latest_preview.json"

    @property
    def image_path(self) -> Path:
        return self._image_path

    @property
    def state_path(self) -> Path:
        return self._state_path

    def save(self, *, state: LiveVisionPreviewState, image) -> tuple[Path, Path]:
        if Image is None:
            raise RuntimeError("Snapshot preview wymaga Pillow.")
        self._output_directory.mkdir(parents=True, exist_ok=True)
        image.save(self._image_path)
        payload = {
            "frame_source": state.frame_source,
            "frame_width": state.frame_width,
            "frame_height": state.frame_height,
            "capture_reliability": state.capture_reliability,
            "capture_latency_ms": state.capture_latency_ms,
            "preview_background_bypass": state.preview_background_bypass,
            "window_guard_block_reason": state.window_guard_block_reason,
            "capture_mode_used_for_preview": state.capture_mode_used_for_preview,
            "selected_target_id": state.selected_target_id,
            "candidate_count": state.candidate_count,
            "free_target_count": state.free_target_count,
            "occupied_target_count": state.occupied_target_count,
            "out_of_zone_target_count": state.out_of_zone_target_count,
            "detection_latency_ms": state.detection_latency_ms,
            "selection_latency_ms": state.selection_latency_ms,
            "total_reaction_latency_ms": state.total_reaction_latency_ms,
            "render_latency_ms": state.render_latency_ms,
            "dropped_or_skipped_frames": state.dropped_or_skipped_frames,
            "preview_mode": state.preview_mode,
            "headline_lines": list(state.headline_lines),
            "image_path": str(self._image_path),
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self._image_path, self._state_path


def _clone_preview_payload(
    payload: tuple[LiveVisionPreviewState, Any],
) -> tuple[LiveVisionPreviewState, Any]:
    state, image = payload
    return state, image.copy() if hasattr(image, "copy") else image


def _resolve_snapshot_payload(
    *,
    latest_payload: tuple[LiveVisionPreviewState, Any] | None,
) -> tuple[LiveVisionPreviewState, Any]:
    if latest_payload is not None:
        return _clone_preview_payload(latest_payload)
    raise RuntimeError("Brak gotowej klatki do zapisu. Poczekaj na pierwsza wyrenderowana klatke.")


@dataclass(slots=True, frozen=True)
class LiveVisionPreviewState:
    frame_source: str
    frame_width: int
    frame_height: int
    capture_reliability: str | None
    capture_latency_ms: float
    preview_background_bypass: bool
    window_guard_block_reason: str | None
    capture_mode_used_for_preview: str
    selected_target_id: str | None
    candidate_count: int
    free_target_count: int
    occupied_target_count: int
    out_of_zone_target_count: int
    detection_latency_ms: float
    selection_latency_ms: float
    total_reaction_latency_ms: float
    render_latency_ms: float
    dropped_or_skipped_frames: int
    preview_mode: str
    headline_lines: tuple[str, ...]


def build_live_preview_state(
    *,
    frame: LiveFrame,
    result: PerceptionFrameResult,
    engage_result: LiveEngageResult | None = None,
    render_latency_ms: float = 0.0,
    dropped_or_skipped_frames: int = 0,
    preview_mode: str = "standard",
    capture_mode_used_for_preview: str = "default",
) -> LiveVisionPreviewState:
    selected_target = result.selected_target
    capture_reliability = frame.metadata.get("capture_reliability")
    if not isinstance(capture_reliability, str) or not capture_reliability:
        capture_reliability = None
    capture_latency_ms = float(frame.metadata.get("capture_latency_ms", 0.0) or 0.0)
    preview_background_bypass = bool(frame.metadata.get("preview_background_bypass"))
    window_guard = frame.metadata.get("window_guard")
    window_guard_block_reason = None
    if isinstance(window_guard, dict):
        block_reason = window_guard.get("block_reason")
        if isinstance(block_reason, str) and block_reason.strip():
            window_guard_block_reason = block_reason
    headline_lines = [
        f"frame_source={frame.source}",
        f"scene={result.scene_name or 'none'}",
        f"targets={len(result.detections)} free={len(result.selectable_detections)} occupied={len(result.occupied_detections)} out_of_zone={len(result.out_of_zone_detections)}",
        f"selected={result.selected_target_id or 'None'}",
        f"capture_ms={capture_latency_ms:.1f} detection_ms={result.timings.detection_latency_ms:.1f} render_ms={render_latency_ms:.1f} selection={result.timings.selection_latency_ms:.1f} reaction={result.timings.total_reaction_latency_ms:.1f}ms",
        f"preview_mode={preview_mode} dropped_or_skipped_frames={dropped_or_skipped_frames}",
        f"candidate_hits={result.candidate_hit_count} merged={result.merged_hit_count}",
        f"capture_mode_used_for_preview={capture_mode_used_for_preview}",
        (
            "selected_info="
            f"{selected_target.target_id} dist={selected_target.distance:.1f} conf={selected_target.confidence:.2f}"
            if selected_target is not None
            else "selected_info=None"
        ),
    ]
    headline_lines.append(f"capture_reliability={capture_reliability or 'unknown'}")
    headline_lines.append(f"preview_background_bypass={'true' if preview_background_bypass else 'false'}")
    headline_lines.append(
        f"window_guard.block_reason={window_guard_block_reason or 'none'}"
    )
    calibration_warning = result.scene_calibration.get("warning")
    if isinstance(calibration_warning, str) and calibration_warning.strip():
        headline_lines.append(f"scene_calibration_warning={calibration_warning}")
    if engage_result is not None:
        headline_lines.append(
            "engage="
            f"{engage_result.outcome.value} "
            f"click={None if engage_result.click_screen_xy is None else list(engage_result.click_screen_xy)} "
            f"verify={engage_result.verification_latency_ms if engage_result.verification_latency_ms is not None else 0.0:.1f}ms"
        )
    return LiveVisionPreviewState(
        frame_source=frame.source,
        frame_width=frame.width,
        frame_height=frame.height,
        capture_reliability=capture_reliability,
        capture_latency_ms=capture_latency_ms,
        preview_background_bypass=preview_background_bypass,
        window_guard_block_reason=window_guard_block_reason,
        capture_mode_used_for_preview=capture_mode_used_for_preview,
        selected_target_id=result.selected_target_id,
        candidate_count=len(result.detections),
        free_target_count=len(result.selectable_detections),
        occupied_target_count=len(result.occupied_detections),
        out_of_zone_target_count=len(result.out_of_zone_detections),
        detection_latency_ms=result.timings.detection_latency_ms,
        selection_latency_ms=result.timings.selection_latency_ms,
        total_reaction_latency_ms=result.timings.total_reaction_latency_ms,
        render_latency_ms=render_latency_ms,
        dropped_or_skipped_frames=dropped_or_skipped_frames,
        preview_mode=preview_mode,
        headline_lines=tuple(headline_lines),
    )


class LiveVisionPreviewRenderer:
    def __init__(
        self,
        *,
        max_width_px: int,
        max_height_px: int,
        render_aux_boxes: bool = True,
        crop_to_spawn_roi: bool = False,
        crop_padding_px: int = 0,
    ) -> None:
        self._max_width_px = max_width_px
        self._max_height_px = max_height_px
        self._render_aux_boxes = render_aux_boxes
        self._crop_to_spawn_roi = crop_to_spawn_roi
        self._crop_padding_px = max(0, int(crop_padding_px))

    def render(
        self,
        *,
        frame: LiveFrame,
        result: PerceptionFrameResult,
        state: LiveVisionPreviewState,
        engage_result: LiveEngageResult | None = None,
        verification_result: PerceptionFrameResult | None = None,
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
        if self._render_aux_boxes and result.scene_zone_polygon:
            draw.polygon(result.scene_zone_polygon, outline=(34, 197, 94), width=3)
            draw.text(
                (int(result.scene_zone_polygon[0][0]), max(4, int(result.scene_zone_polygon[0][1]) - 18)),
                result.scene_name or "scene_zone",
                fill=(134, 239, 172),
                font=font,
            )

        if self._render_aux_boxes:
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
            in_scene_zone = bool(detection.metadata.get("in_scene_zone", True))
            color = (239, 68, 68) if detection.occupied else (34, 197, 94)
            if not in_scene_zone:
                color = (156, 163, 175)
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
            if self._render_aux_boxes:
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
            if not in_scene_zone:
                label = f"{label}/out_of_zone"
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

        if verification_result is not None:
            for detection in verification_result.detections:
                bbox = detection.bbox or (
                    max(0, detection.screen_x - 18),
                    max(0, detection.screen_y - 24),
                    36,
                    48,
                )
                color = (249, 115, 22) if detection.occupied else (56, 189, 248)
                draw.rectangle(
                    (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    outline=color,
                    width=2,
                )

        if engage_result is not None and engage_result.click_screen_xy is not None:
            click_x, click_y = engage_result.click_screen_xy
            draw.line((click_x - 12, click_y, click_x + 12, click_y), fill=(253, 224, 71), width=3)
            draw.line((click_x, click_y - 12, click_x, click_y + 12), fill=(253, 224, 71), width=3)
            draw.text(
                (click_x + 16, max(8, click_y - 12)),
                f"engage:{engage_result.outcome.value}",
                fill=(253, 224, 71),
                font=font,
            )

        self._draw_headline_box(draw=draw, font=font, state=state)
        canvas = self._crop_canvas_to_spawn_roi(canvas=canvas, result=result)
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

    def _crop_canvas_to_spawn_roi(self, *, canvas, result: PerceptionFrameResult):
        if not self._crop_to_spawn_roi:
            return canvas
        roi = result.roi
        left = max(0, int(roi["x"]) - self._crop_padding_px)
        top = max(0, int(roi["y"]) - self._crop_padding_px)
        right = min(canvas.size[0], int(roi["x"]) + int(roi["width"]) + self._crop_padding_px)
        bottom = min(canvas.size[1], int(roi["y"]) + int(roi["height"]) + self._crop_padding_px)
        if right <= left or bottom <= top:
            return canvas
        return canvas.crop((left, top, right, bottom))


def _should_reuse_previous_preview_payload(frame: LiveFrame) -> bool:
    capture_reliability = frame.metadata.get("capture_reliability")
    if capture_reliability == "blocked":
        return True
    window_guard = frame.metadata.get("window_guard", {})
    if isinstance(window_guard, dict):
        block_reason = str(window_guard.get("block_reason") or "")
        if block_reason == "window_minimized":
            return True
    if frame.width <= 200 and frame.height <= 80:
        return True
    return False


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
            render_aux_boxes=settings.live.preview_render_aux_boxes,
            crop_to_spawn_roi=settings.live.preview_crop_to_spawn_roi,
            crop_padding_px=settings.live.preview_crop_padding_px,
        )
        self._snapshot_writer = LivePreviewSnapshotWriter(debug_directory=settings.live.debug_directory)
        self._session_state = LiveSessionState()
        self._tick_index = 0
        self._preview_loop_index = 0
        self._skipped_frame_count = 0
        self._last_render_payload: tuple[LiveVisionPreviewState, Any] | None = None

    def render_next_frame(self):
        self._preview_loop_index += 1
        analyze_every = max(1, self._settings.live.preview_analyze_every_nth_frame)
        if (
            self._settings.live.preview_fast_mode
            and self._last_render_payload is not None
            and (self._preview_loop_index % analyze_every) != 0
        ):
            self._skipped_frame_count += 1
            return self._last_render_payload

        capture_started = time.perf_counter()
        frame = self._capture.capture_frame(
            cycle_id=self._tick_index + 1,
            phase="observation",
            default_ts=float(self._clock()),
            session_state=self._session_state,
            allow_background_capture=True,
        )
        capture_elapsed_ms = max(0.0, (time.perf_counter() - capture_started) * 1000.0)
        frame = replace(
            frame,
            metadata={
                **frame.metadata,
                "capture_latency_ms": capture_elapsed_ms,
            },
        )
        if _should_reuse_previous_preview_payload(frame) and self._last_render_payload is not None:
            self._skipped_frame_count += 1
            reused_state, reused_image = self._last_render_payload
            reused_lines = tuple(
                line
                for line in reused_state.headline_lines
                if not line.startswith("preview_reused_due_to=")
            ) + (
                f"preview_reused_due_to={frame.metadata.get('window_guard', {}).get('block_reason', 'invalid_frame')}",
            )
            reused_state = replace(
                reused_state,
                dropped_or_skipped_frames=self._skipped_frame_count,
                headline_lines=reused_lines,
            )
            self._last_render_payload = (
                reused_state,
                reused_image.copy() if hasattr(reused_image, "copy") else reused_image,
            )
            return self._last_render_payload
        analyze_started = time.perf_counter()
        result = self._analyzer.analyze_frame(
            frame,
            cycle_id=self._tick_index + 1,
            phase="preview",
        )
        analyze_elapsed_ms = max(0.0, (time.perf_counter() - analyze_started) * 1000.0)
        render_started = time.perf_counter()
        preview_mode = "fast" if self._settings.live.preview_fast_mode else "standard"
        state = build_live_preview_state(
            frame=frame,
            result=result,
            render_latency_ms=0.0,
            dropped_or_skipped_frames=self._skipped_frame_count,
            preview_mode=preview_mode,
            capture_mode_used_for_preview="preview_bypass",
        )
        image = self._renderer.render(frame=frame, result=result, state=state)
        render_elapsed_ms = max(0.0, (time.perf_counter() - render_started) * 1000.0)
        state = build_live_preview_state(
            frame=frame,
            result=result,
            render_latency_ms=render_elapsed_ms,
            dropped_or_skipped_frames=self._skipped_frame_count,
            preview_mode=preview_mode,
            capture_mode_used_for_preview="preview_bypass",
        )
        self._tick_index += 1
        refresh_budget_ms = float(self._settings.live.preview_refresh_interval_ms)
        if analyze_elapsed_ms > refresh_budget_ms:
            self._logger.warning(
                "live_preview detection exceeded refresh budget capture_ms=%.1f detection_ms=%.1f budget_ms=%.1f skipped=%s",
                capture_elapsed_ms,
                analyze_elapsed_ms,
                refresh_budget_ms,
                self._skipped_frame_count,
            )
        self._logger.info(
            "live_preview frame=%s targets=%s free=%s occupied=%s selected=%s capture_ms=%.1f detection_ms=%.1f render_ms=%.1f skipped=%s mode=%s reaction_ms=%.3f",
            self._tick_index,
            state.candidate_count,
            state.free_target_count,
            state.occupied_target_count,
            state.selected_target_id,
            capture_elapsed_ms,
            analyze_elapsed_ms,
            render_elapsed_ms,
            self._skipped_frame_count,
            preview_mode,
            state.total_reaction_latency_ms,
        )
        self._last_render_payload = (state, image)
        return self._last_render_payload

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

        latest_snapshot: dict[str, tuple[LiveVisionPreviewState, Any] | None] = {"payload": None}

        image_label = tk.Label(root, bg="#111827")
        image_label.pack()

        controls_frame = tk.Frame(root, bg="#111827")
        controls_frame.pack(fill="x", padx=8, pady=(8, 0))

        snapshot_status = tk.Label(
            controls_frame,
            bg="#111827",
            fg="#93c5fd",
            justify="left",
            anchor="w",
            font=("Consolas", 9),
            text="Snapshot: czekam na pierwsza gotowa klatke",
        )
        snapshot_status.pack(side="left", fill="x", expand=True)

        def save_snapshot(event=None):
            try:
                state, snapshot_image = _resolve_snapshot_payload(
                    latest_payload=latest_snapshot["payload"],
                )
                latest_snapshot["payload"] = (state, snapshot_image.copy() if hasattr(snapshot_image, "copy") else snapshot_image)
                image_path, state_path = self._snapshot_writer.save(state=state, image=snapshot_image)
                snapshot_status.configure(text=f"Snapshot zapisany: {image_path.name} | {state_path.name}")
            except Exception as exc:  # pragma: no cover - GUI path
                snapshot_status.configure(text=f"Snapshot blad: {exc}")

        save_button = tk.Button(
            controls_frame,
            text="Zapisz klatke (S)",
            command=save_snapshot,
        )
        save_button.pack(side="right")

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
        result_queue: queue.Queue[tuple[LiveVisionPreviewState, Any] | Exception] = queue.Queue(maxsize=1)
        worker_state = {"started": False}

        def stop_preview(event=None):
            running["value"] = False
            root.destroy()

        root.bind("<Escape>", stop_preview)
        root.bind("q", stop_preview)
        root.bind("Q", stop_preview)
        root.bind("s", save_snapshot)
        root.bind("S", save_snapshot)

        def worker() -> None:
            interval_s = max(0.01, self._settings.live.preview_refresh_interval_ms / 1000.0)
            while running["value"]:
                started_at = time.perf_counter()
                try:
                    payload: tuple[LiveVisionPreviewState, Any] | Exception = self.render_next_frame()
                except Exception as exc:  # pragma: no cover - runtime specific
                    payload = exc
                try:
                    while True:
                        result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put_nowait(payload)
                except queue.Full:
                    pass
                elapsed_s = time.perf_counter() - started_at
                remaining_s = interval_s - elapsed_s
                if remaining_s > 0:
                    time.sleep(remaining_s)

        def poll_results():
            if not running["value"]:
                return
            try:
                while True:
                    payload = result_queue.get_nowait()
                    if isinstance(payload, Exception):
                        running["value"] = False
                        raise payload
                    state, image = payload
                    latest_snapshot["payload"] = (state, image.copy() if hasattr(image, "copy") else image)
                    photo = ImageTk.PhotoImage(image)
                    image_label.configure(image=photo)
                    image_label.image = photo
                    status_label.configure(text="\n".join(state.headline_lines))
            except queue.Empty:
                pass
            except Exception as exc:  # pragma: no cover - GUI environment specific
                running["value"] = False
                root.destroy()
                raise RuntimeError("Live preview zatrzymal sie podczas analizy klatki.") from exc
            root.after(30, poll_results)

        status_label.configure(text="Trwa analiza pierwszej klatki...")
        root.update_idletasks()
        try:
            initial_state, initial_image = self.render_next_frame()
            latest_snapshot["payload"] = (
                initial_state,
                initial_image.copy() if hasattr(initial_image, "copy") else initial_image,
            )
            initial_photo = ImageTk.PhotoImage(initial_image)
            image_label.configure(image=initial_photo)
            image_label.image = initial_photo
            status_label.configure(text="\n".join(initial_state.headline_lines))
            snapshot_status.configure(text=f"Snapshot gotowy: {self._snapshot_writer.image_path.name}")
        except Exception as exc:  # pragma: no cover - GUI environment specific
            status_label.configure(text=f"Blad pierwszej klatki: {exc}")

        if not worker_state["started"]:
            threading.Thread(target=worker, name="botlab-live-preview", daemon=True).start()
            worker_state["started"] = True
        poll_results()
        root.mainloop()
        return 0


class LiveEngageObserve:
    def __init__(
        self,
        *,
        settings: Settings,
        logger_name: str = "botlab.live.engage.observe",
        enable_console: bool = True,
    ) -> None:
        observe_settings = replace(
            settings,
            live=replace(settings.live, dry_run=False),
        )
        self._safe_dry_run_enabled = True
        self._settings = observe_settings
        self._logger = logging.getLogger(logger_name)
        self._runner = LiveRunner.from_settings(
            observe_settings,
            logger_name=logger_name,
            enable_console=enable_console,
            force_input_dry_run=True,
            force_background_capture=True,
        )
        self._renderer = LiveVisionPreviewRenderer(
            max_width_px=observe_settings.live.preview_max_width_px,
            max_height_px=observe_settings.live.preview_max_height_px,
            render_aux_boxes=observe_settings.live.preview_render_aux_boxes,
            crop_to_spawn_roi=observe_settings.live.preview_crop_to_spawn_roi,
            crop_padding_px=observe_settings.live.preview_crop_padding_px,
        )
        self._snapshot_writer = LivePreviewSnapshotWriter(debug_directory=observe_settings.live.debug_directory)
        self._logger.warning(
            "live_engage_observe forced dry_run input for safety while preserving real live capture."
        )
        self._last_render_payload: tuple[LiveVisionPreviewState, Any] | None = None

    @property
    def safe_dry_run_enabled(self) -> bool:
        return self._safe_dry_run_enabled

    def render_next_attempt(self):
        report = self._runner.run_engage_attempts(1)
        result = report.results[0]
        observation_result = self._runner.runtime.perception_result(
            cycle_id=result.cycle_id,
            phase="observation",
        )
        if observation_result is None:
            raise RuntimeError("Brak observation perception result dla engage observe.")
        verification_result = self._runner.runtime.perception_result(
            cycle_id=result.cycle_id,
            phase="engage_verify",
        )
        frame = self._runner.runtime.frame_result(
            cycle_id=result.cycle_id,
            phase="observation",
            capture_mode="default",
        )
        capture_mode_used_for_preview = "default"
        if frame is None:
            frame = self._runner.runtime.frame_result(
                cycle_id=result.cycle_id,
                phase="observation",
                capture_mode="preview_bypass",
            )
            if frame is not None:
                capture_mode_used_for_preview = "preview_bypass"
        if frame is None:
            frame = self._runner.runtime.capture_frame(
                cycle_id=result.cycle_id,
                phase="observation",
                default_ts=observation_result.timings.frame_captured_ts,
                allow_background_capture=True,
            )
            capture_mode_used_for_preview = "preview_bypass_fallback"
        if _should_reuse_previous_preview_payload(frame) and self._last_render_payload is not None:
            reused_state, reused_image = self._last_render_payload
            reused_lines = tuple(
                line
                for line in reused_state.headline_lines
                if not line.startswith("preview_reused_due_to=")
            ) + (
                f"preview_reused_due_to={frame.metadata.get('window_guard', {}).get('block_reason', 'invalid_frame')}",
            )
            reused_state = replace(reused_state, headline_lines=reused_lines)
            self._last_render_payload = (
                reused_state,
                reused_image.copy() if hasattr(reused_image, "copy") else reused_image,
            )
            return self._last_render_payload
        state = build_live_preview_state(
            frame=frame,
            result=observation_result,
            engage_result=result,
            preview_mode="observe_fast" if self._settings.live.preview_fast_mode else "observe_standard",
            capture_mode_used_for_preview=capture_mode_used_for_preview,
        )
        if self._safe_dry_run_enabled:
            state = replace(state, headline_lines=state.headline_lines + ("input_mode=dry_run_safe",))
        image = self._renderer.render(
            frame=frame,
            result=observation_result,
            state=state,
            engage_result=result,
            verification_result=verification_result,
        )
        self._logger.info(
            "live_engage_observe cycle_id=%s outcome=%s selected=%s click=%s",
            result.cycle_id,
            result.outcome.value,
            result.selected_target_id,
            result.click_screen_xy,
        )
        self._last_render_payload = (
            state,
            image.copy() if hasattr(image, "copy") else image,
        )
        return state, image

    def run(self) -> int:
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - GUI environment specific
            raise RuntimeError("Live engage observe wymaga tkinter.") from exc
        if ImageTk is None:
            raise RuntimeError("Live engage observe wymaga Pillow.ImageTk.")

        root = tk.Tk()
        root.title("botlab live engage observe")
        root.configure(bg="#111827")

        latest_snapshot: dict[str, tuple[LiveVisionPreviewState, Any] | None] = {"payload": None}

        image_label = tk.Label(root, bg="#111827")
        image_label.pack()

        controls_frame = tk.Frame(root, bg="#111827")
        controls_frame.pack(fill="x", padx=8, pady=(8, 0))

        snapshot_status = tk.Label(
            controls_frame,
            bg="#111827",
            fg="#93c5fd",
            justify="left",
            anchor="w",
            font=("Consolas", 9),
            text="Snapshot: czekam na pierwsza gotowa klatke",
        )
        snapshot_status.pack(side="left", fill="x", expand=True)

        def save_snapshot(event=None):
            try:
                state, snapshot_image = _resolve_snapshot_payload(
                    latest_payload=latest_snapshot["payload"],
                )
                latest_snapshot["payload"] = (state, snapshot_image.copy() if hasattr(snapshot_image, "copy") else snapshot_image)
                image_path, state_path = self._snapshot_writer.save(state=state, image=snapshot_image)
                snapshot_status.configure(text=f"Snapshot zapisany: {image_path.name} | {state_path.name}")
            except Exception as exc:  # pragma: no cover - GUI path
                snapshot_status.configure(text=f"Snapshot blad: {exc}")

        save_button = tk.Button(
            controls_frame,
            text="Zapisz klatke (S)",
            command=save_snapshot,
        )
        save_button.pack(side="right")

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
        result_queue: queue.Queue[tuple[LiveVisionPreviewState, Any] | Exception] = queue.Queue(maxsize=1)
        worker_state = {"started": False}

        def stop_preview(event=None):
            running["value"] = False
            root.destroy()

        root.bind("<Escape>", stop_preview)
        root.bind("q", stop_preview)
        root.bind("Q", stop_preview)
        root.bind("s", save_snapshot)
        root.bind("S", save_snapshot)

        def worker() -> None:
            interval_s = max(0.01, self._settings.live.preview_refresh_interval_ms / 1000.0)
            while running["value"]:
                started_at = time.perf_counter()
                try:
                    payload: tuple[LiveVisionPreviewState, Any] | Exception = self.render_next_attempt()
                except Exception as exc:  # pragma: no cover - runtime specific
                    payload = exc
                try:
                    while True:
                        result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put_nowait(payload)
                except queue.Full:
                    pass
                elapsed_s = time.perf_counter() - started_at
                remaining_s = interval_s - elapsed_s
                if remaining_s > 0:
                    time.sleep(remaining_s)

        def poll_results():
            if not running["value"]:
                return
            try:
                while True:
                    payload = result_queue.get_nowait()
                    if isinstance(payload, Exception):
                        running["value"] = False
                        raise payload
                    state, image = payload
                    latest_snapshot["payload"] = (state, image.copy() if hasattr(image, "copy") else image)
                    photo = ImageTk.PhotoImage(image)
                    image_label.configure(image=photo)
                    image_label.image = photo
                    status_label.configure(text="\n".join(state.headline_lines))
            except queue.Empty:
                pass
            except Exception as exc:  # pragma: no cover - GUI environment specific
                running["value"] = False
                root.destroy()
                raise RuntimeError("Live engage observe zatrzymal sie podczas analizy proby.") from exc
            root.after(30, poll_results)

        status_label.configure(text="Trwa analiza pierwszej proby...")
        root.update_idletasks()
        try:
            initial_state, initial_image = self.render_next_attempt()
            latest_snapshot["payload"] = (
                initial_state,
                initial_image.copy() if hasattr(initial_image, "copy") else initial_image,
            )
            initial_photo = ImageTk.PhotoImage(initial_image)
            image_label.configure(image=initial_photo)
            image_label.image = initial_photo
            status_label.configure(text="\n".join(initial_state.headline_lines))
            snapshot_status.configure(text=f"Snapshot gotowy: {self._snapshot_writer.image_path.name}")
        except Exception as exc:  # pragma: no cover - GUI environment specific
            status_label.configure(text=f"Blad pierwszej proby: {exc}")

        if not worker_state["started"]:
            threading.Thread(target=worker, name="botlab-live-engage-observe", daemon=True).start()
            worker_state["started"] = True
        poll_results()
        root.mainloop()
        return 0
