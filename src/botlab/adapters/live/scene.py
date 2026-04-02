from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class SceneProfile:
    scene_name: str
    reference_frame_path: Path | None
    spawn_zone_polygon: tuple[tuple[float, float], ...]
    reference_point_xy: tuple[float, float] | None = None
    notes: str | None = None
    sub_rois: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    exclusion_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()

    def contains_point(self, point_xy: tuple[float, float]) -> bool:
        if not self.spawn_zone_polygon:
            return True
        if not point_in_polygon(point_xy=point_xy, polygon=self.spawn_zone_polygon):
            return False
        for exclusion_polygon in self.exclusion_polygons:
            if point_in_polygon(point_xy=point_xy, polygon=exclusion_polygon):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "reference_frame_path": None
            if self.reference_frame_path is None
            else str(self.reference_frame_path),
            "spawn_zone_polygon": [
                [point_x, point_y] for point_x, point_y in self.spawn_zone_polygon
            ],
            "reference_point_xy": None
            if self.reference_point_xy is None
            else [self.reference_point_xy[0], self.reference_point_xy[1]],
            "notes": self.notes,
            "sub_rois": {
                roi_name: list(roi_value) for roi_name, roi_value in self.sub_rois.items()
            },
            "exclusion_polygons": [
                [[point_x, point_y] for point_x, point_y in polygon]
                for polygon in self.exclusion_polygons
            ],
        }


class SceneProfileLoader:
    def __init__(self, scene_profile_path: Path | None) -> None:
        self._scene_profile_path = scene_profile_path
        self._cached_profile: SceneProfile | None = None

    @property
    def scene_profile_path(self) -> Path | None:
        return self._scene_profile_path

    def load(self) -> SceneProfile | None:
        if self._scene_profile_path is None:
            return None
        if self._cached_profile is not None:
            return self._cached_profile
        path = self._scene_profile_path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Profil sceny {path} musi byc mapa JSON.")

        raw_scene_name = payload.get("scene_name")
        if not isinstance(raw_scene_name, str) or not raw_scene_name.strip():
            raise ValueError(f"Profil sceny {path} musi zawierac niepuste pole 'scene_name'.")

        spawn_zone_polygon = _parse_polygon(
            payload.get("spawn_zone_polygon"),
            field_name="spawn_zone_polygon",
            source_path=path,
        )
        reference_frame_path = _parse_optional_path(
            payload.get("reference_frame_path"),
            source_path=path,
        )
        reference_point_xy = _parse_optional_point(payload.get("reference_point_xy"))
        sub_rois = _parse_sub_rois(payload.get("sub_rois"), source_path=path)
        exclusion_polygons = _parse_exclusion_polygons(
            payload.get("exclusion_polygons"),
            source_path=path,
        )
        notes = payload.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(f"Pole 'notes' w profilu sceny {path} musi byc napisem albo null.")

        self._cached_profile = SceneProfile(
            scene_name=raw_scene_name.strip(),
            reference_frame_path=reference_frame_path,
            spawn_zone_polygon=spawn_zone_polygon,
            reference_point_xy=reference_point_xy,
            notes=notes,
            sub_rois=sub_rois,
            exclusion_polygons=exclusion_polygons,
        )
        return self._cached_profile


def point_in_polygon(
    *,
    point_xy: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    if len(polygon) < 3:
        return False
    point_x = float(point_xy[0])
    point_y = float(point_xy[1])

    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if _point_on_segment(
            point_xy=(point_x, point_y),
            segment_start_xy=(previous_x, previous_y),
            segment_end_xy=(current_x, current_y),
        ):
            return True
        intersects = ((current_y > point_y) != (previous_y > point_y)) and (
            point_x
            < ((previous_x - current_x) * (point_y - current_y) / (previous_y - current_y))
            + current_x
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_segment(
    *,
    point_xy: tuple[float, float],
    segment_start_xy: tuple[float, float],
    segment_end_xy: tuple[float, float],
    epsilon: float = 1e-6,
) -> bool:
    point_x, point_y = point_xy
    start_x, start_y = segment_start_xy
    end_x, end_y = segment_end_xy
    cross_product = ((point_y - start_y) * (end_x - start_x)) - (
        (point_x - start_x) * (end_y - start_y)
    )
    if abs(cross_product) > epsilon:
        return False
    dot_product = ((point_x - start_x) * (end_x - start_x)) + (
        (point_y - start_y) * (end_y - start_y)
    )
    if dot_product < 0.0:
        return False
    squared_length = ((end_x - start_x) ** 2) + ((end_y - start_y) ** 2)
    if dot_product > squared_length:
        return False
    return True


def _parse_optional_path(raw_value: object, *, source_path: Path) -> Path | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"Pole 'reference_frame_path' w profilu sceny {source_path} musi byc napisem albo null."
        )
    candidate = Path(raw_value.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (source_path.parent / candidate).resolve()


def _parse_optional_point(raw_value: object) -> tuple[float, float] | None:
    if raw_value is None:
        return None
    if (
        not isinstance(raw_value, (list, tuple))
        or len(raw_value) != 2
        or not all(isinstance(item, (int, float)) for item in raw_value)
    ):
        raise ValueError("Pole 'reference_point_xy' musi byc [x, y] albo null.")
    return (float(raw_value[0]), float(raw_value[1]))


def _parse_polygon(
    raw_value: object,
    *,
    field_name: str,
    source_path: Path,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw_value, list) or len(raw_value) < 3:
        raise ValueError(
            f"Pole '{field_name}' w profilu sceny {source_path} musi byc lista co najmniej 3 punktow."
        )
    polygon_points: list[tuple[float, float]] = []
    for point_index, raw_point in enumerate(raw_value, start=1):
        if (
            not isinstance(raw_point, (list, tuple))
            or len(raw_point) != 2
            or not all(isinstance(item, (int, float)) for item in raw_point)
        ):
            raise ValueError(
                f"Punkt {point_index} w '{field_name}' w profilu sceny {source_path} musi byc [x, y]."
            )
        polygon_points.append((float(raw_point[0]), float(raw_point[1])))
    return tuple(polygon_points)


def _parse_sub_rois(
    raw_value: object,
    *,
    source_path: Path,
) -> dict[str, tuple[int, int, int, int]]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"Pole 'sub_rois' w profilu sceny {source_path} musi byc mapa.")
    parsed: dict[str, tuple[int, int, int, int]] = {}
    for roi_name, roi_value in raw_value.items():
        if not isinstance(roi_name, str) or not roi_name.strip():
            raise ValueError(f"Klucze w 'sub_rois' w profilu sceny {source_path} musza byc napisami.")
        if (
            not isinstance(roi_value, (list, tuple))
            or len(roi_value) != 4
            or not all(isinstance(item, int) for item in roi_value)
        ):
            raise ValueError(
                f"ROI '{roi_name}' w profilu sceny {source_path} musi miec format [x, y, width, height]."
            )
        parsed[roi_name.strip()] = (
            int(roi_value[0]),
            int(roi_value[1]),
            int(roi_value[2]),
            int(roi_value[3]),
        )
    return parsed


def _parse_exclusion_polygons(
    raw_value: object,
    *,
    source_path: Path,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        raise ValueError(
            f"Pole 'exclusion_polygons' w profilu sceny {source_path} musi byc lista poligonow."
        )
    exclusion_polygons: list[tuple[tuple[float, float], ...]] = []
    for polygon_index, raw_polygon in enumerate(raw_value, start=1):
        exclusion_polygons.append(
            _parse_polygon(
                raw_polygon,
                field_name=f"exclusion_polygons[{polygon_index}]",
                source_path=source_path,
            )
        )
    return tuple(exclusion_polygons)
