# Live Scenes

Ten katalog trzyma profile konkretnych spotow/scen dla live vision.

Kazdy profil sceny to prosty plik JSON z:
- `scene_name`
- `reference_frame_path`
- `spawn_zone_polygon`
- opcjonalnym `reference_point_xy`
- opcjonalnymi `sub_rois`
- opcjonalnymi `exclusion_polygons`

Minimalny przyklad:

```json
{
  "scene_name": "single_spot_scene",
  "reference_frame_path": "../sample_frames/raw/live_spot_scene_1.png",
  "reference_point_xy": [1380, 700],
  "spawn_zone_polygon": [
    [980, 300],
    [1970, 300],
    [2050, 540],
    [2030, 1030],
    [1180, 1040],
    [900, 760],
    [930, 470]
  ],
  "sub_rois": {
    "spawn_focus_roi": [960, 280, 1120, 800]
  }
}
```

Zasada MVP:
- perception nadal analizuje `spawn_roi`,
- ale detections sa dodatkowo oznaczane jako `in_scene_zone`,
- target selection bierze tylko wolne targety wewnatrz `spawn_zone_polygon`,
- detections spoza strefy zostaja w debug artefaktach i metrykach, ale nie sa wybierane do engage.
