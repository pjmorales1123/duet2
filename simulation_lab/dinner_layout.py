"""Read and write exact world-coordinate layouts for the dinner scene."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .scene import TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_SIZE_M, TABLE_Z

LAYOUT_VERSION = 1
CANONICAL_LAYOUT_PATH = Path(__file__).parents[1] / "config" / "dinner-layout.json"


def canonical_dinner_layout() -> dict:
    """Load the user-approved final table poses, requiring every dinner item."""
    expected = ("plate", "side_plate", "mug", "glass", "bottle", "fork", "spoon")
    result = load_dinner_layout(
        CANONICAL_LAYOUT_PATH,
        expected,
    )
    missing = set(expected)-set(result['objects'])
    if missing:
        raise ValueError('Canonical dinner layout is missing: '+', '.join(sorted(missing)))
    return result


def layout_document(
    object_poses: Mapping[str, Mapping[str, object]],
    robot_poses: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    """Create a portable JSON document from object and robot world poses."""
    objects = {}
    for object_id, pose in object_poses.items():
        position = [float(value) for value in pose["position_m"]]
        if len(position) != 3:
            raise ValueError(f"{object_id} position_m must contain exactly three values.")
        objects[object_id] = {
            "position_m": position,
            "yaw_rad": float(pose.get("yaw_rad", 0.0)),
        }
    document = {
        "schema_version": LAYOUT_VERSION,
        "scene": "dinner",
        "coordinate_frame": {
            "table_center_m": [TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_Z],
            "table_size_m": list(TABLE_SIZE_M),
            "table_top_z_m": TABLE_Z,
            "position_semantics": "free-body origin in world coordinates",
        },
        "objects": objects,
    }
    if robot_poses is not None:
        document["robots"] = {
            robot_id: {key: value for key, value in pose.items()}
            for robot_id, pose in robot_poses.items()
        }
    return document


def load_dinner_layout(source: str | Path | Mapping | None, object_ids: Sequence[str]) -> dict | None:
    """Load and validate a layout file or mapping for scene construction."""
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Could not read dinner layout: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dinner layout is not valid JSON: {path}") from exc
    elif isinstance(source, Mapping):
        document = dict(source)
    else:
        raise TypeError("dinner_layout must be a JSON path, mapping, or None.")
    if document.get("scene") not in (None, "dinner"):
        raise ValueError("Dinner layout scene must be 'dinner'.")
    objects = document.get("objects")
    if not isinstance(objects, Mapping):
        raise ValueError("Dinner layout must contain an objects mapping.")
    allowed = set(object_ids)
    normalized = {}
    for object_id, pose in objects.items():
        if object_id not in allowed:
            raise ValueError(f"Unknown dinner object in layout: {object_id}")
        if not isinstance(pose, Mapping):
            raise ValueError(f"Layout pose for {object_id} must be an object.")
        position = pose.get("position_m")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 3:
            raise ValueError(f"Layout pose for {object_id} needs position_m [x, y, z].")
        values = [float(value) for value in position]
        yaw = float(pose.get("yaw_rad", 0.0))
        if not all(math.isfinite(value) for value in [*values, yaw]):
            raise ValueError(f"Layout pose for {object_id} must contain finite numbers.")
        normalized[object_id] = {"position_m": values, "yaw_rad": yaw}
    result = {
        "schema_version": int(document.get("schema_version", LAYOUT_VERSION)),
        "coordinate_frame": document.get("coordinate_frame", {}),
        "objects": normalized,
    }
    robots = document.get("robots")
    if robots is not None:
        if not isinstance(robots, Mapping):
            raise ValueError("Dinner layout robots must be an object.")
        result["robots"] = dict(robots)
    return result


def save_dinner_layout(
    path: str | Path,
    object_poses: Mapping[str, Mapping[str, object]],
    robot_poses: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Write a deterministic, scene-consumable dinner layout document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = layout_document(object_poses, robot_poses)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
