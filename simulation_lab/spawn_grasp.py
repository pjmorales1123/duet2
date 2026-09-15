"""Convert an item's live simulator spawn frame into gripper constraints."""
from __future__ import annotations

import numpy as np


def world_grasp_axis(spawn_rotation, local_axis) -> np.ndarray:
    """Rotate a declared object-local grasp direction into world coordinates."""
    axis = np.asarray(spawn_rotation, dtype=float) @ np.asarray(local_axis, dtype=float)
    length = float(np.linalg.norm(axis))
    if length == 0.:
        raise ValueError('A grasp axis must be non-zero.')
    return axis / length
