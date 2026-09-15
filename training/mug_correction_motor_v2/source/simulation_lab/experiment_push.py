"""Disclosed physical disturbance for training/evaluation harnesses only."""
import numpy as np


def apply_disclosed_bottle_push(model, data, specification):
    data.xfrc_applied[:] = 0.
    if specification is None:
        return False
    start = specification['starts_at_simulation_s']
    if not start <= data.time < start+specification['duration_s']:
        return False
    body = model.body('bottle').id
    force = np.asarray(specification['force_n'], dtype=float)
    point = data.body('bottle').xpos+[0., 0., specification['application_height_above_bottle_origin_m']]
    data.xfrc_applied[body, :3] = force
    data.xfrc_applied[body, 3:] = np.cross(point-data.xipos[body], force)
    return True
