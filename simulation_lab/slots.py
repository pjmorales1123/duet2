"""Slot identities stay fixed; tube identities remain stable when tubes move."""
from __future__ import annotations

import math
import numpy as np

from .scene import TABLE_Z


def tube_slot(data, layout, tube):
    body = data.body(tube["body"])
    expected_z = TABLE_Z+.010+tube["height"]/2
    if body.xmat[8] < math.cos(math.radians(20)) or abs(body.xpos[2]-expected_z) > .008:
        return None
    nearby = [(np.linalg.norm(body.xpos[:2]-np.array(slot["position_m"][:2])), slot) for slot in layout["slots"]]
    distance, slot = min(nearby, key=lambda item: item[0])
    return slot if distance < .014 else None


def blockers(data, layout, slot, exclude=None):
    """Conservative entry-column occupancy, including tilted/displaced tubes."""
    blocked = []
    point = np.array(slot["position_m"][:2])
    for tube in layout["tubes"]:
        if tube["id"] == exclude:
            continue
        body = data.body(tube["body"])
        axis = body.xmat.reshape(3, 3)[:, 2]
        a = body.xpos[:2]-axis[:2]*tube["height"]/2
        b = body.xpos[:2]+axis[:2]*tube["height"]/2
        direction = b-a
        along = np.clip(np.dot(point-a, direction)/max(np.dot(direction, direction), 1e-12), 0, 1)
        distance = np.linalg.norm(point-(a+along*direction))
        low = body.xpos[2]-tube["height"]/2-.011
        high = body.xpos[2]+tube["height"]/2+.011
        if distance < .023 and low < TABLE_Z+.18 and high > TABLE_Z+.008:
            blocked.append(tube["id"])
    return blocked


def slot_state(data, layout):
    occupants = {}
    for tube in layout["tubes"]:
        slot = tube_slot(data, layout, tube)
        if slot:
            occupants.setdefault(slot["id"], []).append(tube["id"])
    return [{**slot, "occupants": occupants.get(slot["id"], []), "blocked_by": blockers(data, layout, slot)}
            for slot in layout["slots"]]
