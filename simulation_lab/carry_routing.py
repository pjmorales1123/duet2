"""Plan conservative planar detours for physically carried dinnerware."""
from __future__ import annotations

import numpy as np


def segment_clear_of_disc(start, end, center, radius: float) -> bool:
    """Return whether a line segment stays outside an inflated circular obstacle."""
    start, end, center = (np.asarray(point, dtype=float) for point in (start, end, center))
    direction = end - start
    length_squared = float(direction @ direction)
    fraction = 0. if length_squared == 0. else float(np.clip((center - start) @ direction / length_squared, 0., 1.))
    return float(np.linalg.norm(start + fraction * direction - center)) >= radius


def plan_disc_detour(start, end, center, clearance: float) -> list[np.ndarray]:
    """Return the shortest validated two-corner route around an inflated obstacle."""
    start, end, center = (np.asarray(point, dtype=float) for point in (start, end, center))
    if segment_clear_of_disc(start, end, center, clearance):
        return [start, end]
    direction = end - start
    distance = float(np.linalg.norm(direction))
    if distance == 0.:
        raise ValueError('A detour needs distinct start and end points.')
    forward = direction / distance
    perpendicular = np.array([-forward[1], forward[0]])
    candidates = []
    for expansion in (1.25, 1.5, 2., 3.):
        radius = clearance * expansion
        for sign in (-1., 1.):
            side = perpendicular * sign
            route = [start, center - forward * radius + side * radius, center + forward * radius + side * radius, end]
            if all(segment_clear_of_disc(a, b, center, clearance) for a, b in zip(route, route[1:])):
                candidates.append(route)
        if candidates:
            break
    if not candidates:
        raise ValueError('No planar detour clears the obstacle.')
    return min(candidates, key=lambda route: sum(float(np.linalg.norm(b-a)) for a, b in zip(route, route[1:])))
