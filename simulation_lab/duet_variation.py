"""Seeded Duet 2 dinner variation sampled once and recorded with every scene."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DinnerVariation:
    """Physical and display samples that define one reproducible dinner scene."""

    seed: int
    profile: str
    object_offsets_m: dict[str, tuple[float, float]]
    object_yaw_rad: dict[str, float]
    mass_scales: dict[str, float]
    friction_scales: dict[str, float]
    light_multiplier: float
    backdrop_rgba: tuple[float, float, float, float]

    def manifest(self) -> dict:
        """Serialize only primitive values for dataset and evaluation manifests."""
        return asdict(self)


def sample_dinner_variation(seed: int, object_ids: tuple[str, ...], profile: str = "duet_v1") -> DinnerVariation:
    """Create a bounded scene sample without changing object geometry or goals."""
    if profile not in ("duet_v1", "duet_wide_probe_v1", "duet_demo_probe_v1", "duet_micro_v1", "duet_free_v1"):
        raise ValueError("Unknown Duet dinner variation profile.")

    rng = np.random.default_rng(seed ^ 0xD0E72)
    offsets, yaw, mass, friction = {}, {}, {}, {}
    for object_id in object_ids:
        if profile == "duet_micro_v1":
            # Deliberately tight: enough real noise that the policy can't just
            # memorize one exact pixel layout, but narrow enough that 10 seeds
            # is real coverage rather than 10 thin samples spread too wide to
            # learn from (a first same-day VLA checkpoint, not a generalist).
            xy_limit = .004 if object_id not in ("fork", "spoon") else .002
            yaw_limit = .05 if object_id not in ("fork", "spoon") else .035
        elif profile == "duet_demo_probe_v1":
            # Wide enough to be obvious by eye in the overhead frame, not just
            # in the recorded coordinates. Verified per-seed for reachability
            # before use; not the profile used for bulk training collection.
            xy_limit = .065 if object_id not in ("fork", "spoon") else .028
            yaw_limit = .45 if object_id not in ("fork", "spoon") else .30
        elif profile == "duet_wide_probe_v1":
            xy_limit = .035 if object_id not in ("fork", "spoon") else .015
            yaw_limit = .30 if object_id not in ("fork", "spoon") else .18
        else:
            xy_limit = .012 if object_id not in ("fork", "spoon") else .005
            yaw_limit = .16 if object_id not in ("fork", "spoon") else .10
        offsets[object_id] = tuple(float(value) for value in rng.uniform(-xy_limit, xy_limit, 2))
        yaw[object_id] = float(rng.uniform(-yaw_limit, yaw_limit))
        mass[object_id] = float(rng.uniform(.88, 1.12))
        friction[object_id] = float(rng.uniform(.84, 1.16))

    backdrop = (
        float(rng.uniform(.04, .55)),
        float(rng.uniform(.04, .50)),
        float(rng.uniform(.08, .60)),
        1.0,
    )
    return DinnerVariation(
        seed=seed,
        profile=profile,
        object_offsets_m=offsets,
        object_yaw_rad=yaw,
        mass_scales=mass,
        friction_scales=friction,
        light_multiplier=float(rng.uniform(.55, 1.55)),
        backdrop_rgba=backdrop,
    )
