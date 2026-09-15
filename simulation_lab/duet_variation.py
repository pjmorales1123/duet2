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
    if profile != "duet_v1":
        raise ValueError("Unknown Duet dinner variation profile.")

    rng = np.random.default_rng(seed ^ 0xD0E72)
    offsets, yaw, mass, friction = {}, {}, {}, {}
    for object_id in object_ids:
        xy_limit = .012 if object_id not in ("fork", "spoon") else .005
        yaw_limit = .16 if object_id not in ("fork", "spoon") else .10
        offsets[object_id] = tuple(float(value) for value in rng.uniform(-xy_limit, xy_limit, 2))
        yaw[object_id] = float(rng.uniform(-yaw_limit, yaw_limit))
        mass[object_id] = float(rng.uniform(.88, 1.12))
        friction[object_id] = float(rng.uniform(.84, 1.16))

    backdrop = (
        float(rng.uniform(.07, .14)),
        float(rng.uniform(.07, .12)),
        float(rng.uniform(.16, .28)),
        1.0,
    )
    return DinnerVariation(
        seed=seed,
        profile=profile,
        object_offsets_m=offsets,
        object_yaw_rad=yaw,
        mass_scales=mass,
        friction_scales=friction,
        light_multiplier=float(rng.uniform(.86, 1.14)),
        backdrop_rgba=backdrop,
    )
