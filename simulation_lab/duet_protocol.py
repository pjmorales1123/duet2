"""Duet 2 seed partitions and manifests for reproducible learning experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable


SCHEMA = "duet-2.dinner-protocol.v1"
TRAINING_SEEDS = tuple(range(1000, 1100))
VALIDATION_SEEDS = tuple(range(2000, 2020))
EVALUATION_SEEDS = tuple(range(3000, 3010))


@dataclass(frozen=True)
class SeedPartition:
    """A declared set of scenes with one permitted purpose."""

    name: str
    seeds: tuple[int, ...]
    purpose: str


PARTITIONS = (
    SeedPartition("training", TRAINING_SEEDS, "Generate demonstrations and fit model parameters."),
    SeedPartition("validation", VALIDATION_SEEDS, "Select a candidate without changing its weights."),
    SeedPartition("evaluation", EVALUATION_SEEDS, "Final held-out reporting and video only."),
)


def partition_for(seed: int) -> SeedPartition:
    """Return the only permitted partition for a declared Duet 2 seed."""
    for partition in PARTITIONS:
        if seed in partition.seeds:
            return partition
    raise ValueError(f"Seed {seed} is outside the declared Duet 2 protocol.")


def is_training_seed(seed: int) -> bool:
    """Return True only when this seed belongs to the declared training split."""
    try:
        return partition_for(seed).name == "training"
    except ValueError:
        return False


def require_partition(seeds: Iterable[int], expected: str) -> tuple[int, ...]:
    """Reject duplicates and cross-split use before a costly simulation run."""
    declared = tuple(int(seed) for seed in seeds)
    if not declared:
        raise ValueError("Provide at least one declared Duet 2 seed.")
    if len(set(declared)) != len(declared):
        raise ValueError("Seeds must be unique.")
    invalid = [seed for seed in declared if partition_for(seed).name != expected]
    if invalid:
        raise ValueError(f"{expected.title()} run received seeds outside its split: {invalid}.")
    return declared


def manifest() -> dict:
    """Return stable metadata recorded alongside every Duet 2 experiment."""
    partitions = [asdict(partition) for partition in PARTITIONS]
    serialized = json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": SCHEMA,
        "partitions": partitions,
        "partition_sha256": sha256(serialized).hexdigest(),
        "rules": {
            "training": "No evaluation seed may produce demonstrations or calibration samples.",
            "validation": "Validation may select a model but may not update weights.",
            "evaluation": "Evaluation seeds remain untouched until the candidate is frozen.",
        },
    }
