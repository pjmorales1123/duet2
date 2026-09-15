"""Collect Duet 2 demonstrations only from the declared training partition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation_lab.duet_protocol import manifest, require_partition
from simulation_lab.storage import require_space
from scripts.collect_dinner_learning import collect_sequence, save_json


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse explicit comma-separated seeds so every sample stays auditable."""
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=tuple(range(1000, 1010)))
    args = parser.parse_args()

    seeds = require_partition(args.seeds, "training")
    if args.output.exists():
        raise FileExistsError("Existing data is preserved; choose a fresh output folder.")

    require_space(args.output, len(seeds) * 600 * 1024**2)
    args.output.mkdir(parents=True)
    protocol = manifest()
    protocol.update({"requested_seeds": list(seeds), "collector": "scripts/collect_duet_demos.py"})
    save_json(args.output / "duet-protocol.json", protocol)

    sequences = []
    for seed in seeds:
        sequences.append(collect_sequence(seed, args.output / f"seed-{seed}"))
        save_json(args.output / "summary.json", {"schema": protocol["schema"], "sequences": sequences})
    print(json.dumps({"collected": len(sequences), "eligible_sequences": sum(
        row["eligible"] == len(row["planned_skills"]) for row in sequences
    )}))
    return int(any(row["eligible"] != len(row["planned_skills"]) for row in sequences))


if __name__ == "__main__":
    raise SystemExit(main())
