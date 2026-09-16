"""Capture a still of the reference (fully-set) dinner table for the demo dashboard.

Boots the FastAPI app in-process via TestClient (handles the lifespan startup/
shutdown for us), resets to the reference layout, grabs one JPEG frame.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

OUTPUT = Path("simulation_lab/web/media/target_arrangement.jpg")


def main() -> None:
    # Import inside main(): the engine's camera worker uses multiprocessing
    # spawn, which re-imports this file in the child process on Windows -
    # importing TestClient/app at module scope would recursively spawn.
    from fastapi.testclient import TestClient
    from simulation_lab.server import app

    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=42)
    seed = args.parse_args().seed

    with TestClient(app) as client:
        client.post("/api/reset", json={"scenario": "dinner", "seed": seed, "dinner_preset": "reference"})
        client.post("/api/control", json={"camera": "overview", "shadows": True})
        time.sleep(1.5)  # let the async camera worker render at least one frame
        response = client.get("/frame.jpg")
        response.raise_for_status()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(response.content)
        print(f"wrote {OUTPUT} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
