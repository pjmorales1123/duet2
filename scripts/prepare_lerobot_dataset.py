"""Convert scripts/collect_dinner_learning.py output into LeRobot dataset format.

Split into two halves on purpose:
  extract_episodes()  -- pure numpy/json, no lerobot import. Runs anywhere,
                          including this local venv where lerobot isn't installed.
  build_lerobot_dataset() -- imports lerobot; run this cell inside Colab (or
                          anywhere lerobot>=0.4 with v3 datasets is installed)
                          after uploading the extracted episodes.

Usage:
  # Locally (no lerobot needed): sanity-check the extraction only.
  python scripts/prepare_lerobot_dataset.py --input datasets/duet-micro-v1 --check

  # In Colab, after `pip install lerobot` and uploading datasets/duet-micro-v1:
  python scripts/prepare_lerobot_dataset.py --input datasets/duet-micro-v1 \
      --repo-id your-hf-username/duet-micro-v1 --build
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FPS = 10  # matches the 100-tick (~10Hz at a 1kHz physics step) observation cadence
STATE_DIM = 24  # qpos[:12] + qvel[:12], both arms
ACTION_DIM = 12  # both arms' actuator targets


def extract_episodes(input_dir: Path):
    """Yield one dict per training-eligible skill episode: images, states,
    actions (all synced to the same frame indices) plus the language
    instruction and skill/seed identifiers. No lerobot dependency."""
    for seed_dir in sorted(input_dir.glob("seed-*")):
        for skill_dir in sorted(seed_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            manifest_path = skill_dir / "manifest.json"
            traj_path = skill_dir / "trajectory.npz"
            obs_path = skill_dir / "observations.npz"
            if not (manifest_path.exists() and traj_path.exists() and obs_path.exists()):
                continue
            manifest = json.loads(manifest_path.read_text())
            if not manifest.get("training_eligible"):
                continue

            traj = np.load(traj_path)
            obs = np.load(obs_path)
            endpoints = traj["actions20"]          # (n_endpoints, 12), float32
            action_indices = traj["action_indices"]  # tick positions of endpoints
            frames = obs["overhead"]                # (n_frames, H, W, 3), uint8
            wrist_frames = obs["wrist"] if "wrist" in obs.files else None
            states = obs["state"]                   # (n_frames, 24), float32
            frame_ticks = obs["action_indices"]     # tick position of each frame

            # Interpolate the sparse 20Hz endpoint targets onto every observed
            # frame's tick, exactly like scripts/collect_dinner_learning.py's
            # replay() does, so (image, state, action) triples are in sync.
            actions = np.stack([
                np.interp(frame_ticks, action_indices, endpoints[:, j])
                for j in range(ACTION_DIM)
            ], axis=1).astype("float32")

            instruction = manifest["language"]["instruction"]
            yield {
                "seed": seed_dir.name.removeprefix("seed-"),
                "skill": manifest["skill"],
                "instruction": instruction,
                "images": frames,
                "wrist_images": wrist_frames,
                "states": states.astype("float32"),
                "actions": actions,
            }


def build_lerobot_dataset(input_dir: Path, repo_id: str, root: Path | None = None):
    """Materialize a LeRobotDataset v3 on disk (and optionally push to Hub).
    Import is local to this function so `--check` works without lerobot installed.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = list(extract_episodes(input_dir))
    if not episodes:
        raise ValueError(f"No training-eligible episodes found under {input_dir}")
    if any(ep["wrist_images"] is None for ep in episodes):
        raise ValueError("Some episodes are missing the wrist camera (old collection run) - "
                          "regenerate the dataset with the updated collect_dinner_learning.py.")
    height, width = episodes[0]["images"].shape[1:3]
    wrist_height, wrist_width = episodes[0]["wrist_images"].shape[1:3]

    # ponytail: image dtype, not video. 4.4K frames total is too small for video
    # encode/decode overhead to pay off; raw frames also removes the CPU-bound
    # AV1 decode bottleneck that was limiting Colab dataloader throughput.
    features = {
        "observation.images.overhead": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (wrist_height, wrist_width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"qpos_or_qvel_{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [f"actuator_{i}" for i in range(ACTION_DIM)],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=root,
        robot_type="so101_dual_arm",
        use_videos=False,
    )

    for ep in episodes:
        n = len(ep["images"])
        for i in range(n):
            dataset.add_frame({
                "observation.images.overhead": ep["images"][i],
                "observation.images.wrist": ep["wrist_images"][i],
                "observation.state": ep["states"][i],
                "action": ep["actions"][i],
                "task": ep["instruction"],
            })
        dataset.save_episode()

    dataset.finalize()
    print(f"Built LeRobot dataset with {len(episodes)} episodes at "
          f"{dataset.root if hasattr(dataset, 'root') else root}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                         help="Folder produced by collect_dinner_learning.py (contains seed-* subfolders)")
    parser.add_argument("--repo-id", help="HF dataset repo id, e.g. you/duet-micro-v1 (required with --build)")
    parser.add_argument("--root", type=Path, default=None, help="Local output folder for the built dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                        help="Extract only; print per-episode shapes and counts. No lerobot needed.")
    group.add_argument("--build", action="store_true",
                        help="Actually build the LeRobot dataset. Requires lerobot installed.")
    args = parser.parse_args()

    if args.check:
        n_episodes, n_frames, skills = 0, 0, {}
        for ep in extract_episodes(args.input):
            n_episodes += 1
            n_frames += len(ep["images"])
            skills[ep["skill"]] = skills.get(ep["skill"], 0) + 1
            assert ep["images"].shape[1:3] and ep["images"].dtype == np.uint8
            assert ep["states"].shape[1] == STATE_DIM
            assert ep["actions"].shape[1] == ACTION_DIM
            assert len(ep["images"]) == len(ep["states"]) == len(ep["actions"])
        print(f"episodes={n_episodes} total_frames={n_frames} by_skill={skills}")
        if n_episodes == 0:
            raise SystemExit("No training-eligible episodes found - nothing to train on.")
    else:
        if not args.repo_id:
            parser.error("--build requires --repo-id")
        build_lerobot_dataset(args.input, args.repo_id, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
