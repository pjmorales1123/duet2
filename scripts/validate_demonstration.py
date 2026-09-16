"""Validate time alignment, action/state structure, image references and hashes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def validate(folder):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("trajectory_complete"):
        raise ValueError("The recording explicitly reports an incomplete trajectory.")
    model = mujoco.MjModel.from_xml_path(str(folder/"scene.xml"))
    actions = read_rows(folder/manifest["actions"])
    observations = read_rows(folder/manifest["observations"])
    assert len(actions) == manifest["action_count"]
    assert len(observations) == manifest["observation_count"]
    for index, action in enumerate(actions):
        assert action["index"] == index
        assert abs(action["time_s"]-index*.005) < 1e-7
        for key in ("target", "ctrl"):
            values = np.array(action[key])
            assert values.shape == (12,) and np.isfinite(values).all()
        assert np.all(np.array(action["target"]) >= model.actuator_ctrlrange[:, 0]-.0001)
        assert np.all(np.array(action["target"]) <= model.actuator_ctrlrange[:, 1]+.0001)
    assert observations and observations[-1]["terminal"]
    for index, observation in enumerate(observations):
        assert observation["index"] == index
        qpos, qvel = np.array(observation["qpos"]), np.array(observation["qvel"])
        assert qpos.shape == (model.nq,) and qvel.shape == (model.nv,)
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        assert abs(observation["time_s"]-observation["action_index"]*.005) < 1e-7
        if not observation["terminal"]:
            assert observation["action_index"] == index*10
            assert abs(actions[observation["action_index"]]["time_s"]-observation["time_s"]) < 1e-7
        else:
            assert observation["action_index"] == len(actions)
    images = 0
    if manifest["images"]["status"] == "completed":
        pairs = set()
        for line in (folder/manifest["images"]["index"]).read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            observation = observations[item["observation_index"]]
            assert item["time_s"] == observation["time_s"]
            assert item["action_index"] == observation["action_index"]
            assert item["camera"] in manifest["images"]["cameras"]
            pair = (item["observation_index"], item["camera"])
            assert pair not in pairs
            pairs.add(pair)
            file = (folder/item["path"]).resolve()
            assert file.is_relative_to(folder.resolve())
            assert hashlib.sha256(file.read_bytes()).hexdigest() == item["sha256"]
            with Image.open(file) as image:
                assert image.size == tuple(manifest["images"]["resolution"])
                image.verify()
            images += 1
        expected = sum(row["index"] % manifest["images"].get("observation_stride", 4) == 0 for row in observations)*len(manifest["images"]["cameras"])
        assert images == expected == manifest["images"]["count"]
    elif manifest["images"]["status"] != "disabled":
        raise ValueError("Image export has not completed: "+manifest["images"]["status"])
    return {"id": manifest["id"], "valid": True, "actions": len(actions), "observations": len(observations),
            "images": images, "duration_s": observations[-1]["time_s"], "outcome": manifest["outcome"]["status"],
            "training_eligible": manifest["training_eligible"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.folder), indent=2))
