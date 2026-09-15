"""Asynchronous episode recording and synchronized camera reconstruction."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from .scene import ASSETS, JOINTS, CAMERAS
from .storage import require_space, GIB

EPISODE_ROOT = Path(__file__).resolve().parents[1] / ".run" / "demonstrations"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.01*2**attempt)


def export_images(folder_name, stop):
    """Render each image from its recorded observation, never pair a stale stream frame."""
    folder = Path(folder_name)
    manifest_path = folder/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    renderer = None
    parent = multiprocessing.parent_process()
    try:
        width, height = manifest["images"]["resolution"]
        camera_count = len(manifest["images"]["cameras"])
        frame_count = manifest.get('observation_count', 0)//manifest['images'].get('observation_stride', 4)+1
        # Raw RGB is a conservative estimate for the compressed image export.
        require_space(folder, frame_count*camera_count*width*height*3 + 32*1024**2)
        model = mujoco.MjModel.from_xml_path(str(folder/"scene.xml"))
        # Multisampling is a render-only expense; retain full image resolution.
        model.vis.quality.offsamples = 0
        data = mujoco.MjData(model)
        width, height = manifest["images"]["resolution"]
        cameras = manifest["images"]["cameras"]
        renderer = mujoco.Renderer(model, height=height, width=width)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        image_dir = folder/"images"
        image_dir.mkdir(exist_ok=True)
        count = 0
        with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as source, (folder/"images.jsonl").open("w", encoding="utf-8") as index:
            for line in source:
                sample = json.loads(line)
                if sample["index"] % manifest['images'].get('observation_stride', 4):
                    continue
                if stop.is_set() or (parent is not None and not parent.is_alive()):
                    raise RuntimeError("Image export interrupted.")
                data.qpos[:] = sample["qpos"]
                data.qvel[:] = sample["qvel"]
                data.time = sample["simulation_time_s"]
                mujoco.mj_forward(model, data)
                for camera in cameras:
                    if count % 30 == 0:
                        require_space(folder, 30*width*height*3 + 8*1024**2)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    file = image_dir/f"{sample['index']:06d}_{camera}.jpg"
                    Image.fromarray(renderer.render()).save(file, format="JPEG", quality=85)
                    row = {"observation_index": sample["index"], "action_index": sample["action_index"],
                           "time_s": sample["time_s"], "camera": camera, "path": file.relative_to(folder).as_posix(),
                           "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                    index.write(json.dumps(row)+"\n")
                    count += 1
                if count % 30 == 0:
                    try:
                        write_json(folder/"image-progress.json", {"frames": count})
                    except OSError:
                        # Optional progress may be stale while Windows holds an
                        # open reader handle; never sacrifice the actual images.
                        pass
        manifest["images"].update(status="completed", count=count, index="images.jsonl", multisampling=0)
        manifest["images"].pop("error", None)
        manifest["training_eligible"] = bool(manifest["trajectory_complete"] and manifest["outcome"]["status"] == "succeeded"
                                             and all(r['passed'] for r in manifest.get('replays', [])))
    except Exception as exc:
        manifest["images"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        manifest["training_eligible"] = False
    finally:
        if renderer:
            renderer.close()
        write_json(manifest_path, manifest)


class EpisodeRecorder:
    """Physics puts copies into a bounded thread queue; file and image work stay outside it."""
    def __init__(self, root=EPISODE_ROOT, queue_size=512, writer_delay_s=0.):
        self.root = Path(root)
        self.pending = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._run, name="benchlab-recording", daemon=True)
        self.meta = {"id": None, "status": "idle", "busy": False}
        self.capturing = False
        self.finish_request = None
        self.overflow = False
        self.action_count = self.observation_count = 0
        self.writer_delay_s = writer_delay_s
        self.export_process = None
        self.export_stop = multiprocessing.get_context("spawn").Event()
        self.worker.start()

    def snapshot(self):
        with self.lock:
            return deepcopy(self.meta)

    def _status(self, **values):
        with self.lock:
            self.meta.update(values)

    def start(self, xml, layout, task, start_time, images=True, image_hz=20, cameras=None):
        if image_hz not in (5,10,20):
            raise ValueError('Image frequency must divide the 20 Hz observation grid: 5, 10 or 20.')
        if cameras is not None and (not cameras or len(set(cameras)) != len(cameras) or any(c not in CAMERAS for c in cameras)):
            raise ValueError('Choose distinct, known recording cameras.')
        if not self.worker.is_alive():
            raise ValueError("The recording worker has stopped. Restart BenchLab before recording again.")
        if self.snapshot()["busy"]:
            raise ValueError("The previous demonstration is still being saved. Wait for it or turn off recording.")
        require_space(self.root, GIB if images else 128*1024**2)
        episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:10]
        self.start_time = float(start_time)
        self.action_count = self.observation_count = 0
        self.finish_request = None
        self.overflow = False
        self.capturing = True
        self.meta = {"id": episode_id, "status": "recording", "busy": True, "actions": 0, "observations": 0,
                     "images": 0, "message": "Recording actions and observations.", "path": str(self.root/episode_id)}
        self.pending.put_nowait(("start", {"xml": xml, "layout": deepcopy(layout), "task": task,
                                           "images": images, "id": episode_id, "start_time": self.start_time,
                                           'image_hz': image_hz, 'cameras': cameras or ['overhead','left_wrist_cam','right_wrist_cam']}))
        return episode_id

    def capture(self, data, target, applied_ctrl, task):
        if not self.capturing:
            return
        index = self.action_count
        observation = self.observation(data, index) if index % 10 == 0 else None
        row = {"index": index, "time_s": round(float(data.time)-self.start_time, 9),
               "stage": task.stage, "target": target.copy(), "ctrl": applied_ctrl.copy()}
        try:
            self.pending.put_nowait(("step", (row, observation)))
            self.action_count += 1
        except queue.Full:
            self.capturing = False
            self.overflow = True
            self._status(status="failed", message="Recording queue overflowed. This episode is incomplete; physics continues.")

    def observation(self, data, action_index, terminal=False):
        row = {"index": self.observation_count, "action_index": action_index,
               "time_s": round(float(data.time)-self.start_time, 9), "simulation_time_s": float(data.time),
               "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "terminal": terminal,
               'robot_joint_position': data.qpos[:12].copy(), 'robot_joint_velocity': data.qvel[:12].copy(),
               'actuator_force': data.actuator_force.copy()}
        self.observation_count += 1
        return row

    def finish(self, task, data):
        if not self.snapshot()["busy"] or self.finish_request is not None:
            return
        self.capturing = False
        self.finish_request = {"outcome": task.snapshot(), "terminal": self.observation(data, self.action_count, True),
                               "complete": not self.overflow}
        self._status(status="finalizing", message="Saving the episode outcome.")

    def close(self):
        self.capturing = False
        self.stopping.set()
        self.export_stop.set()
        self.worker.join(timeout=5)
        if self.export_process is not None:
            self.export_process.join(timeout=5)
            if self.export_process.is_alive():
                self.export_process.terminate()
                self.export_process.join(timeout=2)

    @staticmethod
    def _line(stream, value):
        stream.write(json.dumps(value, default=lambda a: a.tolist(), separators=(",", ":"), allow_nan=False)+"\n")

    def _run(self):
        actions = observations = None
        manifest = folder = None
        written_actions = written_observations = 0
        next_progress_read = 0.
        try:
            while not self.stopping.is_set() or not self.pending.empty() or actions is not None:
                try:
                    operation, payload = self.pending.get(timeout=.05)
                except queue.Empty:
                    operation = None
                if operation == "start":
                    folder = self.root/payload["id"]
                    folder.mkdir(parents=True, exist_ok=False)
                    xml = ET.fromstring(payload["xml"])
                    xml.find("compiler").set("meshdir", os.path.relpath(ASSETS/"assets", folder).replace("\\", "/"))
                    ET.ElementTree(xml).write(folder/"scene.xml", encoding="unicode")
                    actions = gzip.open(folder/"actions.jsonl.gz", "wt", encoding="utf-8", compresslevel=3)
                    observations = gzip.open(folder/"observations.jsonl.gz", "wt", encoding="utf-8", compresslevel=3)
                    written_actions = written_observations = 0
                    manifest = {"schema_version": 1, "id": payload["id"], "created_at": datetime.now(timezone.utc).isoformat(),
                                "engine": "MuJoCo "+mujoco.__version__, "status": "recording", "trajectory_complete": False,
                                "layout": payload["layout"], "requested_goal": payload["task"], "start_simulation_time_s": payload["start_time"],
                                "state_model": "scene.xml", "actions": "actions.jsonl.gz", "observations": "observations.jsonl.gz",
                                "action_hz": 200, "observation_hz": 20, "units": {"arm": "radians", "position": "metres", "time": "seconds"},
                                "joint_order": [side+"_"+j for side in ("left", "right") for j in JOINTS],
                                "action_contract": "qpos/qvel observations at t precede ctrl applied for [t,t+0.005). target is the nominal position target; ctrl includes the gripper torque limit.",
                                "observation_contract": "qpos/qvel follow the saved MuJoCo model. First 12 coordinates are arm joints. Exact simulator object state is privileged teacher/evaluator information.",
                                "images": {"status": "pending" if payload["images"] else "disabled", "hz": payload['image_hz'], "resolution": [640, 480],
                                           'observation_stride': 20//payload['image_hz'],
                                           "cameras": payload['cameras'],
                                           "alignment": "Offline render of exactly the referenced observation qpos/qvel, not asynchronous live stream images."}}
                    write_json(folder/"manifest.json", manifest)
                elif operation == "step" and actions is not None:
                    if written_actions % 200 == 0:
                        require_space(folder, 32*1024**2)
                    row, observation = payload
                    self._line(actions, row)
                    written_actions += 1
                    if observation is not None:
                        self._line(observations, observation)
                        written_observations += 1
                    if self.writer_delay_s:
                        self.stopping.wait(self.writer_delay_s)
                    self._status(actions=written_actions, observations=written_observations)
                if actions is not None and self.pending.empty() and (self.finish_request is not None or self.stopping.is_set()):
                    finish = self.finish_request
                    if finish:
                        self._line(observations, finish["terminal"])
                        written_observations += 1
                    actions.close(); observations.close()
                    actions = observations = None
                    complete = bool(finish and finish["complete"])
                    manifest.update(status="completed" if complete else "incomplete", trajectory_complete=complete,
                                    action_count=written_actions, observation_count=written_observations,
                                    outcome=finish["outcome"] if finish else {"status": "interrupted"},
                                    training_eligible=bool(complete and finish["outcome"]["status"] == "succeeded" and manifest["images"]["status"] == "disabled"))
                    if self.overflow:
                        manifest["recording_error"] = "Queue overflow: samples were not silently dropped or accepted as a valid dataset."
                    if not complete or self.stopping.is_set():
                        manifest["images"]["status"] = "skipped_incomplete" if not complete else "interrupted"
                    write_json(folder/"manifest.json", manifest)
                    if complete and manifest["images"]["status"] == "pending" and not self.stopping.is_set():
                        self._status(status="exporting", observations=written_observations, message="Rendering synchronized camera observations.")
                        self.export_stop.clear()
                        self.export_process = multiprocessing.get_context("spawn").Process(target=export_images, args=(str(folder), self.export_stop), name="benchlab-demo-images", daemon=True)
                        self.export_process.start()
                    else:
                        self._status(status="completed" if complete else "failed", busy=False, observations=written_observations,
                                     message="Demonstration saved." if complete else "Incomplete recording saved with its failure reason.")
                if self.export_process is not None:
                    if self.export_process.is_alive():
                        if time.monotonic() >= next_progress_read:
                            next_progress_read = time.monotonic()+.5
                            try:
                                count = json.loads((folder/"image-progress.json").read_text())["frames"]
                                self._status(images=count)
                            except (OSError, ValueError, KeyError):
                                pass
                    else:
                        self.export_process.join(timeout=.1)
                        final = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
                        good = final["images"]["status"] == "completed"
                        self._status(status="completed" if good else "failed", busy=False, images=final["images"].get("count", 0),
                                     message="Demonstration and synchronized images saved." if good else "Image export failed; raw trajectory remains available.")
                        self.export_process = None
        except Exception as exc:
            self.capturing = False
            self._status(status="failed", busy=False, message=f"Recording failed: {type(exc).__name__}: {exc}")
            if manifest is not None and folder is not None:
                manifest.update(status="incomplete", trajectory_complete=False, training_eligible=False,
                                recording_error=f"{type(exc).__name__}: {exc}")
                try:
                    write_json(folder/"manifest.json", manifest)
                except OSError:
                    pass
        finally:
            for stream in (actions, observations):
                if stream is not None:
                    stream.close()
