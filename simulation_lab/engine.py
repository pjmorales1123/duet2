"""Fixed-step physics/controller owner plus an independent camera process."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import multiprocessing
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np

from .autonomy import LiftReturn
from .dinner import dinner_state
from .dinner_autonomy import DinnerSequence
from .rendering import offer_latest, render_worker
from .recording import EpisodeRecorder
from .scene import CAMERAS, HOME, JOINT_LABELS, JOINTS, RackPose, TABLE_Z, build_scene
from .slots import slot_state

LOG = logging.getLogger(__name__)
RUN_DIR = Path(__file__).resolve().parents[1] / ".run"


@dataclass
class Request:
    operation: str
    payload: dict
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    expired: bool = False


class LabEngine:
    def __init__(self, width=960, height=540, fps=20, render_delay_s=0., scenario="dinner"):
        self.width, self.height, self.fps = width, height, fps
        self.initial_scenario = scenario
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.commands: queue.Queue[Request] = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.state: dict = {}
        self.frame, self.frame_id = b"", 0
        self.error: str | None = None
        self.last_seen = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="benchlab-physics", daemon=True)
        self.render_thread = threading.Thread(target=self._receive_frames, name="benchlab-frames", daemon=True)
        self.render_job = None
        context = multiprocessing.get_context("spawn")
        self.render_inputs = context.Queue(maxsize=1)
        self.render_outputs = context.Queue(maxsize=1)
        self.render_stopping = context.Event()
        self.render_process = context.Process(target=render_worker, args=(self.render_inputs, self.render_outputs, self.render_stopping, width, height, fps, render_delay_s), name="benchlab-camera", daemon=True)
        self.render_stats = {"fps": 0., "frame_work_ms": 0., "frame_simulation_time_s": 0.}
        self.running, self.camera, self.shadows = True, "center", True
        self.motion_started: float | None = None
        self.generation = 0
        self.real_time_factor = 1.
        self.dropped_wall_time = 0.
        self.recorder = None
        self.dinner_suite=RUN_DIR.parent/'models'/'dinner_suite'/'suite.json'

    def start(self):
        self.render_process.start()
        self.thread.start()
        self.render_thread.start()
        if not self.ready.wait(timeout=60):
            raise RuntimeError("The simulator did not start within 60 seconds.")
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.stopping.set()
        self.render_stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        if self.recorder is not None:
            if self.task.active:
                self.task.cancel(self.target)
            self._finish_recording()
            self.recorder.close()
        if self.render_thread.ident is not None:
            self.render_thread.join(timeout=5)
        if self.render_process.pid is not None:
            self.render_process.join(timeout=5)
            if self.render_process.is_alive():
                self.render_process.terminate()
                self.render_process.join(timeout=2)
        for channel in (self.render_inputs, self.render_outputs):
            channel.cancel_join_thread()
            channel.close()

    def touch(self):
        self.last_seen = time.monotonic()

    def snapshot(self):
        self.touch()
        with self.lock:
            return deepcopy(self.state)

    def image(self):
        self.touch()
        with self.lock:
            return self.frame_id, self.frame

    def submit(self, operation: str, payload: dict):
        self.touch()
        if self.error:
            raise RuntimeError(self.error)
        request = Request(operation, payload)
        try:
            self.commands.put(request, timeout=1)
        except queue.Full as exc:
            raise RuntimeError("The simulator is busy. Try again shortly.") from exc
        if not request.done.wait(timeout=15):
            request.expired = True
            raise RuntimeError("The simulator did not finish the requested change. Check its state before retrying.")
        if request.error:
            raise request.error
        return self.snapshot()

    def _reset(self, seed=42, rack_count=3, racks=None, practice=False, transfer_side=None, scenario="chemistry", dinner_preset="task", drawer_open=False, bottle_start='upright'):
        if bottle_start not in ('upright','sideways','wide_left','wide_rectangle'):raise ValueError('Unknown bottle start.')
        if bottle_start != 'upright' and (scenario != 'dinner' or dinner_preset != 'task'):
            raise ValueError('Bottle practice requires the dinner Task start scene.')
        xml, layout = build_scene(seed, rack_count, racks, practice=practice, transfer_side=transfer_side,
                                  scenario=scenario, dinner_preset=dinner_preset, drawer_open=drawer_open)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        target = np.asarray(HOME * 2, dtype=float)
        data.qpos[:12] = target
        data.ctrl[:] = target
        if scenario == "dinner":
            data.qpos[model.joint("drawer_slide").qposadr[0]] = layout["drawer"]["initial_open_m"]
            if bottle_start == 'sideways':
                angle = np.pi/4
                c,s = np.cos(angle/2),np.sin(angle/2)
                adr = model.joint('bottle_free').qposadr[0]
                data.qpos[adr:adr+7] = [-.10,-.15,layout['table_z']+.026,*(np.array([c,-s,c,s])*np.sqrt(.5))]
            elif bottle_start=='wide_left':
                from .dinner import left_reach_bottle_pose
                pose=left_reach_bottle_pose(seed);adr=model.joint('bottle_free').qposadr[0]
                data.qpos[adr:adr+7]=[pose['x'],pose['y'],layout['table_z']+.001,np.cos(pose['yaw']/2),0,0,np.sin(pose['yaw']/2)]
            elif bottle_start=='wide_rectangle':
                try:from .wide_bottle_task import wide_bottle_pose
                except ImportError:raise ValueError('Use the training Python environment for wider bottle practice.') from None
                adr=model.joint('bottle_free').qposadr[0]
                data.qpos[adr:adr+7]=wide_bottle_pose(seed,layout['table_z'])
                from .wide_bottle_setup import validate_start
                layout['wide_bottle_reset']=validate_start(model,data)
            layout['bottle_start'] = bottle_start
        for _ in range(300 if bottle_start != 'upright' else 200):
            mujoco.mj_step(model, data)
        data.time = 0.
        self.last_command_object = None
        mujoco.mj_forward(model, data)
        if hasattr(self, "task") and self.task.active:
            self.task.cancel(self.target)
            self._record_result()
        if hasattr(self,'task') and hasattr(self.task,'close'):self.task.close()
        self.model, self.data, self.layout, self.target = model, data, layout, target
        self.xml = xml
        self.task = DinnerSequence(model,data,layout) if scenario == "dinner" else LiftReturn(model,data,layout)
        self.motion_started = None
        self.generation += 1
        self.dropped_wall_time = 0.
        self.slot_cache_time = float("-inf")
        self.slots_cache = []
        # The renderer compiles its own model and owns its own MjData.

    def _control(self, payload):
        if self.task.active and (payload.get("preset") or payload.get("targets_deg") is not None):
            raise ValueError("A goal owns the arm controls. Cancel it before moving joints manually.")
        if payload.get("step") and payload.get("running", self.running):
            raise ValueError("Pause the simulation before single stepping.")
        if payload.get("shadows") is not None:
            self.shadows = payload["shadows"]
        if payload.get("camera") is not None:
            if payload["camera"] not in CAMERAS:
                raise ValueError("Unknown camera.")
            self.camera = payload["camera"]
        if payload.get("running") is not None:
            self.running = payload["running"]
        if payload.get("preset") == "home":
            self.target[:] = HOME * 2
            self.motion_started = None
        elif payload.get("preset") == "gentle":
            self.target[:] = HOME * 2
            self.motion_started = float(self.data.time)
            self.running = True
        if payload.get("targets_deg") is not None:
            side = payload.get("arm")
            if side not in ("left", "right"):
                raise ValueError("Select the left or right arm.")
            values = np.deg2rad(payload["targets_deg"])
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("Supply six finite joint targets.")
            offset = 0 if side == "left" else 6
            for i, value in enumerate(values):
                lo, hi = self.model.actuator_ctrlrange[offset + i]
                joint_range = self.model.jnt_range[self.model.actuator_trnid[offset + i, 0]]
                self.target[offset + i] = np.clip(value, max(lo, joint_range[0]), min(hi, joint_range[1]))
            self.motion_started = None
        if payload.get("step"):
            self._step(10)

    def _task_command(self, payload):
        if payload["action"] == "cancel":
            self.task.cancel(self.target)
            if self.task.request_pause:
                self.running = False
                self.task.request_pause = False
        else:
            if self.task.active:raise ValueError('A goal is already running. Cancel it first.')
            if self.motion_started is not None:
                raise ValueError("Wait for the motion test to finish or reset the scene before starting a goal.")
            if payload.get("record") and self.recorder is not None and self.recorder.snapshot()["busy"]:
                raise ValueError("The previous demonstration is still being saved. Wait or turn off recording.")
            if payload.get('record'):
                from .storage import require_space,GIB
                from .recording import EPISODE_ROOT
                require_space(self.recorder.root if self.recorder else EPISODE_ROOT,GIB if payload.get('record_images',True) else 128*1024**2)
            if self.layout.get("scenario") == "dinner":
                if hasattr(self.task,'close'):self.task.close()
                self.task=DinnerSequence(self.model,self.data,self.layout)
                self.task.start(side=payload.get("arm","auto"),object_id=payload.get("object_id"),kind=payload.get("kind","set_table"))
            else:
                self.task.start(payload.get("arm", "auto"), payload.get("tube_id"),
                                kind=payload.get("kind", "lift_return"), destination_slot=payload.get("destination_slot"))
            if payload.get("record"):
                if self.recorder is None:
                    self.recorder = EpisodeRecorder()
                self.task.recording_id = self.recorder.start(self.xml, self.layout, self.task.snapshot(), self.data.time,
                                                              images=payload.get("record_images", True))
            self.running = True

    def _language_command(self,payload):
        from .language import parse_command
        from .command_task import CommandSequence
        plan=parse_command(payload['text'],getattr(self,'last_command_object',None))
        if plan.get('control')=='cancel':
            self._task_command({'action':'cancel'});return
        if self.layout.get('scenario')!='dinner':raise ValueError('Language commands use the dinner scene.')
        if self.task.active:raise ValueError('A task is running. Say stop or wait for it to finish.')
        if payload.get('mode')=='learned_dinner_wide':
            try:from .wide_bottle_task import make_sequence
            except ImportError:raise ValueError('Use the training Python environment for wider bottle control.') from None
            if hasattr(self.task,'close'):self.task.close()
            candidate=make_sequence(self.model,self.data,self.layout,plan)
        elif payload.get('mode') in ('learned_dinner','learned_dinner_visual'):
            if not self.dinner_suite.is_file():raise ValueError('The learned dinner suite is not installed.')
            try:
                from .learned_dinner import LearnedDinnerSequence,observe_scene
                from .learned_plan import plan_learned_steps
            except ImportError:raise ValueError('Use the training Python environment for learned dinner control.') from None
            paths=json.loads(self.dinner_suite.read_text(encoding='utf-8-sig'))
            visual_plan=plan_learned_steps(plan,observe_scene(self.model,self.data),paths)
            checkpoints={name:self.dinner_suite.parent/path for name,path in paths.items()}
            if payload.get('mode')=='learned_dinner_visual':
                from .mug_visual_profile import make_sequence
                if hasattr(self.task,'close'):self.task.close()
                candidate=make_sequence(self.model,self.data,self.layout,checkpoints,visual_plan['steps'])
            else:
                candidate=LearnedDinnerSequence(self.model,self.data,self.layout,checkpoints,visual_plan['steps'])
            candidate.plan=visual_plan
            candidate.message='Plan: '+' → '.join(visual_plan['steps'])+'. '+' '.join(visual_plan['reasons'])
        elif payload.get('mode') in ('learned_bottle','learned_bottle_legacy'):
            steps=plan['steps']
            if len(steps)!=1 or steps[0]['kind']!='dinner_place' or steps[0]['object_id']!='bottle' or steps[0]['arm']=='right' or (steps[0].get('destination') or {}).get('kind','default')!='default':
                raise ValueError('This learned model currently supports “place the bottle” with the left arm and its trained destination. Other instructions require the programmed mode.')
            try:from .learned_task import LearnedBottleTask,DEFAULT_CHECKPOINT,LEGACY_CHECKPOINT
            except ImportError:raise ValueError('Use the training Python environment to enable learned control.') from None
            checkpoint=LEGACY_CHECKPOINT if payload.get('mode')=='learned_bottle_legacy' else DEFAULT_CHECKPOINT
            candidate=LearnedBottleTask(self.model,self.data,self.layout,checkpoint=checkpoint)
        else:
            candidate=CommandSequence(self.model,self.data,self.layout)
            candidate.start_plan(plan)
        if hasattr(self.task,'close'):self.task.close()
        self.task=candidate;self.motion_started=None;self.running=True
        self.last_command_object=plan.get('last_object')

    def _finish_recording(self):
        if self.recorder is not None and not self.task.active and getattr(self.task, "recording_id", None) == self.recorder.snapshot()["id"]:
            self.recorder.finish(self.task, self.data)

    def _step(self, steps):
        for _ in range(steps):
            if self.motion_started is not None:
                elapsed = self.data.time - self.motion_started
                self.target[:] = HOME * 2
                if elapsed >= 6:
                    self.motion_started = None
                else:
                    wave = math.sin(elapsed * math.pi / 3)
                    self.target[0] += .12*wave
                    self.target[6] -= .12*wave
                    self.target[4] += .20*wave
                    self.target[10] -= .20*wave
            self.task.update(self.target)
            self._finish_recording()
            if self.task.request_pause:
                self.task.request_pause = False
                self.running = False
                break
            self.data.ctrl[:] = self.task.apply_gripper_limit(self.target)
            if self.recorder is not None and self.task.active and getattr(self.task, "recording_id", None) == self.recorder.snapshot()["id"]:
                self.recorder.capture(self.data, self.target, self.data.ctrl, self.task)
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("The physics state became invalid. Restart or reset the scene.")

    def _record_result(self):
        self._finish_recording()
        if self.task.status in ("succeeded", "failed", "cancelled") and not self.task.result_written:
            self.task.result_written = True
            record = {"recorded_at": datetime.now(timezone.utc).isoformat(), "seed": self.layout["seed"],
                      "practice": self.layout["practice"], "layout": self.layout, "task": self.task.snapshot()}
            try:
                folder = RUN_DIR / "autonomy"
                folder.mkdir(parents=True, exist_ok=True)
                output = json.dumps(record, indent=2)
                (RUN_DIR / "autonomy-last.json").write_text(output, encoding="utf-8")
                (folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")).write_text(output, encoding="utf-8")
            except OSError:
                LOG.exception("Could not save task telemetry")

    def _publish(self, idle=False):
        if self.data.time-self.slot_cache_time >= .20:
            self.slots_cache = slot_state(self.data, self.layout)
            self.slot_cache_time = float(self.data.time)
        assignments = {tube_id: slot for slot in self.slots_cache for tube_id in slot["occupants"]}
        racks = [{**rack, "slots": [slot["index"] for slot in self.slots_cache if slot["rack"] == rack["id"] and slot["occupants"]]}
                 for rack in self.layout["racks"]]
        arms = {}
        for side_index, side in enumerate(("left", "right")):
            joints = []
            for i, joint in enumerate(JOINTS):
                item = self.model.joint(f"{side}_{joint}")
                joints.append({"name": joint, "label": JOINT_LABELS[i], "actual_deg": float(np.rad2deg(self.data.qpos[item.qposadr[0]])), "target_deg": float(np.rad2deg(self.target[side_index*6+i])), "min_deg": float(np.rad2deg(item.range[0])), "max_deg": float(np.rad2deg(item.range[1]))})
            arms[side] = {"joints": joints, "tool_position_m": self.data.site(f"{side}_gripperframe").xpos.tolist()}
        tubes = []
        for tube in self.layout["tubes"]:
            body = self.data.body(tube["body"])
            location = assignments.get(tube["id"])
            tubes.append({"id": tube["id"], "rack": location["rack"] if location else None, "slot": location["id"] if location else None,
                          "initial_rack": tube["rack"], "position_m": body.xpos.tolist(), "upright": bool(body.xmat[8] > math.cos(math.radians(20))), "on_table": bool(body.xpos[2] > TABLE_Z)})
        snapshot = {"ready": True, "engine": "MuJoCo " + mujoco.__version__, "seed": self.layout["seed"], "running": self.running, "idle": idle, "camera": self.camera, "camera_label": CAMERAS[self.camera], "cameras": CAMERAS, "simulation_time_s": round(float(self.data.time), 3), "physics_timestep_s": float(self.model.opt.timestep), "contacts": int(self.data.ncon), "rack_count": len(self.layout["racks"]), "racks": self.layout["racks"], "tube_count": len(tubes), "upright_count": sum(t["upright"] and t["on_table"] for t in tubes), "tubes": tubes, "arms": arms, "motion_test": self.motion_started is not None, "resolution": [self.width, self.height], "controller": "ground_truth_ik" if self.task.active else "manual_joint_targets", "task": self.task.snapshot(), "practice": self.layout["practice"], "scene_version": self.generation, "shadows": self.shadows, "control_hz": 200, "real_time_factor": round(self.real_time_factor, 3), "dropped_wall_time_s": round(self.dropped_wall_time, 3)}
        job = (self.xml, self.generation, self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), float(self.data.time), self.camera, self.shadows, idle)
        snapshot.update(racks=racks, slots=self.slots_cache, transfer_side=self.layout["transfer_side"],
                        scenario=self.layout.get("scenario", "chemistry"),
                        recording=self.recorder.snapshot() if self.recorder else {"id": None, "status": "idle", "busy": False})
        snapshot.update(dinner_state(self.model, self.data, self.layout))
        if getattr(self.task,'kind',None)=='learned_bottle':snapshot['controller']='learned_bottle'
        if getattr(self.task,'kind',None) in ('learned_dinner','learned_dinner_sequence','learned_bottle_wide'):snapshot['controller']='learned_dinner'
        snapshot['learned_dinner_available']=self.dinner_suite.is_file()
        snapshot['visual_mug_available']=(RUN_DIR.parent/'models/dinner_visual_mug_v1/profile.json').is_file()
        snapshot['wide_bottle_available']=(RUN_DIR.parent/'models/bottle_wide_v1/profile.json').is_file()
        with self.lock:
            snapshot.update(self.render_stats)
            snapshot["frame_id"] = self.frame_id
            self.state = snapshot
            self.render_job = job
        self._record_result()

    def _receive_frames(self):
        try:
            while not self.stopping.is_set():
                # Even "get_nowait" on a multiprocessing pipe can wait for the
                # rest of a serialized message. Keep every IPC operation off
                # the physics thread; it only publishes an in-memory snapshot.
                with self.lock:
                    job = self.render_job
                if job is not None:
                    offer_latest(self.render_inputs, job)
                try:
                    result = self.render_outputs.get(timeout=.05)
                except queue.Empty:
                    if not self.render_process.is_alive():
                        raise RuntimeError("Camera process exited.")
                    continue
                if "error" in result:
                    raise RuntimeError(result["error"])
                with self.lock:
                    if self.generation == result["version"]:
                        self.frame_id += 1
                        self.frame = result["jpeg"]
                        self.render_stats = result["stats"]
                self.ready.set()
        except Exception as exc:
            LOG.exception("Camera worker stopped")
            self.error = f"Camera {type(exc).__name__}: {exc}"
            self.stopping.set()
            self.ready.set()

    def _run(self):
        pending = []
        try:
            self._reset(scenario=self.initial_scenario)
            self._publish()
            previous = checkpoint = time.perf_counter()
            sim_checkpoint = self.data.time
            accumulator = 0.
            published = previous
            while not self.stopping.is_set():
                now = time.perf_counter()
                elapsed = now-previous
                previous = now
                pending = []
                while True:
                    try:
                        request = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if request.expired:
                        request.done.set()
                        continue
                    try:
                        if request.operation == "reset":
                            self._reset(**request.payload)
                            accumulator = 0.
                            sim_checkpoint, checkpoint = self.data.time, time.perf_counter()
                        elif request.operation == "layout":
                            if self.layout.get("scenario") != "chemistry":
                                raise ValueError("Rack controls are available in the chemistry scene. Use a dinner reset to change this scene.")
                            self._reset(seed=self.layout["seed"], racks=[RackPose(**r) for r in request.payload["racks"]], practice=self.layout["practice"], transfer_side=self.layout["transfer_side"])
                            accumulator = 0.
                            sim_checkpoint, checkpoint = self.data.time, time.perf_counter()
                        elif request.operation == "task":
                            self._task_command(request.payload)
                        elif request.operation == "language":
                            self._language_command(request.payload)
                        else:
                            self._control(request.payload)
                    except Exception as exc:
                        request.error = exc
                    pending.append(request)
                idle = time.monotonic()-self.last_seen > 15 and not self.task.active
                if self.running and not idle:
                    accumulator += elapsed
                    if accumulator > .1:
                        self.dropped_wall_time += accumulator-.1
                        accumulator = .1
                    steps = int((accumulator+1e-9)/self.model.opt.timestep)
                    if steps:
                        self._step(steps)
                        accumulator -= steps*self.model.opt.timestep
                else:
                    accumulator = 0.
                if now-checkpoint >= 1.:
                    self.real_time_factor = (self.data.time-sim_checkpoint)/max(now-checkpoint, .001)
                    sim_checkpoint, checkpoint = self.data.time, now
                if pending or now-published >= .05:
                    self._publish(idle)
                    published = now
                for request in pending:
                    request.done.set()
                self.stopping.wait(.001)
        except Exception as exc:
            LOG.exception("Simulator stopped")
            self.error = f"{type(exc).__name__}: {exc}"
            self.stopping.set()
            self.ready.set()
        finally:
            if hasattr(self,'task') and hasattr(self.task,'close'):self.task.close()
            for request in pending:
                if not request.done.is_set():
                    request.error = RuntimeError(self.error or "Simulator stopped.")
                    request.done.set()
            while not self.commands.empty():
                request = self.commands.get_nowait()
                request.error = RuntimeError(self.error or "Simulator stopped.")
                request.done.set()
