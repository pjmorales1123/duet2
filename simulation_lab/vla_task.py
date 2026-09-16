"""Live SmolVLA task: raw camera pixels + free-text instruction -> motor targets.

Unlike learned_task.py's bespoke single-skill PrimitivePolicy, this loads the
real fine-tuned LeRobot SmolVLA checkpoint and runs its actual pre/post
processing (rename, normalize, tokenize, unnormalize) by hand, because
lerobot's PreTrainedConfig.from_pretrained crashes under this project's
Python 3.14 (draccus can't build an argparser for one of SmolVLAConfig's
typed-dict fields) - see models/duet-smolvla-v1 for the checkpoint.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from .autonomy import LiftReturn
from .scene import HOME

_MODELS = Path(__file__).resolve().parents[1] / "models"
# Newest checkpoint wins, but stay runnable with whatever is actually on disk:
# v2 is the 7000-step run (pjmorales04/duet-smolvla-v2), v1 the earlier one.
# DUET_VLA_CHECKPOINT overrides for A/B runs - see scripts/benchmark_live_vla.py.
CHECKPOINT = Path(os.environ.get("DUET_VLA_CHECKPOINT") or next(
    (p for p in (_MODELS / "duet-smolvla-v2", _MODELS / "duet-smolvla-v1") if p.is_dir()),
    _MODELS / "duet-smolvla-v2"))
_CACHE: dict = {}

# CPU-only, vendor-neutral speedups (no Intel/AMD-specific backend): leave a
# couple of cores for the physics thread and UI server, and trade a few of
# the flow-matching model's 10 default denoising steps for latency - fewer
# steps means a slightly less refined action chunk, not a wrong one.
_CPU_THREADS = max(1, (os.cpu_count() or 4) - 2)
_INFERENCE_STEPS = 5

# The checkpoint ships bfloat16, and the demo CPU (i5-8265U, Whiskey Lake) has
# no bf16 instructions - torch emulates every op. Measured on the vision tower,
# identical weights: 16907ms bf16 vs 1904ms fp32, an 8.9x emulation tax for a
# format this machine cannot execute. fp32 is also the more accurate of the two.
# Set DUET_VLA_BF16=1 to run the checkpoint's original dtype for comparison.
_USE_FLOAT32 = os.environ.get("DUET_VLA_BF16", "") not in ("1", "true", "True")
# Vision tower through OpenVINO when an exported IR is present (see
# scripts/export_smolvla_vision.py). DUET_VLA_OPENVINO=0 forces pure PyTorch;
# DUET_VLA_OV_DEVICE picks the plugin - CPU by default, because MuJoCo's GL
# context already owns the UHD 620 and the iGPU plugin silently runs fp16.
_USE_OPENVINO = os.environ.get("DUET_VLA_OPENVINO", "1") not in ("0", "false", "False")
_OV_DEVICE = os.environ.get("DUET_VLA_OV_DEVICE", "CPU")

# The teacher used a fixed arm per skill; the live VLA doesn't choose a side
# explicitly, so feed the wrist view the checkpoint actually trained on for
# whichever object the instruction names. ponytail: keyword match, not NLU -
# good enough since the training instructions themselves are this literal.
_WRIST_BY_KEYWORD = [
    ("fork", "right_wrist_cam"), ("mug", "right_wrist_cam"), ("bottle", "right_wrist_cam"),
    ("spoon", "left_wrist_cam"), ("plate", "left_wrist_cam"),
]

# Which object an instruction is about, for scoring only - never for control.
# Same literal keyword match as _WRIST_BY_KEYWORD, and the same caveat: this is
# not NLU, it just has to agree with the training instructions, which are this
# literal.
_OBJECT_BY_KEYWORD = ("bottle", "plate", "mug", "fork", "spoon")

# The scripted teacher's own placement gate (dinner_autonomy.py's verify stage),
# quoted deliberately so the live policy is graded against the same standard as
# the controller that taught it rather than a softer one invented for it.
_PLACEMENT_XY_M, _PLACEMENT_Z_M, _DISTURBANCE_M = .006, .003, .004


def _load_policy():
    if "policy" not in _CACHE:
        torch.set_num_threads(_CPU_THREADS)
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        cfg = json.loads((CHECKPOINT / "config.json").read_text())
        cfg.pop("type", None)
        # config.json's declared input_features (state dim 6, 3 cameras) is
        # stale metadata inherited from lerobot/smolvla_base - harmless, since
        # the real per-feature shapes live in the normalizer stats below and
        # the model pads state to config.max_state_dim internally regardless.
        cfg["input_features"] = {k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
                                  for k, v in cfg.pop("input_features").items()}
        cfg["output_features"] = {k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
                                   for k, v in cfg.pop("output_features").items()}
        # config.json also carries fields (e.g. use_peft, pretrained_path,
        # rtc_config, compile_model) from the lerobot revision used at export
        # time that this installed lerobot's SmolVLAConfig dataclass no longer
        # (or doesn't yet) declare. Drop anything the dataclass won't accept
        # instead of hard-failing on lerobot version drift.
        accepted = {f.name for f in dataclasses.fields(SmolVLAConfig)}
        cfg = {k: v for k, v in cfg.items() if k in accepted}
        # Tried cfg["load_vlm_weights"] = False here to skip downloading the
        # pretrained HuggingFaceTB/SmolVLM2-500M-Video-Instruct backbone before
        # load_state_dict(strict=True) below overwrites every parameter from
        # this checkpoint anyway. Measured slower cold starts (~147s vs ~55-70s
        # with weights loaded): SmolVLMWithExpertModel's from_pretrained path
        # uses low_cpu_mem_usage=True (skips real weight init, just streams
        # bytes in); the from_config fallback fully initializes ~500M random
        # parameters first, which apparently costs more than the download+copy
        # it avoids. Left as load_vlm_weights=True (the checkpoint's own
        # declared default). See simulation_lab/PERF_NOTES.md.
        # Force every modality to IDENTITY inside the model itself, regardless
        # of what config.json declares. This checkpoint's export-time lerobot
        # normalized/unnormalized outside the model (a separate processor
        # pipeline - see policy_preprocessor.json/policy_postprocessor.json),
        # which is exactly what this module's update() still does by hand
        # against policy_preprocessor_step_5_normalizer_processor.safetensors
        # below. The installed lerobot's SmolVLAConfig instead builds real
        # Normalize/Unnormalize buffers into the policy when a modality maps
        # to MEAN_STD, and (a) asserts they're non-infinite before every
        # forward pass and (b) expects model.safetensors to contain them -
        # neither holds for this checkpoint. IDENTITY skips creating those
        # buffers entirely, so this module's hand-rolled normalize/unnormalize
        # stays the single source of truth and strict loading succeeds.
        cfg["normalization_mapping"] = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                                         "ACTION": NormalizationMode.IDENTITY}
        config = SmolVLAConfig(**cfg)
        policy = SmolVLAPolicy(config)
        policy.load_state_dict(load_file(CHECKPOINT / "model.safetensors"), strict=True)
        policy.eval()
        policy.config.num_steps = _INFERENCE_STEPS
        # ponytail: tried dynamic INT8 quantization here too, but the VLM
        # backbone's own attention path breaks on the quantized linear op's
        # dtype expectations ("qlinear_dynamic (ONEDNN): data type of input
        # should be float") - not worth debugging under time pressure, so
        # only the free wins (thread count, fewer denoising steps) ship.
        # Two independent accelerations, applied in this order because the
        # OpenVINO IR was exported from the fp32 tower and must match it.
        # Both degrade to the stock PyTorch path rather than failing.
        runtime = "PyTorch bfloat16"
        if _USE_FLOAT32:
            from .vla_openvino import use_float32
            use_float32(policy)
            runtime = "PyTorch CPU FP32"
        if _USE_FLOAT32 and _USE_OPENVINO:
            from .vla_openvino import bind_openvino_vision
            if bind_openvino_vision(policy, CHECKPOINT / "openvino", _OV_DEVICE, _CPU_THREADS):
                runtime = f"OpenVINO {_OV_DEVICE} FP32 vision tower + PyTorch CPU FP32 expert"
        stats = load_file(CHECKPOINT / "policy_preprocessor_step_5_normalizer_processor.safetensors")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        _CACHE.update(policy=policy, stats=stats, tokenizer=tokenizer, runtime=runtime)
    return _CACHE["policy"], _CACHE["stats"], _CACHE["tokenizer"]


class SmolVLATask(LiftReturn):
    # One action per 100 physics ticks = 0.5s, which is the cadence the training
    # frames were ACTUALLY recorded at: collect_dinner_learning.py samples every
    # 100 ticks, and this scene's timestep is 0.005s (200Hz), so observations are
    # 0.5s apart. The LeRobot dataset labels them fps=10 because
    # prepare_lerobot_dataset.py assumed a 1kHz physics step ("~10Hz at a 1kHz
    # physics step") - a 5x mislabel that is harmless during training, since
    # behaviour cloning only sees consecutive frames, but fatal at inference:
    # querying every 20 ticks (0.1s) gave each predicted action a fifth of the
    # real time it was trained to take, so the arms tracked 5x too aggressively,
    # overshot, and knocked objects away from their targets. Measured: the policy
    # scored WORSE than not moving at all until this was corrected.
    QUERY_EVERY_TICKS = 100
    # A training episode is 102 frames at that true 0.5s cadence, i.e. ~50.5s of
    # simulated time - not the 10.1s the mislabelled timestamps suggest. Running
    # 25s stopped the policy halfway through the motion it was taught.
    DEFAULT_DURATION_S = 51.

    def start(self, instruction: str, duration_s: float | None = None):
        if self.active:
            raise ValueError("A goal is already running. Cancel it before starting another.")
        if self.layout.get("scenario") != "dinner":
            raise ValueError("The live VLA requires the dinner scene.")
        self.__init__(self.model, self.data, self.layout)
        policy, stats, tokenizer = _load_policy()
        self.policy, self.stats = policy, stats
        # The policy is cached across tasks (_CACHE), and it carries a 50-action
        # chunk queue between select_action() calls. Without this reset a new
        # instruction would spend its first 50 queries replaying actions the
        # *previous* instruction predicted, before the queue drained and the
        # new language tokens ever reached the model.
        policy.reset()
        self._frames = None
        self.instruction = instruction.strip()
        self.wrist_cam = next((cam for kw, cam in _WRIST_BY_KEYWORD if kw in self.instruction.lower()), "right_wrist_cam")
        self._scoring = self._placement_reference()
        encoded = tokenizer(self.instruction + "\n", max_length=48, padding="max_length", truncation=True, return_tensors="pt")
        self.lang_tokens, self.lang_mask = encoded["input_ids"], encoded["attention_mask"].bool()
        self.duration_s = duration_s or self.DEFAULT_DURATION_S
        self.kind, self.stage, self.status = "learned_vla", "neural_control", "running"
        self.message = f'Live SmolVLA running: "{self.instruction}"'
        self.tick, self.inference_ms = 0, []
        self.last_action = np.array(HOME * 2, dtype=float)
        original_offsamples = self.model.vis.quality.offsamples
        self.model.vis.quality.offsamples = 0
        try:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        finally:
            self.model.vis.quality.offsamples = original_offsamples
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0

    def _placement_reference(self):
        """Snapshot what 'done correctly' means for this instruction, at t=0.

        Read-only bookkeeping for scoring: which object the instruction names,
        where it starts, where the scene declares it belongs, and where every
        other object starts so we can tell if this run disturbed them. None of
        this reaches the policy - the policy still sees only pixels and the
        instruction. Returns None when the instruction names no known object,
        in which case the run is reported as unscored rather than as a success.
        """
        name = next((o for o in _OBJECT_BY_KEYWORD if o in self.instruction.lower()), None)
        item = next((o for o in self.layout.get("objects", []) if o["id"] == name), None)
        target = next((t for t in self.layout.get("targets", []) if t["object_id"] == name), None)
        if item is None or target is None:
            return None
        body = self.model.body(item["body"]).id
        others = {o["id"]: self.model.body(o["body"]).id
                  for o in self.layout["objects"] if o["id"] != name and o.get("body")}
        goal = np.array(target["position_m"], dtype=float)
        start = self.data.xpos[body].copy()
        return {"object": name, "body": body, "goal": goal, "start": start,
                "start_error_m": float(np.linalg.norm(start[:2]-goal[:2])),
                "others": {k: self.data.xpos[v].copy() for k, v in others.items()},
                "other_bodies": others}

    def _placement_result(self):
        """Grade the finished run against the teacher's gate. Never raises."""
        reference = getattr(self, "_scoring", None)
        if not reference:
            return None
        end = self.data.xpos[reference["body"]]
        error = float(np.linalg.norm(end[:2]-reference["goal"][:2]))
        z_error = abs(float(end[2]-reference["goal"][2]))
        disturbed = max((float(np.linalg.norm(self.data.xpos[index]-reference["others"][name]))
                         for name, index in reference["other_bodies"].items()), default=0.)
        start_error = reference["start_error_m"]
        return {"object": reference["object"], "placement_xy_error_mm": error*1000,
                "placement_z_error_mm": z_error*1000,
                "start_xy_error_mm": start_error*1000,
                "other_object_max_displacement_mm": disturbed*1000,
                # Objects start near their declared poses, so "inside the gate"
                # alone is satisfiable by never moving. Require that the run
                # also left the object no worse than it found it.
                "improved": error <= start_error,
                "passed": bool(error < _PLACEMENT_XY_M and z_error < _PLACEMENT_Z_M
                               and error <= start_error and disturbed < _DISTURBANCE_M)}

    def close(self):
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None

    def _finish(self, status, message, pause=False):
        super()._finish(status, message, pause)
        self.close()

    def _frame(self, camera):
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = self.renderer.render().copy()
        return torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255

    def _needs_fresh_frames(self):
        """True when the next select_action() will actually look at the pixels.

        SmolVLA is an action-chunk policy: select_action() runs the real forward
        pass only when its internal action queue is empty (chunk_size=50 here),
        and otherwise just pops an already-predicted action - the images in the
        batch are normalized (IDENTITY for this checkpoint) and then dropped. So
        49 out of every 50 queries were paying ~440ms to render two frames that
        nothing ever reads. Rendering only on the call that consumes them is
        bit-identical in behaviour and is the single biggest win on this laptop.

        Guarded, not assumed: _queues is lerobot-internal and this checkpoint
        already drifts from the installed lerobot (see _load_policy). If the
        attribute ever disappears, fall back to rendering every query - slow,
        but never wrong.
        """
        try:
            from lerobot.constants import ACTION
            return len(self.policy._queues[ACTION]) == 0
        except Exception:
            return True

    def update(self, targets):
        if not self.active:
            return
        if float(self.data.time) - self.started > self.duration_s:
            targets[:] = self.last_action
            # Grade the physical outcome. This used to report "succeeded" purely
            # because the timer expired, which meant a policy that never moved -
            # or that pushed the object away - still reported success to the
            # dashboard. Now the same gate the scripted teacher is held to
            # decides, and a run that cannot be scored says so.
            self.result = self._placement_result()
            if self.result is None:
                self._finish("completed", f'Ran the live policy for {self.duration_s:.0f}s on '
                             f'"{self.instruction}". No declared target for this instruction, so not scored.', True)
            elif self.result["passed"]:
                self._finish("succeeded", f'{self.result["object"].title()} placed within '
                             f'{self.result["placement_xy_error_mm"]:.1f}mm of its declared pose '
                             f'(gate: {_PLACEMENT_XY_M*1000:.0f}mm).', True)
            else:
                self._finish("failed", f'{self.result["object"].title()} ended '
                             f'{self.result["placement_xy_error_mm"]:.1f}mm from its declared pose '
                             f'(started {self.result["start_xy_error_mm"]:.1f}mm away, '
                             f'gate: {_PLACEMENT_XY_M*1000:.0f}mm).', True)
            return
        if self.tick % self.QUERY_EVERY_TICKS == 0:
            state = torch.tensor(np.r_[self.data.qpos[:12], self.data.qvel[:12]], dtype=torch.float32)[None]
            state_norm = (state - self.stats["observation.state.mean"]) / (self.stats["observation.state.std"] + 1e-8)
            if self._needs_fresh_frames() or self._frames is None:
                self._frames = (self._frame("overview"), self._frame(self.wrist_cam))
            batch = {
                "observation.images.camera1": self._frames[0],
                "observation.images.camera2": self._frames[1],
                "observation.state": state_norm,
                "observation.language.tokens": self.lang_tokens,
                "observation.language.attention_mask": self.lang_mask,
                "task": [self.instruction],
            }
            started = time.perf_counter()
            with torch.inference_mode():
                action = self.policy.select_action(batch)
            self.inference_ms.append((time.perf_counter() - started) * 1000)
            action_real = action * (self.stats["action.std"] + 1e-8) + self.stats["action.mean"]
            self.last_action = np.clip(action_real[0].numpy(),
                                        self.model.actuator_ctrlrange[:12, 0], self.model.actuator_ctrlrange[:12, 1])
        targets[:] = self.last_action
        self.tick += 1

    def snapshot(self):
        result = super().snapshot()
        result.update(object_id=None, stage_label="Live SmolVLA: camera + language -> motor targets",
                       policy_mode="learned_vla", instruction=getattr(self, "instruction", None),
                       # Which runtime actually executed the forward pass, reported the same
                       # way primitive_policy.py and bottle_refinement_runtime.py do it.
                       policy_details={"neural_runtime": _CACHE.get("runtime", "PyTorch"),
                                       "checkpoint": CHECKPOINT.name},
                       placement=getattr(self, "result", None),
                       inference_median_ms=float(np.median(self.inference_ms)) if self.inference_ms else None,
                       progress=(min(.99, (float(self.data.time) - self.started) / self.duration_s)
                                 if self.status == "running" else (1. if self.status == "succeeded" else 0.)))
        result["stages"] = [{"id": "neural_control", "label": "Live camera + language -> motor targets"}]
        return result
