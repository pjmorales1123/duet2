"""Live SmolVLA task: raw camera pixels + free-text instruction -> motor targets.

Unlike learned_task.py's bespoke single-skill PrimitivePolicy, this loads the
real fine-tuned LeRobot SmolVLA checkpoint and runs its actual pre/post
processing (rename, normalize, tokenize, unnormalize) by hand, because
lerobot's PreTrainedConfig.from_pretrained crashes under this project's
Python 3.14 (draccus can't build an argparser for one of SmolVLAConfig's
typed-dict fields) - see models/duet-smolvla-v1 for the checkpoint.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from .autonomy import LiftReturn
from .scene import HOME

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "duet-smolvla-v2"
_CACHE: dict = {}

# CPU-only, vendor-neutral speedups (no Intel/AMD-specific backend): leave a
# couple of cores for the physics thread and UI server, and trade a few of
# the flow-matching model's 10 default denoising steps for latency - fewer
# steps means a slightly less refined action chunk, not a wrong one.
_CPU_THREADS = max(1, (os.cpu_count() or 4) - 2)
_INFERENCE_STEPS = 5

# The teacher used a fixed arm per skill; the live VLA doesn't choose a side
# explicitly, so feed the wrist view the checkpoint actually trained on for
# whichever object the instruction names. ponytail: keyword match, not NLU -
# good enough since the training instructions themselves are this literal.
_WRIST_BY_KEYWORD = [
    ("fork", "right_wrist_cam"), ("mug", "right_wrist_cam"), ("bottle", "right_wrist_cam"),
    ("spoon", "left_wrist_cam"), ("plate", "left_wrist_cam"),
]


def _load_policy():
    if "policy" not in _CACHE:
        torch.set_num_threads(_CPU_THREADS)
        from lerobot.configs.types import FeatureType, PolicyFeature
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
        stats = load_file(CHECKPOINT / "policy_preprocessor_step_5_normalizer_processor.safetensors")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        _CACHE.update(policy=policy, stats=stats, tokenizer=tokenizer)
    return _CACHE["policy"], _CACHE["stats"], _CACHE["tokenizer"]


class SmolVLATask(LiftReturn):
    QUERY_EVERY_TICKS = 20  # ~10 Hz physics-tick cadence, matching the training dataset's frame rate
    DEFAULT_DURATION_S = 25.

    def start(self, instruction: str, duration_s: float | None = None):
        if self.active:
            raise ValueError("A goal is already running. Cancel it before starting another.")
        if self.layout.get("scenario") != "dinner":
            raise ValueError("The live VLA requires the dinner scene.")
        self.__init__(self.model, self.data, self.layout)
        policy, stats, tokenizer = _load_policy()
        self.policy, self.stats = policy, stats
        self.instruction = instruction.strip()
        self.wrist_cam = next((cam for kw, cam in _WRIST_BY_KEYWORD if kw in self.instruction.lower()), "right_wrist_cam")
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

    def update(self, targets):
        if not self.active:
            return
        if float(self.data.time) - self.started > self.duration_s:
            targets[:] = self.last_action
            self._finish("succeeded", f'Ran the live policy for {self.duration_s:.0f}s on "{self.instruction}".', True)
            return
        if self.tick % self.QUERY_EVERY_TICKS == 0:
            state = torch.tensor(np.r_[self.data.qpos[:12], self.data.qvel[:12]], dtype=torch.float32)[None]
            state_norm = (state - self.stats["observation.state.mean"]) / (self.stats["observation.state.std"] + 1e-8)
            batch = {
                "observation.images.camera1": self._frame("overview"),
                "observation.images.camera2": self._frame(self.wrist_cam),
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
                       inference_median_ms=float(np.median(self.inference_ms)) if self.inference_ms else None,
                       progress=(min(.99, (float(self.data.time) - self.started) / self.duration_s)
                                 if self.status == "running" else (1. if self.status == "succeeded" else 0.)))
        result["stages"] = [{"id": "neural_control", "label": "Live camera + language -> motor targets"}]
        return result
