"""OpenVINO execution adapter for SmolVLA's vision tower.

Only the vision tower executes through OpenVINO. The language model, action
expert, flow-matching denoiser, image resize/normalization, tokenizer and the
action post-processing all remain explicit PyTorch in `vla_task.py` - this
module never produces a motor target and never sees simulator state.

The tower is exactly `connector(vision_model(pixel_values).last_hidden_state)`,
which is what `SmolVLMWithExpertModel.embed_image()` computes and, measured on
the demo laptop, ~88% of a forward pass. Binding replaces that one method on one
policy instance; every other code path is untouched.

Two independent wins, layered so the second is optional:

1. `use_float32()` - the checkpoint ships bfloat16 and this CPU (i5-8265U) has
   no bf16 instructions, so torch emulates every op. fp32 runs the *same
   weights* 8.9x faster and strictly more accurately.
2. `bind_openvino_vision()` - runs the fp32 tower through OpenVINO CPU FP32,
   faster again, gated on a frozen IR whose parity was checked at export time
   (`scripts/export_smolvla_vision.py`).

Both are no-ops that fall back to stock PyTorch if anything is missing, so a
checkout without an exported IR still runs - just slower.
"""
import json
from pathlib import Path
import numpy as np
import openvino as ov
import torch


def compile_vision_tower(xml, device='CPU', threads=None):
    """Compile a vision-tower IR with this repo's standard OpenVINO options.

    Matches `rgb_servo_openvino.py` / `bottle_refinement_runtime.py`: LATENCY
    hint, explicit f32 precision (never let the plugin silently pick fp16 - the
    iGPU does, and it costs ~9e-01 of accuracy on this graph). Unlike those
    small observers, the thread count is *not* pinned to 2: this tower is the
    dominant cost of a query and gets the same budget as the torch path.
    """
    options = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'}
    if device == 'CPU' and threads:
        options['INFERENCE_NUM_THREADS'] = int(threads)
    return ov.Core().compile_model(str(xml), device, options)


class VisionTower(torch.nn.Module):
    """`embed_image` as a standalone module, so it can be traced and exported."""

    def __init__(self, vlm_with_expert):
        super().__init__()
        vlm = vlm_with_expert.get_vlm_model()
        self.vision_model, self.connector = vlm.vision_model, vlm.connector

    def forward(self, pixel_values):
        return self.connector(self.vision_model(pixel_values=pixel_values).last_hidden_state)


class OpenVinoVisionTower:
    """Drop-in replacement for `SmolVLMWithExpertModel.embed_image`."""

    def __init__(self, folder, device='CPU', threads=None):
        folder = Path(folder)
        self.parity = json.loads((folder/'parity.json').read_text(encoding='utf-8-sig'))
        if not self.parity.get('parity_passed'):
            raise ValueError('Refusing an OpenVINO vision tower whose export parity did not pass.')
        self.compiled = compile_vision_tower(folder/'vision_tower.xml', device, threads)
        self.request = self.compiled.create_infer_request()
        self.device = device
        self.calls = 0

    def __call__(self, image):
        # The caller (modeling_smolvla.py) hands over float [-1,1] at the shape
        # the export pinned. Anything else is a silent accuracy bug, so refuse
        # it rather than let OpenVINO reshape underneath us.
        expected = tuple(self.parity['input_shape'])
        if tuple(image.shape) != expected:
            raise ValueError(f'OpenVINO vision tower expects {expected}, got {tuple(image.shape)}.')
        array = image.detach().cpu().float().numpy()
        outputs = self.request.infer({0: array})
        self.calls += 1
        return torch.from_numpy(np.asarray(outputs[self.compiled.output(0)]).copy())


def use_float32(policy):
    """Cast the policy out of bfloat16, which this CPU has to emulate.

    Every dtype cast inside lerobot's SmolVLA is relative to the weight dtype
    (`.to(dtype=layer.self_attn.q_proj.weight.dtype)`), so this propagates
    cleanly; only `smolvlm_with_expert.py`'s load-time `torch_dtype="bfloat16"`
    pins it, and that has already happened by the time we get here. fp32 is also
    the more accurate of the two - bf16 carries ~3 decimal digits.
    """
    policy.float()
    return policy


def bind_openvino_vision(policy, folder, device='CPU', threads=None):
    """Route `embed_image` through OpenVINO. Returns the adapter, or None.

    Shadows the bound method on this one instance - `nn.Module.__setattr__`
    passes plain callables through to the instance dict - so no other policy,
    and no other lerobot user in-process, is affected. Any failure (missing
    export, OpenVINO device gone, parity not recorded) leaves the stock PyTorch
    path in place: slower, never wrong.
    """
    folder = Path(folder)
    if not (folder/'vision_tower.xml').exists() or not (folder/'parity.json').exists():
        return None
    try:
        adapter = OpenVinoVisionTower(folder, device, threads)
    except Exception:
        return None
    policy.model.vlm_with_expert.embed_image = adapter
    return adapter
