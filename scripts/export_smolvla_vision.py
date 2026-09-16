"""Export SmolVLA's vision tower to OpenVINO IR and check numerical parity.

The tower is `connector(vision_model(pixel_values).last_hidden_state)` - exactly
what `SmolVLMWithExpertModel.embed_image()` computes, and ~88% of a forward pass
on a CPU-only machine. Only that subgraph is exported: the language model, the
action expert, the flow-matching denoiser, normalization and the tokenizer all
stay in PyTorch and are untouched by this script.

Why fp32 and not the checkpoint's bfloat16: the demo CPU (i5-8265U, Whiskey
Lake) has no bf16 instructions, so torch emulates every bf16 op. Measured 16907ms
(bf16) vs 1904ms (fp32) for one image on identical weights - an 8.9x emulation
tax. OpenVINO has no bf16 emulation path on this target either, so the IR is
saved with compress_to_fp16=False and run with INFERENCE_PRECISION_HINT f32,
matching every other OpenVINO artifact in this repo.

Writes models/<checkpoint>/openvino/{vision_tower.xml,.bin,parity.json}.
Refuses to overwrite an existing export, and fails loudly if parity regresses.

    python scripts/export_smolvla_vision.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import openvino as ov
import torch

# Parity budget. The reference is the SAME weights in torch fp32, so any
# difference here is OpenVINO graph-compilation rounding, not quantization.
# Measured 4.96e-05 on this tower; 1e-3 leaves headroom without hiding a real
# numerical break (bf16's own error against fp32 is 3.7e-01 for scale).
MAX_ABSOLUTE_PARITY = 1e-3
IMAGE_SHAPE = (1, 3, 512, 512)  # modeling_smolvla.py resizes with padding to config.resize_imgs_with_padding


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    from simulation_lab.vla_task import CHECKPOINT, _load_policy, _CPU_THREADS
    from simulation_lab.vla_openvino import VisionTower, compile_vision_tower, use_float32

    output = CHECKPOINT/'openvino'
    if output.exists() and not args.force:
        raise FileExistsError(f'Preserve earlier exports: {output} (pass --force to replace)')
    torch.set_num_threads(_CPU_THREADS)
    policy, _, _ = _load_policy()

    # Cast the whole policy first, then wrap. VisionTower holds *references* to
    # the policy's own submodules, so casting the tower would silently mutate
    # the policy anyway - doing it in this order makes that explicit and
    # guarantees the exported graph is the exact fp32 tower the runtime binds.
    use_float32(policy)
    tower = VisionTower(policy.model.vlm_with_expert).eval()
    sample = torch.rand(*IMAGE_SHAPE)*2-1  # prepare_images() hands over [-1,1], matching SigLIP
    with torch.inference_mode():
        started = time.perf_counter()
        reference = tower(sample)
        torch_ms = (time.perf_counter()-started)*1000

    converted = ov.convert_model(tower, example_input=sample)
    output.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, output/'vision_tower.xml', compress_to_fp16=False)

    compiled = compile_vision_tower(output/'vision_tower.xml', device=args.device, threads=_CPU_THREADS)
    request = compiled.create_infer_request()
    array = sample.numpy()
    request.infer({0: array})  # warm the kernels before timing
    latencies = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        actual = request.infer({0: array})
        latencies.append((time.perf_counter()-started)*1000)
    produced = torch.from_numpy(np.asarray(actual[compiled.output(0)]).copy())

    parity = float((produced-reference).abs().max())
    passed = parity <= MAX_ABSOLUTE_PARITY and produced.shape == reference.shape
    result = {'checkpoint': CHECKPOINT.name, 'checkpoint_sha256': sha(CHECKPOINT/'model.safetensors'),
        'source_sha256': sha(Path(__file__)), 'runtime_source_sha256': sha(ROOT/'simulation_lab/vla_openvino.py'),
        'ir_sha256': {name: sha(output/name) for name in ('vision_tower.xml', 'vision_tower.bin')},
        'input_shape': list(IMAGE_SHAPE), 'output_shape': list(produced.shape),
        'maximum_absolute_parity': parity, 'parity_budget': MAX_ABSOLUTE_PARITY, 'parity_passed': bool(passed),
        'torch_fp32_reference_ms': torch_ms, 'openvino_median_ms': float(np.median(latencies)),
        'all_inference_ms': latencies, 'device': ov.Core().get_property(args.device, 'FULL_DEVICE_NAME'),
        'execution_devices': list(compiled.get_property('EXECUTION_DEVICES')),
        'precision': str(compiled.get_property('INFERENCE_PRECISION_HINT')),
        'openvino': ov.__version__, 'torch': torch.__version__, 'threads': _CPU_THREADS,
        'scope': 'Vision tower (SigLIP encoder + connector) only, CPU FP32. The language model, action expert, '
                 'flow-matching denoiser, normalization and tokenizer remain in PyTorch. Numerical/export check '
                 'against identical fp32 weights; no task-success or Intel performance claim.'}
    (output/'parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('maximum_absolute_parity', 'parity_passed',
        'torch_fp32_reference_ms', 'openvino_median_ms', 'device', 'execution_devices')}, indent=2), flush=True)
    if not passed:
        raise ValueError(f'Vision tower parity failed: {parity:.3e} > {MAX_ABSOLUTE_PARITY:.0e}; preserve the result.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='CPU', help="OpenVINO device. 'GPU' shares the UHD 620 with MuJoCo's GL context.")
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--force', action='store_true', help='Replace an existing export.')
    run(parser.parse_args())
