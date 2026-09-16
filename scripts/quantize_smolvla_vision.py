"""INT8 post-training quantization of the SmolVLA vision tower, via NNCF.

The fp32 OpenVINO tower (scripts/export_smolvla_vision.py) is ~61% of what is
left of a forward pass. NNCF quantizes the *exported IR*, not the live PyTorch
graph - which is why this does not hit the ONEDNN "data type of input should be
float" failure that killed the earlier torch dynamic-quantization attempt (see
the ponytail: note in vla_task.py).

Calibration draws real rendered frames from the dinner scene, preprocessed
exactly as modeling_smolvla.prepare_images does (resize with padding to 512x512,
then [0,1] -> [-1,1] for SigLIP). Collecting calibration statistics fits model
parameters, so the seeds are required to come from the **training** partition -
duet_protocol.require_partition enforces that here the same way it does for
demonstration recording.

Quantization is lossy by construction, so the gate is not a tight numerical
budget like the fp32 export's. It reports tower-level error and leaves the
decision to the action-level check, whose only meaningful reference is the
model's own sampling noise (see OPTIMIZATIONS.md section 6).

Measured on the demo laptop: **41% slower** than FP32, because an i5-8265U is
AVX2-only and has no VNNI/DL Boost for OpenVINO to issue int8 dot products
against. Not part of the shipped path, and deliberately not in requirements.txt.
Kept because this is expected to flip on any Ice Lake or newer Intel CPU.

    pip install nncf
    python scripts/quantize_smolvla_vision.py
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

CALIBRATION_SEEDS = (1000, 1001, 1002, 1003)  # training partition, enforced below
IMAGE_SHAPE = (1, 3, 512, 512)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_frames(seeds, per_seed):
    """Real dinner-scene frames through the exact inference preprocessing."""
    import mujoco
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
    from simulation_lab.scene import build_scene

    samples = []
    for seed in seeds:
        xml, _ = build_scene(seed, practice=False, scenario="dinner", dinner_preset="task")
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        original = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        try:
            renderer = mujoco.Renderer(model, height=240, width=320)
        finally:
            model.vis.quality.offsamples = original
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        for index in range(per_seed):
            for _ in range(40):
                mujoco.mj_step(model, data)
            for camera in ("overview", "right_wrist_cam", "left_wrist_cam"):
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                rgb = renderer.render().copy()
                image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()/255
                samples.append((resize_with_pad(image, 512, 512, pad_value=0)*2-1).numpy())
            print({'seed': seed, 'step': index+1, 'frames': len(samples)}, flush=True)
        renderer.close()
    return samples


def run(args):
    import nncf
    from simulation_lab.duet_protocol import require_partition
    from simulation_lab.vla_task import CHECKPOINT, _CPU_THREADS
    from simulation_lab.vla_openvino import compile_vision_tower

    seeds = require_partition(args.seeds, 'training')
    folder = CHECKPOINT/'openvino'
    source = folder/'vision_tower.xml'
    if not source.exists():
        raise FileNotFoundError(f'Run scripts/export_smolvla_vision.py first: {source}')
    target = folder/'vision_tower_int8.xml'
    if target.exists() and not args.force:
        raise FileExistsError(f'Preserve earlier exports: {target} (pass --force to replace)')
    torch.set_num_threads(_CPU_THREADS)

    samples = calibration_frames(seeds, args.frames_per_seed)
    core = ov.Core()
    model = core.read_model(source)
    quantized = nncf.quantize(model, nncf.Dataset(samples, lambda item: item),
                              subset_size=len(samples), model_type=nncf.ModelType.TRANSFORMER)
    ov.save_model(quantized, target, compress_to_fp16=False)

    # Same input for both runtimes; the fp32 IR is the reference, not torch,
    # because fp32-IR parity against torch is already gated at export time.
    probe = samples[0]
    results, latencies = {}, {}
    for label, xml in (('fp32', source), ('int8', target)):
        compiled = compile_vision_tower(xml, args.device, _CPU_THREADS)
        request = compiled.create_infer_request()
        request.infer({0: probe})
        timings = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            outputs = request.infer({0: probe})
            timings.append((time.perf_counter()-started)*1000)
        results[label] = np.asarray(outputs[compiled.output(0)]).copy()
        latencies[label] = float(np.median(timings))

    absolute = float(np.abs(results['int8']-results['fp32']).max())
    relative = absolute/(float(np.abs(results['fp32']).max())+1e-9)
    result = {'checkpoint': CHECKPOINT.name, 'source_sha256': sha(Path(__file__)),
        'fp32_ir_sha256': sha(source), 'ir_sha256': {name: sha(folder/name)
            for name in ('vision_tower_int8.xml', 'vision_tower_int8.bin')},
        'calibration_seeds': list(seeds), 'calibration_partition': 'training',
        'calibration_frames': len(samples), 'input_shape': list(IMAGE_SHAPE),
        'fp32_median_ms': latencies['fp32'], 'int8_median_ms': latencies['int8'],
        'speedup': latencies['fp32']/latencies['int8'],
        'maximum_absolute_difference_vs_fp32': absolute, 'relative_difference_vs_fp32': relative,
        'bytes': {'fp32': (folder/'vision_tower.bin').stat().st_size,
                  'int8': (folder/'vision_tower_int8.bin').stat().st_size},
        'device': core.get_property(args.device, 'FULL_DEVICE_NAME'),
        'openvino': ov.__version__, 'nncf': nncf.__version__, 'threads': _CPU_THREADS,
        'scope': 'Vision tower only, INT8 post-training quantization calibrated on training-partition '
                 'renders. Lossy by construction: this file records tower-level error and latency, and '
                 'makes no task-success claim. The action-level effect must be judged against the '
                 "policy's own flow-matching sampling noise, with the noise seed pinned."}
    (folder/'parity_int8.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('fp32_median_ms', 'int8_median_ms', 'speedup',
        'maximum_absolute_difference_vs_fp32', 'relative_difference_vs_fp32', 'bytes')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(CALIBRATION_SEEDS))
    parser.add_argument('--frames-per-seed', type=int, default=4)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--force', action='store_true')
    run(parser.parse_args())
