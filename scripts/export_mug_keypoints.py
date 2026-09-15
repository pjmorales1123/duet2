"""Export a qualified stereo observer and recheck every development observation."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import openvino as ov
import torch
from safetensors.torch import load_file
from scripts.mug_keypoint_experiment import protocol, read, sha, space, summarize, write
from scripts.train_mug_keypoints import checked_data
from simulation_lab.mug_keypoint_observer import MugImagePointNet, decoded_outputs, reconstruct


def decode_ov(outputs):
    tensors = tuple(torch.from_numpy(outputs[index].copy()) for index in range(3))
    with torch.inference_mode():
        decoded = decoded_outputs(tensors)
    return {key: value.numpy() for key, value in decoded.items()}


def rows_from_network(compiled, data, p, expected_network=None):
    rows, parity, latencies = [], [], []
    for index in range(data['states']):
        images = data['rgb'][index*2:index*2+2].numpy().astype(np.float32)/255
        started = time.perf_counter()
        actual = compiled([images])
        latencies.append((time.perf_counter()-started)*1000)
        decoded = decode_ov(actual)
        if expected_network is not None:
            with torch.inference_mode():
                expected = decoded_outputs(expected_network(torch.from_numpy(images)))
            parity.append(float(np.max(abs(decoded['pixels']-expected['pixels'].numpy()))))
        observation = reconstruct(decoded, data['calibrations'][index], p['inference'])
        present = bool(data['present'][index])
        truth = data['world_points'][index].mean(0)
        good = present and observation['status'] == 'observed'
        delta = np.asarray(observation['midpoint_m'])-truth if good else None
        rows.append({'index': index, 'observation': observation, 'scoring_only_present': present,
                     'scoring_only_midpoint_m': truth.tolist() if present else None,
                     'error_3d_mm': float(np.linalg.norm(delta)*1000) if good else None,
                     'error_xy_mm': float(np.linalg.norm(delta[:2])*1000) if good else None})
        if (index+1) % 128 == 0:
            print({'scored_observations': index+1, 'total': data['states']}, flush=True)
    return rows, parity, latencies


def run(args):
    p = protocol(args.protocol)
    raw = ROOT/p['raw_root']
    output = raw/'openvino'
    if output.exists():
        raise FileExistsError('Preserve earlier exports.')
    selection, training = read(raw/'fit/selection.json'), read(raw/'fit/training.json')
    if selection['protocol_sha256'] != sha(args.protocol) or not selection['development_gate_passed']:
        raise ValueError('Export requires frozen successful development.')
    checkpoint = ROOT/selection['checkpoint']
    if sha(checkpoint) != selection['checkpoint_sha256'] or sha(raw/'fit/training.json') != selection['training_report_sha256']:
        raise ValueError('Selected training artifacts changed.')
    for split in ('training', 'development'):
        audit = read(raw/split/'audit.json')
        if not audit['passed'] or audit['recipes_sha256'] != sha(raw/split/'recipes.npz'):
            raise ValueError('Complete independent data/geometry audits are required before export.')
    for name, digest in training['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A frozen fit/inference source changed: '+name)
    preflight = space(p, output, 64*1024**2)
    manifest, development = checked_data(raw/'development', p, args.protocol)
    torch.set_num_threads(2)
    network = MugImagePointNet().eval()
    network.load_state_dict(load_file(str(checkpoint)))
    sample = torch.zeros(2, 3, 240, 320)
    traced = torch.jit.trace(network, sample)
    converted = ov.convert_model(traced, example_input=(sample,))
    output.mkdir(parents=True)
    space(p, output, 32*1024**2)
    ov.save_model(converted, output/'observer.xml', compress_to_fp16=False)
    core = ov.Core()
    compiled = core.compile_model(str(output/'observer.xml'), 'CPU',
        {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
    rows, errors, latencies = rows_from_network(compiled, development, p, network)
    summary = summarize(rows, p['perception_gate'])
    parity_passed = max(errors) <= p['export']['maximum_absolute_keypoint_parity_px']
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'checkpoint_sha256': sha(checkpoint),
        'selection_sha256': sha(raw/'fit/selection.json'), 'source_sha256': sha(Path(__file__)),
        'ir_sha256': {name: sha(output/name) for name in ('observer.xml', 'observer.bin')},
        'development_manifest_sha256': sha(raw/'development/manifest.json'), 'observations': len(rows),
        'maximum_absolute_keypoint_parity_px': max(errors), 'all_keypoint_parity_px': errors,
        'parity_passed': parity_passed, 'development': {'summary': summary, 'rows': rows},
        'passed': bool(parity_passed and summary['gate_passed']), 'inference_median_ms': float(np.median(latencies)),
        'all_inference_ms': latencies, 'device': core.get_property('CPU', 'FULL_DEVICE_NAME'),
        'execution_devices': list(compiled.get_property('EXECUTION_DEVICES')),
        'precision': str(compiled.get_property('INFERENCE_PRECISION_HINT')), 'openvino': ov.__version__, 'preflight': preflight,
        'scope': 'CPU FP32 numerical/export and exposed development check; no fresh perception, Intel execution or motor-control claim.'}
    space(p, output, 8*1024**2)
    write(output/'parity.json', result)
    print({key: result[key] for key in ('maximum_absolute_keypoint_parity_px', 'parity_passed', 'passed', 'inference_median_ms', 'device')}, flush=True)
    if not result['passed']:
        raise ValueError('Export/parity or CPU development gate failed; preserve the result.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-keypoint-observer-v1.json')
    run(parser.parse_args())
