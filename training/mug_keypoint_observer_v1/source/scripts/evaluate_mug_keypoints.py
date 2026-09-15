"""Once-only fresh perception and complete exposed physical-state regression."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import openvino as ov
from PIL import Image
import torch
from scripts.mug_keypoint_experiment import protocol, read, sha, space, summarize, write
from scripts.export_mug_keypoints import decode_ov, rows_from_network
from scripts.train_mug_keypoints import checked_data
from simulation_lab.mug_keypoint_observer import reconstruct


def frozen_export(p, protocol_path):
    raw = ROOT/p['raw_root']
    selection, parity = read(raw/'fit/selection.json'), read(raw/'openvino/parity.json')
    if not selection['development_gate_passed'] or not parity['passed'] or parity['protocol_sha256'] != sha(protocol_path):
        raise ValueError('A frozen successful development/export gate is required.')
    if sha(ROOT/selection['checkpoint']) != selection['checkpoint_sha256'] or parity['checkpoint_sha256'] != selection['checkpoint_sha256']:
        raise ValueError('Selected weights changed.')
    for name, digest in parity['ir_sha256'].items():
        if sha(raw/'openvino'/name) != digest:
            raise ValueError('The evaluated IR changed.')
    fit = read(raw/'fit/training.json')
    if sha(raw/'fit/training.json') != selection['training_report_sha256']:
        raise ValueError('The fitted candidate record changed.')
    for name, digest in fit['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A frozen trained/decoded dependency changed: '+name)
    return selection, parity


def run(args):
    p = protocol(args.protocol)
    raw = ROOT/p['raw_root']
    selection, parity = frozen_export(p, args.protocol)
    output = raw/('evaluation/perception.json' if args.mode == 'evaluation' else 'physical-regression.json')
    if output.exists():
        raise FileExistsError('Preserve every previous evaluation.')
    if args.mode == 'physical':
        fresh = read(raw/'evaluation/perception.json')
        if not fresh['summary']['gate_passed'] or fresh['checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise ValueError('Exposed physical-state regression follows successful fresh perception only.')
    preflight = space(p, output, 16*1024**2)
    torch.set_num_threads(2)
    core = ov.Core()
    compiled = core.compile_model(str(raw/'openvino/observer.xml'), 'CPU',
        {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
    started, inputs = time.perf_counter(), {}
    if args.mode == 'evaluation':
        audit = read(raw/'evaluation/audit.json')
        if not audit['passed'] or audit['recipes_sha256'] != sha(raw/'evaluation/recipes.npz'):
            raise ValueError('The complete fresh data must pass independent reconstruction first.')
        manifest, data = checked_data(raw/'evaluation', p, args.protocol)
        rows, _, latencies = rows_from_network(compiled, data, p)
        inputs[(raw/'evaluation/manifest.json').relative_to(ROOT).as_posix()] = sha(raw/'evaluation/manifest.json')
        inputs[(raw/'evaluation/recipes.npz').relative_to(ROOT).as_posix()] = sha(raw/'evaluation/recipes.npz')
        scope = 'Once-only new synthetic stereo perception, with all present/absent examples retained; no physics or control claim.'
    else:
        source = ROOT/p['evaluation']['physical_regression_manifest']
        visibility = read(source)
        if visibility['protocol_sha256'] != p['camera_protocol_sha256'] or len(visibility['states']) != 144:
            raise ValueError('The complete declared exposed physical-state set is required.')
        inputs[source.relative_to(ROOT).as_posix()] = sha(source)
        for name, digest in visibility['input_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('A preserved physical input changed: '+name)
        by_view = {(r['state_index'], r['configuration'], r['slot']): r for r in visibility['views']}
        order = list(read(ROOT/p['camera_protocol'])['camera_configurations'])
        image_row = order.index(p['camera_configuration'])
        rows, latencies, current = [], [], None
        for index, state in enumerate(visibility['states']):
            path = source.parent/state['mosaic']
            if sha(path) != state['mosaic_sha256']:
                raise ValueError('A preserved physical RGB mosaic changed.')
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
            pixels = np.asarray(Image.open(path).convert('RGB'))
            images = np.stack([pixels[image_row*262:image_row*262+240, slot*320:(slot+1)*320] for slot in p['camera_slots']])
            cameras = [by_view[index, p['camera_configuration'], slot]['calibration'] for slot in p['camera_slots']]
            begin = time.perf_counter()
            decoded = decode_ov(compiled([images.transpose(0, 3, 1, 2).astype(np.float32)/255]))
            observation = reconstruct(decoded, cameras, p['inference'])
            latencies.append((time.perf_counter()-begin)*1000)
            # Actual physics state is accessed only AFTER RGB-only inference.
            trace = ROOT/state['trace_root']/state['trace']
            if current != trace:
                current = trace
                model = mujoco.MjModel.from_xml_path(str(trace/'scene.xml'))
                data = mujoco.MjData(model)
                with np.load(trace/'states.npz', allow_pickle=False) as archive:
                    qpos = archive['qpos'].copy()
                for name in ('scene.xml', 'states.npz', 'report.json'):
                    inputs[(trace/name).relative_to(ROOT).as_posix()] = sha(trace/name)
            data.qpos[:] = qpos[state['frame_index']]
            mujoco.mj_forward(model, data)
            truth = data.body('mug').xpos+data.body('mug').xmat.reshape(3, 3)@np.asarray(p['target_midpoint_local_m'])
            recorded = state['configurations'][p['camera_configuration']]['scoring_only_midpoint_m']
            if not np.allclose(truth, recorded, rtol=0, atol=1e-12):
                raise ValueError('An independently reconstructed physical midpoint disagrees with its preserved record.')
            good = observation['status'] == 'observed'
            delta = np.asarray(observation['midpoint_m'])-truth if good else None
            rows.append({'index': index, 'trace': state['trace'], 'frame_index': state['frame_index'], 'phase': state['phase'],
                'observation': observation, 'scoring_only_present': True, 'scoring_only_midpoint_m': truth.tolist(),
                'error_3d_mm': float(np.linalg.norm(delta)*1000) if good else None,
                'error_xy_mm': float(np.linalg.norm(delta[:2])*1000) if good else None})
        scope = 'All 144 already-exposed contact-physics states scored from their original RGB mosaics. No fitting, new physical scene, control or generalization claim.'
    result = {'schema': p['schema'], 'mode': args.mode, 'protocol_sha256': sha(args.protocol),
        'source_sha256': sha(Path(__file__)), 'checkpoint_sha256': selection['checkpoint_sha256'],
        'selection_sha256': sha(raw/'fit/selection.json'), 'parity_sha256': sha(raw/'openvino/parity.json'),
        'ir_sha256': parity['ir_sha256'], 'input_sha256': inputs, 'summary': summarize(rows, p['perception_gate']), 'rows': rows,
        'wall_seconds': time.perf_counter()-started, 'inference_median_ms': float(np.median(latencies)), 'all_inference_ms': latencies,
        'device': core.get_property('CPU', 'FULL_DEVICE_NAME'), 'precision': str(compiled.get_property('INFERENCE_PRECISION_HINT')),
        'storage_preflight': preflight, 'new_physical_trials': 0, 'scope': scope}
    frozen_export(p, args.protocol)
    space(p, output, 16*1024**2)
    write(output, result)
    print({'mode': args.mode, 'summary': result['summary'], 'wall_seconds': result['wall_seconds'], 'inference_median_ms': result['inference_median_ms']}, flush=True)
    # A completed failed scientific gate is retained and stops later dependent work.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-keypoint-observer-v1.json')
    parser.add_argument('--mode', choices=['evaluation', 'physical'], required=True)
    run(parser.parse_args())
