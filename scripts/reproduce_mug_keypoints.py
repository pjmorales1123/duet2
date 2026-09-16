"""Re-render all fresh stereo recipes and score packaged IR without raw files."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import openvino as ov
import torch
from scripts.mug_keypoint_experiment import read, sha, summarize, write
from scripts.probe_mug_visual_visibility import fixed_camera
from simulation_lab.mug_keypoint_observer import decoded_outputs, reconstruct
from simulation_lab.storage import require_space


def reproduce(args):
    if args.output.exists():
        raise FileExistsError('Preserve earlier reproduction results.')
    preflight = require_space(args.output, 16*1024**2)
    model_root, training_root, evidence_root = [ROOT/name for name in
        ('models/mug_keypoint_observer_v1', 'training/mug_keypoint_observer_v1', 'docs/robotics/evidence/mug-keypoint-observer-v1')]
    checked = 0
    for root in (model_root, training_root, evidence_root):
        for name, digest in read(root/'manifest.json')['files'].items():
            if sha(root/name) != digest:
                raise ValueError('A packaged artifact changed: '+name)
            checked += 1
    p = read(model_root/'protocol.json')
    for name, digest in read(training_root/'reproduction-inputs.json')['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Restore the packaged frozen source before reproduction: '+name)
    folder = training_root/'evaluation'
    manifest = read(folder/'manifest.json')
    with np.load(folder/'recipes.npz', allow_pickle=False) as archive:
        recipes = {name: archive[name].copy() for name in archive.files}
    camera_protocol = read(ROOT/p['camera_protocol'])
    camera_settings = [camera_protocol['camera_configurations'][p['camera_configuration']][slot] for slot in p['camera_slots']]
    old = read(ROOT/p['pose_recipe_protocol'])
    expected = read(evidence_root/'fresh-perception.json')
    torch.set_num_threads(2)
    core = ov.Core()
    compiled = core.compile_model(str(model_root/'openvino/observer.xml'), 'CPU',
        {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
    rows, mismatches, point_differences, latency = [], [], [], []
    hashes, started = 0, time.perf_counter()
    for ep, name in enumerate(manifest['episodes']):
        indices = np.flatnonzero(recipes['episode_index'] == ep)
        if not len(indices):
            continue
        model = mujoco.MjModel.from_xml_path(str(ROOT/old['initial_scene_root']/name/'scene.xml'))
        model.vis.quality.offsamples = 0
        data, renderer = mujoco.MjData(model), mujoco.Renderer(model, width=320, height=240)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        try:
            for index in indices:
                data.qpos[:] = recipes['qpos'][index]
                data.qvel[:] = 0
                data.time = 0
                mujoco.mj_forward(model, data)
                images = []
                for slot, settings in enumerate(camera_settings):
                    renderer.update_scene(data, camera=fixed_camera(settings), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    image = np.round(renderer.render().astype(float)*recipes['exposure_scale'][index]).clip(0, 255).astype(np.uint8)
                    if hashlib.sha256(image.tobytes()).hexdigest() != recipes['rgb_sha256'][index, slot]:
                        mismatches.append([int(index), slot, 'RGB hash'])
                    hashes += 1
                    images.append(image)
                begin = time.perf_counter()
                outputs = compiled([np.stack(images).transpose(0, 3, 1, 2).astype(np.float32)/255])
                latency.append((time.perf_counter()-begin)*1000)
                with torch.inference_mode():
                    decoded = {key: value.numpy() for key, value in decoded_outputs(tuple(torch.from_numpy(outputs[i].copy()) for i in range(3))).items()}
                observation = reconstruct(decoded, manifest['calibration_by_episode'][name], p['inference'])
                original = expected['rows'][index]
                point_differences.append(float(np.max(abs(decoded['pixels']-np.asarray(original['observation']['views']['pixels'])))))
                if observation['status'] != original['observation']['status']:
                    mismatches.append([int(index), 'acceptance'])
                present = bool(recipes['present'][index])
                truth = recipes['world_points'][index].mean(0)
                good = present and observation['status'] == 'observed'
                delta = np.asarray(observation['midpoint_m'])-truth if good else None
                rows.append({'index': int(index), 'observation': observation, 'scoring_only_present': present,
                    'scoring_only_midpoint_m': truth.tolist() if present else None,
                    'error_3d_mm': float(np.linalg.norm(delta)*1000) if good else None,
                    'error_xy_mm': float(np.linalg.norm(delta[:2])*1000) if good else None})
        finally:
            renderer.close()
    rows.sort(key=lambda r: r['index'])
    if [r['index'] for r in rows] != list(range(p['data']['evaluation_states'])):
        raise ValueError('Incomplete reproduction.')
    summary = summarize(rows, p['perception_gate'])
    passed = not mismatches and summary['gate_passed'] and max(point_differences) <= p['export']['maximum_absolute_keypoint_parity_px']
    require_space(args.output, 16*1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'schema': p['schema'], 'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(model_root/'protocol.json'),
        'checked_package_files': checked, 'reconstructed_rgb_views': hashes, 'raw_run_files_read': 0,
        'all_keypoint_difference_px': point_differences, 'maximum_keypoint_difference_px': max(point_differences),
        'summary': summary, 'rows': rows, 'mismatches': mismatches, 'passed': passed,
        'all_inference_ms': latency, 'median_inference_ms': float(np.median(latency)), 'wall_seconds': time.perf_counter()-started,
        'device': core.get_property('CPU', 'FULL_DEVICE_NAME'), 'precision': str(compiled.get_property('INFERENCE_PRECISION_HINT')),
        'preflight': preflight, 'new_physical_trials': 0,
        'scope': 'Packaged-data/IR reproduction of the already exposed fresh perception split, not another independent evaluation or physical control trial.'})
    print({'passed': passed, 'package_files': checked, 'rgb_views': hashes, 'maximum_keypoint_difference_px': max(point_differences), 'summary': summary}, flush=True)
    if not passed:
        raise ValueError('Reproduction failed; preserve the report.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    reproduce(parser.parse_args())
