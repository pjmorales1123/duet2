"""Measure held-mug visibility in declared fixed cameras on exposed states."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from simulation_lab.dinner_vision import components
from simulation_lab.policy_cameras import camera_argument
from simulation_lab.rgb_servo_cameras import calibration, project, triangulate
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def space(protocol, destination, expected):
    roots = [ROOT/protocol[name] for name in ('raw_root', 'evidence_package')]
    used = sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file())
    used += sum(p.stat().st_size for p in (ROOT/'.run/final-goal').glob(roots[0].name+'*') if p.is_file())
    if used+expected > protocol['budget']['maximum_raw_and_packaged_mib']*1024**2:
        raise ValueError('Declared cumulative data budget would be exceeded.')
    return require_space(destination, expected, protocol['budget']['reserve_gib']*1024**3)


def fixed_camera(settings):
    if isinstance(settings, str):
        return camera_argument(settings)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = settings['lookat']
    for name in ('distance', 'azimuth', 'elevation'):
        setattr(camera, name, settings[name])
    return camera


def rgb_component(image, projection, method):
    """Image/calibration-only inference; truth is absent from this interface."""
    low, high = np.asarray(method['roi_world_box_m'])
    corners = np.asarray(list(itertools.product(*zip(low, high))))
    pixels = project(corners, projection)
    margin = method['roi_pixel_margin']
    height, width = image.shape[:2]
    left, top = np.maximum(np.floor(pixels.min(0))-margin, 0).astype(int)
    right, bottom = np.minimum(np.ceil(pixels.max(0))+margin+1, [width, height]).astype(int)
    roi = np.zeros((height, width), bool)
    roi[top:bottom, left:right] = True
    r, g, b = image.astype(float).transpose(2, 0, 1)
    colored = (g > 1.5*r) & (g > 1.015*b) & (b > 1.3*r) & (g > 45)
    candidates = components(colored & roi)
    candidates.sort(key=lambda p: (-len(p), int(p[:, 0].min()), int(p[:, 1].min())))
    mask = np.zeros_like(roi)
    if candidates:
        mask[tuple(candidates[0].T)] = True
        centroid = candidates[0].mean(0)[::-1].tolist()
    else:
        centroid = None
    return mask, {'centroid_px': centroid, 'rgb_pixels': int(mask.sum()),
                  'candidate_components': len(candidates), 'roi_xyxy': [int(left), int(top), int(right), int(bottom)]}


def verify_sources(protocol):
    hashes = {}
    for group in protocol['trace_groups']:
        folder = ROOT/group['root']
        freeze_path = folder/'frozen-inputs.json'
        if sha(freeze_path) != group['freeze_sha256']:
            raise ValueError('An exposed physical freeze changed.')
        hashes[freeze_path.relative_to(ROOT).as_posix()] = group['freeze_sha256']
        frozen = read(freeze_path)['inputs']
        # Verify runtime and asset bytes before re-rendering recorded state.
        for field in ('source_sha256', 'asset_sha256'):
            for name, digest in frozen.get(field, {}).items():
                if sha(ROOT/name) != digest:
                    raise ValueError('A frozen input changed: '+name)
                hashes[name] = digest
    return hashes


def run(args):
    protocol = read(args.protocol)
    output = ROOT/protocol['raw_root']
    if output.exists():
        raise FileExistsError('Preserve the earlier diagnostic and choose a new declared experiment.')
    inputs = verify_sources(protocol)
    free = space(protocol, output, 128*1024**2)
    output.mkdir(parents=True)
    started = time.perf_counter()
    views, states, predicted_masks, scoring_masks = [], [], [], []
    width, height = protocol['image_size']
    gate = protocol['visibility_gate']
    for group in protocol['trace_groups']:
        for seed in group['seeds']:
            for preset in protocol['presets']:
                if time.perf_counter()-started > protocol['budget']['maximum_wall_minutes']*60:
                    raise TimeoutError('Declared diagnostic compute budget reached.')
                space(protocol, output, 8*1024**2)
                trace = f'{seed}-{preset}-{protocol["controller"]}'
                folder = ROOT/group['root']/trace
                for name in ('scene.xml', 'states.npz', 'report.json'):
                    inputs[(folder/name).relative_to(ROOT).as_posix()] = sha(folder/name)
                report = read(folder/'report.json')
                original = next(r for r in report['task']['results'] if r['skill'] == 'mug')
                if protocol['require_zero_guard_waits'] and original['policy_details']['tracking_guard_wait_calls'] != 0:
                    raise ValueError('Time-aligned sampling requires no tracking-guard wait calls.')
                with np.load(folder/'states.npz', allow_pickle=False) as archive:
                    frames = {name: archive[name] for name in archive.files}
                mug_frames = np.flatnonzero(frames['stage'] == 'mug')
                origin = float(frames['time'][mug_frames[0]])
                model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
                model.vis.quality.offsamples = 0
                data = mujoco.MjData(model)
                mug_id = model.body('mug').id
                mug_geoms = np.flatnonzero(model.geom_bodyid == mug_id)
                option = mujoco.MjvOption()
                option.geomgroup[3:] = 0
                renderer = mujoco.Renderer(model, width=width, height=height)
                try:
                    for label, seconds in zip(protocol['sample_labels'], protocol['sample_seconds_from_first_mug_frame']):
                        space(protocol, output, 2*1024**2)
                        index = int(mug_frames[np.argmin(abs(frames['time'][mug_frames]-origin-seconds))])
                        if abs(float(frames['time'][index])-origin-seconds) > .031:
                            raise ValueError('A requested mug phase is absent from the trace.')
                        data.qpos[:] = frames['qpos'][index]
                        data.qvel[:] = frames['qvel'][index]
                        data.ctrl[:] = frames['targets'][index]
                        data.time = float(frames['time'][index])
                        mujoco.mj_forward(model, data)
                        state_index = len(states)
                        mosaic = Image.new('RGB', (width*3, (height+22)*3), '#0d1c2b')
                        drawing = ImageDraw.Draw(mosaic)
                        configuration_rows = {}
                        for row_index, (name, cameras) in enumerate(protocol['camera_configurations'].items()):
                            selected = []
                            for slot, settings in enumerate(cameras):
                                renderer.update_scene(data, camera=fixed_camera(settings), scene_option=option)
                                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                                image = renderer.render().copy()
                                camera = calibration(renderer, width, height)
                                predicted, observation = rgb_component(image, camera['projection'], protocol['rgb_method'])
                                # Privileged segmentation/geometry is used only AFTER image inference.
                                renderer.enable_segmentation_rendering()
                                segmentation = renderer.render().copy()
                                renderer.disable_segmentation_rendering()
                                truth = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 0], mug_geoms)
                                overlap = int((truth & predicted).sum())
                                precision = overlap/max(1, observation['rgb_pixels'])
                                recall = overlap/max(1, int(truth.sum()))
                                usable = bool(observation['rgb_pixels'] >= gate['minimum_rgb_component_pixels']
                                    and truth.sum() >= gate['minimum_ground_truth_visible_pixels']
                                    and precision >= gate['minimum_precision'] and recall >= gate['minimum_recall'])
                                item = {'state_index': state_index, 'configuration': name, 'slot': slot,
                                    'rgb_observation': observation, 'calibration': camera,
                                    'scoring_only': {'visible_pixels': int(truth.sum()), 'intersection_pixels': overlap,
                                        'precision': precision, 'recall': recall, 'usable': usable}}
                                views.append(item)
                                predicted_masks.append(np.packbits(predicted.flatten()))
                                scoring_masks.append(np.packbits(truth.flatten()))
                                if usable:
                                    selected.append(item)
                                mosaic.paste(Image.fromarray(image), (width*slot, (height+22)*row_index))
                                drawing.text((width*slot+5, (height+22)*row_index+height+4),
                                    f'{name}:{slot} RGB {observation["rgb_pixels"]} P {precision:.3f} R {recall:.3f}', fill='white')
                            accepted = len(selected) >= gate['minimum_usable_views_per_state']
                            midpoint, estimate, error = None, None, None
                            if accepted:
                                estimate = triangulate([r['rgb_observation']['centroid_px'] for r in selected],
                                                       [r['calibration']['projection'] for r in selected])
                                midpoint = data.body('mug').xpos+data.body('mug').xmat.reshape(3, 3)@np.asarray(protocol['centroid_diagnostic']['target_local_m'])
                                error = float(np.linalg.norm(estimate-midpoint)*1000)
                            area = sorted([r['rgb_observation']['rgb_pixels'] for r in selected], reverse=True)
                            configuration_rows[name] = {'usable_views': len(selected), 'visibility_gate_passed': accepted,
                                'second_largest_usable_component_pixels': area[1] if accepted else 0,
                                'centroid_triangulation_m': estimate.tolist() if estimate is not None else None,
                                'scoring_only_midpoint_m': midpoint.tolist() if midpoint is not None else None,
                                'scoring_only_centroid_error_mm': error}
                        path = output/f'{trace}-{label}.png'
                        space(protocol, path, 2*1024**2)
                        mosaic.save(path)
                        states.append({'trace': trace, 'trace_root': group['root'], 'frame_index': index,
                            'phase': label, 'skill_seconds': float(data.time-origin), 'mug_outcome': original['status'],
                            'mosaic': path.name, 'mosaic_sha256': sha(path), 'configurations': configuration_rows})
                finally:
                    renderer.close()
                print(json.dumps({'completed_trace': trace, 'states': len(states), 'views': len(views)}), flush=True)
    summaries = {}
    for name in protocol['camera_configurations']:
        subset = [state['configurations'][name] for state in states]
        accepted = [r for r in subset if r['visibility_gate_passed']]
        errors = [r['scoring_only_centroid_error_mm'] for r in accepted]
        summaries[name] = {'states': len(subset), 'accepted': len(accepted),
            'coverage_fraction': len(accepted)/len(subset),
            'median_second_largest_usable_component_pixels': float(np.median([r['second_largest_usable_component_pixels'] for r in accepted])) if accepted else 0,
            'gate_passed': bool(len(accepted)/len(subset) >= gate['minimum_state_coverage_fraction']),
            'scoring_only_centroid_error_p95_mm': float(np.quantile(errors, .95)) if errors else None,
            'scoring_only_centroid_error_max_mm': max(errors) if errors else None,
            'phase_coverage': {phase: sum(s['configurations'][name]['visibility_gate_passed'] for s in states if s['phase'] == phase) for phase in protocol['sample_labels']}}
    eligible = [name for name in summaries if summaries[name]['gate_passed']]
    selected = min(eligible, key=lambda name: (-summaries[name]['accepted'], -summaries[name]['median_second_largest_usable_component_pixels'], list(summaries).index(name))) if eligible else None
    space(protocol, output, 32*1024**2)
    with (output/'masks.npz').open('xb') as stream:
        np.savez_compressed(stream, predicted=np.asarray(predicted_masks), scoring_only=np.asarray(scoring_masks))
    result = {'schema': protocol['schema'], 'protocol_sha256': sha(args.protocol), 'script_sha256': sha(Path(__file__)),
        'parent_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'input_sha256': inputs, 'preflight': free, 'masks_sha256': sha(output/'masks.npz'),
        'summaries': summaries, 'selected_configuration': selected, 'states': states, 'views': views,
        'wall_seconds': time.perf_counter()-started, 'new_training_steps': 0, 'new_physical_trials': 0, 'scope': protocol['scope']}
    verify_sources(protocol)
    write_json(output/'report.json', result)
    for source, name in ((args.protocol, 'protocol.json'), (Path(__file__), Path(__file__).name)):
        space(protocol, output/name, source.stat().st_size+1024)
        with (output/name).open('xb') as stream:
            stream.write(source.read_bytes())
    space(protocol, output, 0)
    print(json.dumps({'summaries': summaries, 'selected_configuration': selected,
                      'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-visual-visibility-v1.json')
    run(parser.parse_args())
