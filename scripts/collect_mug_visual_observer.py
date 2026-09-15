"""Create compact, reproducible posed-render data for the declared mug observer."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from simulation_lab.mug_visual_observer import encode_images, FEATURE_DIMENSIONS
from simulation_lab.rgb_servo_cameras import calibration
from scripts.mug_observer_experiment import load_protocol, read, sha, space, write_json
from scripts.probe_mug_visual_visibility import fixed_camera


def quaternion(euler):
    roll, pitch, yaw = np.asarray(euler)/2
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.asarray([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                       cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def collect(args):
    p = load_protocol(args.protocol)
    output = ROOT/p['raw_root']/args.split
    if output.exists():
        raise FileExistsError('Preserve earlier datasets.')
    if args.split == 'evaluation':
        selection = read(ROOT/p['raw_root']/'fit/selection.json')
        parity = read(ROOT/p['raw_root']/'openvino/parity.json')
        if selection['protocol_sha256'] != sha(args.protocol) or not selection['development_gate_passed'] or not parity['passed']:
            raise ValueError('Fresh data requires frozen successful development and export.')
        if sha(ROOT/selection['checkpoint']) != selection['checkpoint_sha256']:
            raise ValueError('Selected observer changed.')
    preflight = space(p, output, 64*1024**2)
    source = ROOT/p['training_input']
    meta = read(source/'retrieval.json')
    with np.load(source/'retrieval.npz', allow_pickle=False) as archive:
        actions, seconds, bounds = (archive[n].copy() for n in ('actions', 'seconds', 'bounds'))
    originals = read(source/'input-replay.json')
    if not originals['passed'] or originals['source_sha256'] != p['training_input_sha256'] or len(meta['episodes']) != 41:
        raise ValueError('All 41 preserved original mug input replays are required.')
    visibility = read(ROOT/p['visibility_protocol'])
    cameras = visibility['camera_configurations'][p['camera_configuration']]
    count = p['data'][args.split+'_states']
    rng = np.random.default_rng(p['data'][args.split+'_rng_seed'])
    episode_index = rng.integers(0, len(meta['episodes']), count)
    t = rng.uniform(*p['data']['nominal_seconds_range'], count)
    joint_jitter = rng.normal(size=(count, 5))*p['data']['right_joint_jitter_std_rad']
    gripper = rng.uniform(*p['data']['gripper_position_range_rad'], count)
    displacement = rng.uniform(-1, 1, (count, 3))*p['data']['mug_translation_jitter_m']
    euler_jitter = rng.uniform(-1, 1, (count, 3))*p['data']['mug_roll_pitch_yaw_jitter_rad']
    exposure = rng.uniform(*p['data']['exposure_scale_range'], count)
    present = np.arange(count) % p['data']['absent_every_nth_state'] != 0
    features = np.zeros((count, FEATURE_DIMENSIONS), np.float32)
    labels = np.zeros((count, 3), np.float64)
    usable = np.zeros(count, np.uint8)
    rgb_counts = np.zeros((count, 3), np.int32)
    rgb_sha256 = np.full((count, 3), '', dtype='U64')
    source_hashes = {str(path.relative_to(ROOT).as_posix()): sha(path) for path in
        [args.protocol, ROOT/p['visibility_protocol'], source/'retrieval.npz', source/'retrieval.json', source/'input-replay.json']}
    for name, digest in read(ROOT/'training/mug_release_v1/reproduction-inputs.json')['repository_asset_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A scene asset changed: '+name)
        source_hashes[name] = digest
    for path in [*sorted((ROOT/'simulation_lab').glob('*.py')), Path(__file__), ROOT/'scripts/mug_observer_experiment.py', ROOT/'scripts/probe_mug_visual_visibility.py']:
        source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    output.mkdir(parents=True)
    started, state_qpos, calibration_by_episode = time.perf_counter(), None, {}
    selected_images = {0, 1, 63, 127, 255, count-1}
    for episode, name in enumerate(meta['episodes']):
        indices = np.flatnonzero(episode_index == episode)
        if not len(indices):
            continue
        space(p, output, 8*1024**2)
        folder = ROOT/p['initial_scene_root']/name
        paths = [folder/'scene.xml', folder/'mug/initial-integration-state.npy', folder/'mug/manifest.json']
        for path in paths:
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        lineage = next(row for row in meta['lineage'] if row['episode'] == name)
        if sha(paths[2]) != lineage['manifest_sha256']:
            raise ValueError('Original demonstration lineage changed: '+name)
        model = mujoco.MjModel.from_xml_path(str(paths[0]))
        model.vis.quality.offsamples = 0
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, np.load(paths[1], allow_pickle=False), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        initial = data.qpos.copy()
        rotation = data.body('mug').xmat.reshape(3, 3)
        initial_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        address = int(model.joint('mug_free').qposadr[0])
        if state_qpos is None:
            state_qpos = np.zeros((count, model.nq), np.float64)
        a, b = bounds[episode]
        renderer = mujoco.Renderer(model, width=320, height=240)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        try:
            for index in indices:
                if time.perf_counter()-started > p['budget']['maximum_data_generation_minutes_per_split']*60:
                    raise TimeoutError('Declared rendering time budget reached; preserve the partial dataset.')
                space(p, output, 2*1024**2)
                data.qpos[:] = initial
                data.qvel[:] = 0
                target = np.asarray([np.interp(t[index], seconds[a:b], actions[a:b, j]) for j in range(12)])
                data.qpos[:12] = target
                data.qpos[6:11] += joint_jitter[index]
                data.qpos[11] = gripper[index]
                data.time = 0
                mujoco.mj_forward(model, data)
                hand = data.body('right_gripper')
                point = hand.xpos+hand.xmat.reshape(3, 3)@np.asarray(p['data']['tool_point_local_m'])
                position = point-np.asarray(p['data']['nominal_tool_minus_mug_world_m'])+displacement[index]
                position[2] = np.clip(position[2], *p['data']['mug_bottom_z_range_m'])
                if not present[index]:
                    position = np.asarray([2., 2., .76])
                angles = euler_jitter[index]+[0, 0, initial_yaw]
                data.qpos[address:address+3] = position
                data.qpos[address+3:address+7] = quaternion(angles)
                mujoco.mj_forward(model, data)
                state_qpos[index] = data.qpos
                images, calibrations = [], []
                for slot, settings in enumerate(cameras):
                    renderer.update_scene(data, camera=fixed_camera(settings), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    image = np.round(renderer.render().astype(float)*exposure[index]).clip(0, 255).astype(np.uint8)
                    images.append(image)
                    calibrations.append(calibration(renderer))
                    import hashlib
                    rgb_sha256[index, slot] = hashlib.sha256(image.tobytes()).hexdigest()
                calibration_by_episode[name] = calibrations
                features[index], observations = encode_images(images, calibrations, visibility['rgb_method'])
                usable[index] = sum(r['usable'] for r in observations)
                rgb_counts[index] = [r['rgb_pixels'] for r in observations]
                labels[index] = data.body('mug').xpos+data.body('mug').xmat.reshape(3, 3)@np.asarray(p['data']['target_local_m'])
                if index in selected_images:
                    mosaic = Image.new('RGB', (960, 265), '#0d1c2b')
                    for slot, image in enumerate(images):
                        mosaic.paste(Image.fromarray(image), (320*slot, 0))
                    ImageDraw.Draw(mosaic).text((8, 246), f'{args.split} {index}: posed render, present={bool(present[index])}, RGB views={int(usable[index])}', fill='white')
                    space(p, output/f'example-{index:04d}.png', 1024**2)
                    mosaic.save(output/f'example-{index:04d}.png')
        finally:
            renderer.close()
        print(f'{args.split}: rendered {name}, {int(np.count_nonzero(rgb_sha256[:, 0] != ""))}/{count} states', flush=True)
    if np.any(rgb_sha256 == '') or not np.isfinite(features).all():
        raise ValueError('Incomplete or invalid dataset.')
    space(p, output/'data.npz', 48*1024**2)
    with (output/'data.npz').open('xb') as stream:
        np.savez_compressed(stream, features=features, labels_m=labels, present=present, usable_views=usable,
            rgb_component_pixels=rgb_counts, qpos=state_qpos, episode_index=episode_index,
            exposure_scale=exposure, rgb_sha256=rgb_sha256, nominal_seconds=t,
            joint_jitter_rad=joint_jitter, gripper_rad=gripper, translation_jitter_m=displacement, euler_jitter_rad=euler_jitter)
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'split': args.split, 'states': count,
        'episodes': meta['episodes'], 'source_sha256': source_hashes, 'data_sha256': sha(output/'data.npz'),
        'calibration_by_episode': calibration_by_episode, 'preflight': preflight,
        'parent_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'present': int(present.sum()), 'present_accepted': int((present & (usable >= 2)).sum()),
        'absent': int((~present).sum()), 'absent_false_accepts': int((~present & (usable >= 2)).sum()),
        'wall_seconds': time.perf_counter()-started,
        'scope': 'Synthetic posed images; no physics stepping or physical grasp/success claim. All numeric examples and RGB hashes retained.'}
    write_json(output/'manifest.json', result)
    space(p, output, 0)
    print({key: result[key] for key in ('split', 'states', 'present', 'present_accepted', 'absent', 'absent_false_accepts', 'wall_seconds')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-visual-observer-v1.json')
    parser.add_argument('--split', choices=['training', 'development', 'evaluation'], required=True)
    collect(parser.parse_args())
