"""Render declared stereo mug-axis supervision from compact, explicit poses."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scripts.collect_mug_visual_observer import quaternion
from scripts.mug_keypoint_experiment import protocol, read, sha, space, write
from scripts.probe_mug_visual_visibility import fixed_camera
from simulation_lab.rgb_servo_cameras import calibration, project


def sample_poses(p, split, inherited):
    """Sample only the frozen pose distribution; no learned model is consulted."""
    old = read(ROOT/p['pose_recipe_protocol'])
    settings = old['data']
    count = p['data'][split+'_states']
    rng = np.random.default_rng(p['data'][split+'_rng_seed'])
    episode = rng.integers(0, len(inherited['episodes']), count)
    seconds = rng.uniform(*settings['nominal_seconds_range'], count)
    joints = rng.normal(size=(count, 5))*settings['right_joint_jitter_std_rad']
    gripper = rng.uniform(*settings['gripper_position_range_rad'], count)
    displacement = rng.uniform(-1, 1, (count, 3))*settings['mug_translation_jitter_m']
    angle = rng.uniform(-1, 1, (count, 3))*settings['mug_roll_pitch_yaw_jitter_rad']
    exposure = rng.uniform(*settings['exposure_scale_range'], count)
    present = np.arange(count) % settings['absent_every_nth_state'] != 0
    with np.load(ROOT/old['training_input']/'retrieval.npz', allow_pickle=False) as original:
        bounds, times, actions = (original[name] for name in ('bounds', 'seconds', 'actions'))
    qpos = None
    for ep, name in enumerate(inherited['episodes']):
        folder = ROOT/old['initial_scene_root']/name
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, np.load(folder/'mug/initial-integration-state.npy', allow_pickle=False), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        initial = data.qpos.copy()
        rotation = data.body('mug').xmat.reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        address = int(model.joint('mug_free').qposadr[0])
        if qpos is None:
            qpos = np.zeros((count, model.nq), np.float64)
        a, b = bounds[ep]
        for index in np.flatnonzero(episode == ep):
            data.qpos[:] = initial
            data.qpos[:12] = [np.interp(seconds[index], times[a:b], actions[a:b, j]) for j in range(12)]
            data.qpos[6:11] += joints[index]
            data.qpos[11] = gripper[index]
            mujoco.mj_forward(model, data)
            hand = data.body('right_gripper')
            tool = hand.xpos+hand.xmat.reshape(3, 3)@np.asarray(settings['tool_point_local_m'])
            position = tool-np.asarray(settings['nominal_tool_minus_mug_world_m'])+displacement[index]
            position[2] = np.clip(position[2], *settings['mug_bottom_z_range_m'])
            if not present[index]:
                position = np.asarray([2., 2., .76])
            data.qpos[address:address+3] = position
            data.qpos[address+3:address+7] = quaternion(angle[index]+[0, 0, yaw])
            mujoco.mj_forward(model, data)
            qpos[index] = data.qpos
    return {'qpos': qpos, 'episode_index': episode, 'exposure_scale': exposure, 'present': present,
            'nominal_seconds': seconds, 'joint_jitter_rad': joints, 'gripper_rad': gripper,
            'translation_jitter_m': displacement, 'euler_jitter_rad': angle}


def collect(args):
    p = protocol(args.protocol)
    output = ROOT/p['raw_root']/args.split
    if output.exists():
        raise FileExistsError('Preserve earlier image batches and their evidence.')
    if args.split == 'evaluation':
        selection = read(ROOT/p['raw_root']/'fit/selection.json')
        parity = read(ROOT/p['raw_root']/'openvino/parity.json')
        if selection['protocol_sha256'] != sha(args.protocol) or not selection['development_gate_passed'] or not parity['passed']:
            raise ValueError('Fresh perception requires frozen successful development and export.')
        if sha(ROOT/selection['checkpoint']) != selection['checkpoint_sha256'] or parity['checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise ValueError('Selected weights changed.')
    inherited = read(ROOT/p['training_recipe_manifest'])
    source_hashes = inherited['source_sha256'].copy()
    for name, digest in source_hashes.items():
        if sha(ROOT/name) != digest:
            raise ValueError('An inherited training/scene source changed: '+name)
    for path in [args.protocol, ROOT/p['camera_protocol'], ROOT/p['training_recipe'], ROOT/p['training_recipe_manifest'],
                 Path(__file__), ROOT/'scripts/mug_keypoint_experiment.py', ROOT/'simulation_lab/mug_keypoint_observer.py']:
        source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    preflight = space(p, output, 768*1024**2 if args.split == 'training' else 128*1024**2)
    output.mkdir(parents=True)
    started = time.perf_counter()
    if args.split == 'training':
        with np.load(ROOT/p['training_recipe'], allow_pickle=False) as source:
            rows = {name: source[name].copy() for name in ('qpos', 'episode_index', 'exposure_scale', 'present', 'nominal_seconds',
                     'joint_jitter_rad', 'gripper_rad', 'translation_jitter_m', 'euler_jitter_rad')}
            expected_rgb = source['rgb_sha256'][:, p['camera_slots']].copy()
    else:
        rows, expected_rgb = sample_poses(p, args.split, inherited), None
    count = p['data'][args.split+'_states']
    if len(rows['qpos']) != count:
        raise ValueError('Wrong number of declared recipes.')
    rgb = np.zeros((count, 2, 240, 320, 3), np.uint8)
    masks = np.zeros((count, 2, 240, 320), np.uint8)
    labels_px = np.zeros((count, 2, 2, 2), np.float32)
    visible = np.zeros((count, 2), bool)
    visible_pixels = np.zeros((count, 2), np.int32)
    world = np.zeros((count, 2, 3), np.float64)
    hashes = np.full((count, 2), '', dtype='U64')
    camera_protocol = read(ROOT/p['camera_protocol'])
    camera_settings = [camera_protocol['camera_configurations'][p['camera_configuration']][slot] for slot in p['camera_slots']]
    old = read(ROOT/p['pose_recipe_protocol'])
    calibrations = {}
    example_indices = {0, 1, 63, 127, 255, count-1}
    rendered = 0
    for ep, name in enumerate(inherited['episodes']):
        indices = np.flatnonzero(rows['episode_index'] == ep)
        if not len(indices):
            continue
        space(p, output, 16*1024**2)
        folder = ROOT/old['initial_scene_root']/name
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        model.vis.quality.offsamples = 0
        data = mujoco.MjData(model)
        mug_geoms = np.flatnonzero(model.geom_bodyid == model.body('mug').id)
        renderer = mujoco.Renderer(model, width=320, height=240)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        try:
            for index in indices:
                if time.perf_counter()-started > p['budget']['maximum_generation_minutes_per_split']*60:
                    raise TimeoutError('Declared data-generation time budget reached.')
                space(p, output, 2*1024**2)
                data.qpos[:] = rows['qpos'][index]
                data.qvel[:] = 0
                data.time = 0
                mujoco.mj_forward(model, data)
                body = data.body('mug')
                world[index] = np.asarray(p['keypoints_local_m'])@body.xmat.reshape(3, 3).T+body.xpos
                camera_rows = []
                for slot, settings in enumerate(camera_settings):
                    renderer.update_scene(data, camera=fixed_camera(settings), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    image = np.round(renderer.render().astype(float)*rows['exposure_scale'][index]).clip(0, 255).astype(np.uint8)
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if expected_rgb is not None and digest != expected_rgb[index, slot]:
                        raise ValueError('A reused training view failed its original RGB hash.')
                    rgb[index, slot], hashes[index, slot] = image, digest
                    camera = calibration(renderer)
                    camera_rows.append(camera)
                    points = project(world[index], camera['projection'])
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    mask = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 0], mug_geoms)
                    masks[index, slot] = mask
                    visible_pixels[index, slot] = int(mask.sum())
                    is_visible = bool(rows['present'][index] and mask.sum() >= p['data']['minimum_visible_mug_pixels']
                                      and np.all(points >= 0) and np.all(points < [320, 240]))
                    visible[index, slot] = is_visible
                    # Invisible image points carry no regression loss; retain world truth separately.
                    labels_px[index, slot] = points if is_visible else 0
                calibrations[name] = camera_rows
                if index in example_indices:
                    canvas = Image.new('RGB', (640, 266), '#0d1c2b')
                    draw = ImageDraw.Draw(canvas)
                    for slot in range(2):
                        canvas.paste(Image.fromarray(rgb[index, slot]), (320*slot, 0))
                        if visible[index, slot]:
                            for point, color in zip(labels_px[index, slot], ('#ff4991', '#f4e45d')):
                                x, y = point+[320*slot, 0]
                                draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
                    draw.text((8, 247), f'{args.split} {index}: posed supervision; present={bool(rows["present"][index])}', fill='white')
                    space(p, output, 1024**2)
                    canvas.save(output/f'example-{index:04d}.png')
                rendered += 1
        finally:
            renderer.close()
        print(f'{args.split}: rendered {rendered}/{count} stereo states', flush=True)
    shards = []
    for begin in range(0, count, p['data']['states_per_shard']):
        end = min(count, begin+p['data']['states_per_shard'])
        path = output/f'rgb-{begin:04d}-{end:04d}.npz'
        space(p, path, (end-begin)*2*240*320*4+2*1024**2)
        with path.open('xb') as stream:
            np.savez_compressed(stream, rgb=rgb[begin:end], mask=masks[begin:end], keypoints_px=labels_px[begin:end],
                                visible=visible[begin:end], world_points=world[begin:end], present=rows['present'][begin:end])
        shards.append({'file': path.name, 'begin': begin, 'end': end, 'sha256': sha(path), 'bytes': path.stat().st_size})
    space(p, output/'recipes.npz', 16*1024**2)
    with (output/'recipes.npz').open('xb') as stream:
        np.savez_compressed(stream, **rows, world_points=world, keypoints_px=labels_px, visible=visible,
                            visible_pixels=visible_pixels, rgb_sha256=hashes)
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'split': args.split, 'states': count,
        'completed_states': rendered, 'episodes': inherited['episodes'], 'calibration_by_episode': calibrations,
        'source_sha256': source_hashes, 'recipes_sha256': sha(output/'recipes.npz'), 'shards': shards,
        'present': int(rows['present'].sum()), 'absent': int((~rows['present']).sum()), 'visible_views': int(visible.sum()),
        'original_training_rgb_hashes_verified': count*2 if args.split == 'training' else 0,
        'preflight': preflight, 'wall_seconds': time.perf_counter()-started,
        'parent_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'Posed stereo RGB supervision with offline point/segmentation labels only. No new physics or manipulation claim.'}
    space(p, output, 4*1024**2)
    write(output/'manifest.json', result)
    print({key: result[key] for key in ('split', 'states', 'present', 'absent', 'visible_views', 'original_training_rgb_hashes_verified', 'wall_seconds')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-keypoint-observer-v1.json')
    parser.add_argument('--split', choices=['training', 'development', 'evaluation'], required=True)
    collect(parser.parse_args())
