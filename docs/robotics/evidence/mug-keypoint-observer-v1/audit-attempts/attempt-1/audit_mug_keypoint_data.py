"""Check all image bytes/point labels and regenerate every new development view."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.mug_keypoint_experiment import protocol, read, sha, space, write
from scripts.probe_mug_visual_visibility import fixed_camera
from simulation_lab.mug_keypoint_observer import reconstruct


def audit(args):
    p = protocol(args.protocol)
    folder = ROOT/p['raw_root']/args.split
    output = folder/'audit.json'
    if output.exists():
        raise FileExistsError('Preserve previous audits.')
    manifest = read(folder/'manifest.json')
    if manifest['protocol_sha256'] != sha(args.protocol) or sha(folder/'recipes.npz') != manifest['recipes_sha256']:
        raise ValueError('Declared data changed.')
    for name, digest in manifest['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A dataset input changed: '+name)
    preflight = space(p, output, 16*1024**2)
    with np.load(folder/'recipes.npz', allow_pickle=False) as archive:
        recipes = {name: archive[name].copy() for name in archive.files}
    count = p['data'][args.split+'_states']
    if len(recipes['qpos']) != count or manifest['completed_states'] != count:
        raise ValueError('Every declared state is required.')
    original_hashes = None
    if args.split == 'training':
        with np.load(ROOT/p['training_recipe'], allow_pickle=False) as original:
            for name in ('qpos', 'episode_index', 'exposure_scale', 'present'):
                if not np.array_equal(original[name], recipes[name]):
                    raise ValueError('The reused training recipe was altered: '+name)
            original_hashes = original['rgb_sha256'][:, p['camera_slots']].copy()
    checked_images, checked_pixels, mismatches = 0, 0, []
    started = time.perf_counter()
    for shard in manifest['shards']:
        path = folder/shard['file']
        if sha(path) != shard['sha256']:
            raise ValueError('A raw image shard changed.')
        with np.load(path, allow_pickle=False) as raw:
            a, b = shard['begin'], shard['end']
            for field in ('keypoints_px', 'visible', 'world_points', 'present'):
                if not np.array_equal(raw[field], recipes[field][a:b]):
                    mismatches.append([shard['file'], field])
            for local, index in enumerate(range(a, b)):
                for slot in range(2):
                    digest = hashlib.sha256(raw['rgb'][local, slot].tobytes()).hexdigest()
                    if digest != recipes['rgb_sha256'][index, slot] or (original_hashes is not None and digest != original_hashes[index, slot]):
                        mismatches.append([index, slot, 'image bytes'])
                    if not np.isin(raw['mask'][local, slot], [0, 1]).all() or int(raw['mask'][local, slot].sum()) != int(recipes['visible_pixels'][index, slot]):
                        mismatches.append([index, slot, 'mask storage'])
                    checked_images += 1
                    checked_pixels += raw['rgb'][local, slot].shape[0]*raw['rgb'][local, slot].shape[1]
    old = read(ROOT/p['pose_recipe_protocol'])
    camera_protocol = read(ROOT/p['camera_protocol'])
    camera_settings = [camera_protocol['camera_configurations'][p['camera_configuration']][slot] for slot in p['camera_slots']]
    maximum_world_difference, maximum_pixel_difference, maximum_oracle_error = 0., 0., 0.
    oracle_present, oracle_false_absent, rendered_views = 0, 0, 0
    for ep, name in enumerate(manifest['episodes']):
        indices = np.flatnonzero(recipes['episode_index'] == ep)
        if not len(indices):
            continue
        model = mujoco.MjModel.from_xml_path(str(ROOT/old['initial_scene_root']/name/'scene.xml'))
        model.vis.quality.offsamples = 0
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, width=320, height=240) if args.split != 'training' else None
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        matrices = [np.asarray(c['projection']) for c in manifest['calibration_by_episode'][name]]
        ids = np.flatnonzero(model.geom_bodyid == model.body('mug').id)
        try:
            for index in indices:
                data.qpos[:] = recipes['qpos'][index]
                data.qvel[:] = 0
                data.time = 0
                mujoco.mj_forward(model, data)
                body = data.body('mug')
                points = np.stack([body.xpos+body.xmat.reshape(3, 3)@point for point in np.asarray(p['keypoints_local_m'])])
                difference = float(np.max(abs(points-recipes['world_points'][index])))
                maximum_world_difference = max(maximum_world_difference, difference)
                if difference > 1e-12:
                    mismatches.append([int(index), 'world labels'])
                for slot, matrix in enumerate(matrices):
                    projected = np.concatenate((points, np.ones((2, 1))), axis=1)@matrix.T
                    pixels = projected[:, :2]/projected[:, 2:]
                    expected_visible = bool(recipes['present'][index] and recipes['visible_pixels'][index, slot] >= p['data']['minimum_visible_mug_pixels']
                                            and np.all(pixels >= 0) and np.all(pixels < [320, 240]))
                    label = pixels.astype(np.float32) if expected_visible else np.zeros((2, 2), np.float32)
                    error = float(np.max(abs(label-recipes['keypoints_px'][index, slot])))
                    maximum_pixel_difference = max(maximum_pixel_difference, error)
                    if error != 0 or expected_visible != bool(recipes['visible'][index, slot]):
                        mismatches.append([int(index), slot, 'projected label or visibility'])
                    if renderer is not None:
                        renderer.update_scene(data, camera=fixed_camera(camera_settings[slot]), scene_option=option)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                        image = np.round(renderer.render().astype(float)*recipes['exposure_scale'][index]).clip(0, 255).astype(np.uint8)
                        if hashlib.sha256(image.tobytes()).hexdigest() != recipes['rgb_sha256'][index, slot]:
                            mismatches.append([int(index), slot, 'independent RGB render'])
                        renderer.enable_segmentation_rendering()
                        segments = renderer.render().copy()
                        renderer.disable_segmentation_rendering()
                        mask = (segments[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segments[:, :, 0], ids)
                        if int(mask.sum()) != int(recipes['visible_pixels'][index, slot]):
                            mismatches.append([int(index), slot, 'independent visibility render'])
                        rendered_views += 1
                # Ideal labelled correspondences test the geometry decoder only.
                # These are never neural outputs, training metrics or control input.
                decoded = {'pixels': recipes['keypoints_px'][index], 'peaks': np.ones((2, 2)), 'variances': np.ones((2, 2))*11.52,
                           'visibility': recipes['visible'][index].astype(float), 'foreground_pixels': recipes['visible_pixels'][index]}
                observed = reconstruct(decoded, manifest['calibration_by_episode'][name], p['inference'])
                if recipes['present'][index] and observed['status'] == 'observed':
                    oracle_present += 1
                    maximum_oracle_error = max(maximum_oracle_error, float(np.linalg.norm(np.asarray(observed['midpoint_m'])-points.mean(0))))
                if not recipes['present'][index] and observed['status'] == 'observed':
                    oracle_false_absent += 1
        finally:
            if renderer is not None:
                renderer.close()
        print(f'{args.split}: audited episode {ep+1}/{len(manifest["episodes"])}', flush=True)
    if oracle_present != int(recipes['present'].sum()) or oracle_false_absent or maximum_oracle_error > .00001:
        mismatches.append(['ideal-correspondence geometry gate'])
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'source_sha256': sha(Path(__file__)),
        'split': args.split, 'recipes_sha256': sha(folder/'recipes.npz'), 'states': count,
        'verified_rgb_hashes': checked_images, 'verified_rgb_pixels': checked_pixels,
        'original_training_rgb_hashes_verified': checked_images if original_hashes is not None else 0,
        'new_split_rgb_views_independently_rendered': rendered_views,
        'maximum_geometric_label_difference_m': maximum_world_difference, 'maximum_projected_label_difference_px': maximum_pixel_difference,
        'ideal_geometry_only': {'present_accepted': oracle_present, 'absent_false_accepts': oracle_false_absent, 'maximum_midpoint_error_m': maximum_oracle_error},
        'mismatches': mismatches, 'passed': not mismatches, 'preflight': preflight, 'wall_seconds': time.perf_counter()-started,
        'scope': 'Dataset storage/geometry and ideal-correspondence decoder checks. Not neural inference accuracy, physical success or deployment evidence.'}
    space(p, output, 4*1024**2)
    write(output, result)
    print({key: result[key] for key in ('split', 'states', 'verified_rgb_hashes', 'new_split_rgb_views_independently_rendered',
                                      'maximum_geometric_label_difference_m', 'maximum_projected_label_difference_px', 'ideal_geometry_only', 'passed', 'wall_seconds')})
    if not result['passed']:
        raise ValueError('Independent data/geometry audit failed; preserve its complete outcome.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-keypoint-observer-v1.json')
    parser.add_argument('--split', choices=['training', 'development', 'evaluation'], required=True)
    audit(parser.parse_args())
