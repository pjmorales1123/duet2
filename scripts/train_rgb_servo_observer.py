"""Train/evaluate a bounded RGB keypoint observer on the RTX GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from simulation_lab.rgb_servo_network import BottleKeypointNet, observation_loss, decode_heatmaps
from simulation_lab.rgb_servo_cameras import triangulate, project
from simulation_lab.storage import require_space, GIB


def load_data(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['completed_states'] != manifest['states']:
        raise ValueError('Dataset is incomplete.')
    rows = {}
    for shard in manifest['shards']:
        path = folder/shard['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256']:
            raise ValueError('Dataset hash mismatch.')
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                rows.setdefault(key, []).append(data[key])
    arrays = {k: np.concatenate(v) for k, v in rows.items()}
    states, views = arrays['rgb'].shape[:2]
    return manifest, {
        'rgb': torch.from_numpy(arrays['rgb'].reshape(-1, 240, 320, 3).transpose(0, 3, 1, 2).copy()),
        'mask': torch.from_numpy(arrays['mask'].reshape(-1, 240, 320)),
        'keypoints': torch.from_numpy(arrays['keypoints_px'].reshape(-1, 2, 2)),
        'visible': torch.from_numpy(arrays['visible'].reshape(-1).astype('float32')),
        'world_points': arrays['world_points'], 'present': arrays['present'], 'states': states, 'views': views,
    }


def evaluate(network, dataset, manifest, device, batch_size):
    predictions, peaks, variances, visibility = [], [], [], []
    network.eval()
    with torch.inference_mode():
        for begin in range(0, len(dataset['rgb']), batch_size):
            images = dataset['rgb'][begin:begin+batch_size].to(device).float()/255
            maps, _, vis = network(images)
            xy, peak, variance = decode_heatmaps(maps)
            predictions.append(xy.cpu().numpy()); peaks.append(peak.cpu().numpy())
            variances.append(variance.cpu().numpy()); visibility.append(vis.sigmoid().cpu().numpy())
    predictions = np.concatenate(predictions)
    peaks = np.concatenate(peaks); variances = np.concatenate(variances); visibility = np.concatenate(visibility)
    # Fixed provisional confidence gate, declared in the first implementation.
    accepted = (visibility >= .7) & (peaks.min(1) >= .02) & (variances.max(1) <= 36.)
    truth_visible = dataset['visible'].numpy().astype(bool)
    errors = np.linalg.norm(predictions-dataset['keypoints'].numpy(), axis=2).max(1)
    error_quantiles = lambda x: {'median': float(np.median(x)), 'p95': float(np.quantile(x, .95)), 'max': float(np.max(x))} if len(x) else None
    shape = (dataset['states'], dataset['views'])
    xy = predictions.reshape(*shape, 2, 2); accepted_views = accepted.reshape(shape)
    matrices = [manifest['calibrations'][v]['projection'] for v in manifest['views']]
    triangulated, rows = [], []
    for index in range(dataset['states']):
        valid = np.flatnonzero(accepted_views[index])
        row = {'state': index, 'present': bool(dataset['present'][index]), 'accepted_views': valid.tolist(), 'accepted': False}
        if len(valid) >= 2:
            points = np.asarray([triangulate(xy[index, valid, k], [matrices[j] for j in valid]) for k in range(2)])
            residual = max(float(np.linalg.norm(project(points, matrices[j])-xy[index, j], axis=1).max()) for j in valid)
            axis = points[1]-points[0]; length = float(np.linalg.norm(axis))
            geometry_valid = residual <= 2. and .08 <= length <= .13 and axis[2] > .075
            row.update(reprojection_error_px=residual, axis_length_m=length, accepted=bool(geometry_valid), predicted_points_m=points.tolist())
            if dataset['present'][index]:
                error = float(np.linalg.norm(points-dataset['world_points'][index], axis=1).max()*1000)
                row['scoring_only_error_mm'] = error
                if geometry_valid:
                    triangulated.append(error)
        rows.append(row)
    result = {
        'states': dataset['states'], 'views': len(accepted), 'accepted_views': int(accepted.sum()),
        'visible_views': int(truth_visible.sum()), 'accepted_visible_views': int((accepted & truth_visible).sum()),
        'false_accepted_invisible_views': int((accepted & ~truth_visible).sum()),
        'pixel_error_visible': error_quantiles(errors[truth_visible]),
        'pixel_error_accepted_visible': error_quantiles(errors[accepted & truth_visible]),
        'accepted_present_states': sum(r['accepted'] and r['present'] for r in rows),
        'present_states': int(dataset['present'].sum()),
        'false_accepted_absent_states': sum(r['accepted'] and not r['present'] for r in rows),
        'triangulated_error_mm': error_quantiles(triangulated),
        'confidence_gate': {'visibility': .7, 'heatmap_peak_min': .02, 'variance_px2_max': 36., 'reprojection_px_max': 2., 'axis_length_m': [.08, .13]},
        'rows': rows,
    }
    network.train()
    return result


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the documented CUDA training environment.')
    if args.steps > 30000 or args.steps < 1:
        raise ValueError('Steps exceed the declared budget.')
    require_space(args.output, 256*1024**2)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026095403)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_manifest, train_data = load_data(args.train)
    if args.extra_train:
        extra_manifest, extra_data = load_data(args.extra_train)
        if extra_manifest['split'] != 'train':
            raise ValueError('Extra observations must be declared training data.')
        for key in ['rgb', 'mask', 'keypoints', 'visible']:
            train_data[key] = torch.cat([train_data[key], extra_data[key]])
        del extra_data
    dev_manifest, dev_data = load_data(args.development)
    if train_manifest['split'] != 'train' or dev_manifest['split'] != 'development':
        raise ValueError('Wrong dataset split.')
    device = torch.device('cuda')
    network = BottleKeypointNet().to(device)
    if args.warm_start:
        network.load_state_dict(load_file(str(args.warm_start), device='cuda'))
    if not 0 < args.learning_rate <= .001:
        raise ValueError('Learning rate is outside the bounded experiment range.')
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0001)
    manifest = {'schema': 'talos.rgb-servo-observer-training.v1', 'architecture': 'BottleKeypointNet',
                'parameters': sum(p.numel() for p in network.parameters()), 'device': torch.cuda.get_device_name(),
                'train': args.train.as_posix(), 'development': args.development.as_posix(), 'seed': 2026095403,
                'max_steps': args.steps, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'steps': [], 'pretrained_weights': False,
                'warm_start': args.warm_start.as_posix() if args.warm_start else None,
                'extra_training': args.extra_train.as_posix() if args.extra_train else None,
                'label_usage': 'Segmentation, body positions and visibility labels are training and scoring inputs only. Inference receives RGB.',
                'reserve_gib': 10}
    started = time.perf_counter()
    for step in range(1, args.steps+1):
        if time.perf_counter()-started > 30*60:
            manifest['stop_reason'] = 'Declared 30-minute budget reached.'
            break
        indices = torch.randint(len(train_data['rgb']), (args.batch_size,))
        rgb = train_data['rgb'][indices].to(device).float()/255
        keypoints = train_data['keypoints'][indices].to(device)
        masks = train_data['mask'][indices].to(device)
        visible = train_data['visible'][indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = observation_loss(network(rgb), keypoints, masks, visible)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
        optimizer.step(); schedule.step()
        peak = torch.cuda.max_memory_allocated()/GIB
        if peak > 10:
            raise RuntimeError('Declared VRAM budget exceeded.')
        if step % 100 == 0 or step == 1:
            row = {'step': step, 'loss': float(loss.detach()), **metrics,
                   'wall_seconds': time.perf_counter()-started, 'peak_vram_gib': peak}
            manifest['steps'].append(row); print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            require_space(args.output, 64*1024**2)
            folder = args.output/f'step-{step:06d}'; folder.mkdir()
            save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}, str(folder/'model.safetensors'))
            result = evaluate(network, dev_data, dev_manifest, device, args.batch_size)
            (folder/'development.json').write_text(json.dumps(result, indent=2)+'\n')
            row = {k: v for k, v in result.items() if k != 'rows'}
            print(json.dumps({'step': step, 'development': row}), flush=True)
            manifest['latest_checkpoint'] = folder.name
            manifest['completed_steps'] = step
            manifest['wall_seconds'] = time.perf_counter()-started
            manifest['peak_vram_gib'] = peak
            manifest['calibrations'] = dev_manifest['calibrations']
            (args.output/'training.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--extra-train', type=Path)
    parser.add_argument('--warm-start', type=Path)
    train(parser.parse_args())
