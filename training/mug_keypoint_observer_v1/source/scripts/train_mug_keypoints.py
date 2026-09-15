"""One preregistered stereo image-point fit and complete development selection."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import save_file
from scripts.mug_keypoint_experiment import protocol, read, sha, space, summarize, write
from scripts.train_rgb_servo_observer import load_data
from simulation_lab.mug_keypoint_observer import MugImagePointNet, predict_batch, reconstruct
from simulation_lab.rgb_servo_network import observation_loss


def checked_data(folder, p, protocol_path):
    manifest, data = load_data(folder)
    if manifest['protocol_sha256'] != sha(protocol_path) or data['states'] != p['data'][manifest['split']+'_states'] or data['views'] != 2:
        raise ValueError('A dataset does not match the declared stereo split.')
    if sha(folder/'recipes.npz') != manifest['recipes_sha256']:
        raise ValueError('The compact pose recipe changed.')
    for name, digest in manifest['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A data dependency changed: '+name)
    with np.load(folder/'recipes.npz', allow_pickle=False) as archive:
        indices = archive['episode_index'].copy()
    data['calibrations'] = [manifest['calibration_by_episode'][manifest['episodes'][int(i)]] for i in indices]
    return manifest, data


def evaluate(network, data, p, device='cuda'):
    decoded = {}
    for begin in range(0, len(data['rgb']), p['training']['batch_size']):
        values = predict_batch(network, data['rgb'][begin:begin+p['training']['batch_size']], device)
        for name, value in values.items():
            decoded.setdefault(name, []).append(value)
    decoded = {name: np.concatenate(values).reshape(data['states'], 2, *values[0].shape[1:]) for name, values in decoded.items()}
    rows = []
    for index in range(data['states']):
        observed = reconstruct({name: values[index] for name, values in decoded.items()}, data['calibrations'][index], p['inference'])
        present = bool(data['present'][index])
        midpoint = data['world_points'][index].mean(0)
        good = present and observed['status'] == 'observed'
        delta = np.asarray(observed['midpoint_m'])-midpoint if good else None
        rows.append({'index': index, 'observation': observed, 'scoring_only_present': present,
            'scoring_only_midpoint_m': midpoint.tolist() if present else None,
            'error_3d_mm': float(np.linalg.norm(delta)*1000) if good else None,
            'error_xy_mm': float(np.linalg.norm(delta[:2])*1000) if good else None})
    return {'summary': summarize(rows, p['perception_gate']), 'rows': rows}


def train(args):
    p = protocol(args.protocol)
    raw = ROOT/p['raw_root']
    output = raw/'fit'
    if output.exists() or (raw/'evaluation').exists():
        raise FileExistsError('Preserve prior fits and do not fit after exposing evaluation.')
    if not torch.cuda.is_available():
        raise RuntimeError('This declared fit requires CUDA.')
    torch.set_num_threads(4)
    torch.manual_seed(p['training']['rng_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    training_manifest, training = checked_data(raw/'training', p, args.protocol)
    development_manifest, development = checked_data(raw/'development', p, args.protocol)
    preflight = space(p, output, 64*1024**2)
    output.mkdir(parents=True)
    train_tensors = {name: training[name].cuda() for name in ('rgb', 'mask', 'keypoints', 'visible')}
    del training
    network = MugImagePointNet().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=p['training']['learning_rate'], weight_decay=p['training']['weight_decay'])
    started, curve, candidates = time.perf_counter(), [], []
    for step in range(1, p['training']['updates']+1):
        if time.perf_counter()-started > p['budget']['maximum_fit_minutes']*60:
            raise TimeoutError('Declared CNN fit time exhausted; retain every partial checkpoint.')
        reserved = torch.cuda.max_memory_reserved()/1024**3
        if reserved > p['budget']['maximum_torch_reserved_gib']:
            raise MemoryError('Declared Torch reserved-memory cap exceeded.')
        fraction = (step-1)/(p['training']['updates']-1)
        lr = p['training']['final_learning_rate']+(p['training']['learning_rate']-p['training']['final_learning_rate'])*(1+math.cos(math.pi*fraction))/2
        for group in optimizer.param_groups:
            group['lr'] = lr
        ids = torch.randint(len(train_tensors['rgb']), (p['training']['batch_size'],), device='cuda')
        network.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = observation_loss(network(train_tensors['rgb'][ids].float()/255), train_tensors['keypoints'][ids],
                                         train_tensors['mask'][ids], train_tensors['visible'][ids])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite keypoint loss; no selection.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), p['training']['gradient_clip_norm'])
        optimizer.step()
        if step == 1 or step % 200 == 0:
            torch.cuda.synchronize()
            space(p, output, 16*1024**2)
            row = {'step': step, 'loss': float(loss.detach()), **metrics, 'learning_rate': lr,
                   'wall_seconds': time.perf_counter()-started, 'peak_torch_reserved_gib': torch.cuda.max_memory_reserved()/1024**3}
            curve.append(row)
            print(row, flush=True)
        if step in p['training']['checkpoints']:
            folder = output/f'step-{step:06d}'
            space(p, folder, 16*1024**2)
            folder.mkdir()
            checkpoint = folder/'model.safetensors'
            save_file({name: value.detach().cpu().contiguous() for name, value in network.state_dict().items()}, str(checkpoint))
            result = evaluate(network, development, p)
            result.update({'step': step, 'protocol_sha256': sha(args.protocol), 'checkpoint_sha256': sha(checkpoint),
                           'development_manifest_sha256': sha(raw/'development/manifest.json'), 'scoring_device': 'PyTorch CUDA FP32'})
            space(p, folder, 8*1024**2)
            write(folder/'development.json', result)
            candidates.append({'step': step, 'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
                               'checkpoint_sha256': sha(checkpoint), 'development': result['summary']})
            print({'checkpoint': step, 'development': result['summary']}, flush=True)
    torch.cuda.synchronize()
    qualified = [r for r in candidates if r['development']['gate_passed']]
    selected = min(qualified, key=lambda r: (r['development']['p95_3d_error_mm'], r['development']['maximum_3d_error_mm'], r['step'])) if qualified else None
    dependencies = [Path(__file__), ROOT/'scripts/mug_keypoint_experiment.py', ROOT/'scripts/collect_mug_keypoints.py',
                    ROOT/'scripts/train_rgb_servo_observer.py', ROOT/'simulation_lab/mug_keypoint_observer.py',
                    ROOT/'simulation_lab/rgb_servo_network.py', ROOT/'simulation_lab/rgb_servo_cameras.py']
    report = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'settings': p['training'],
        'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in dependencies},
        'dataset_manifest_sha256': {split: sha(raw/split/'manifest.json') for split in ('training', 'development')},
        'parent_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'completed_updates': p['training']['updates'], 'parameters': sum(v.numel() for v in network.parameters()),
        'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
        'peak_torch_reserved_gib': torch.cuda.max_memory_reserved()/1024**3, 'wall_seconds': time.perf_counter()-started,
        'curve': curve, 'candidates': candidates, 'development_gate_passed': bool(selected), 'preflight': preflight,
        'new_physical_trials': 0, 'scope': 'One synthetic stereo image-point fit; calibration is explicit and there is no motor-control integration.'}
    write(output/'training.json', report)
    selection = {'protocol_sha256': sha(args.protocol), 'training_report_sha256': sha(output/'training.json'),
        'development_gate_passed': bool(selected), 'checkpoint': selected['checkpoint'] if selected else None,
        'checkpoint_sha256': selected['checkpoint_sha256'] if selected else None, 'step': selected['step'] if selected else None,
        'next_step': 'FP32 CPU export/development check, then fresh perception' if selected else 'Stop before export, fresh perception and any motor integration.'}
    write(output/'selection.json', selection)
    space(p, output, 0)
    print({'selection': selection, 'wall_seconds': report['wall_seconds'], 'peak_torch_reserved_gib': report['peak_torch_reserved_gib']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-keypoint-observer-v1.json')
    train(parser.parse_args())
