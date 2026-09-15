"""Checks and result aggregation shared only by the mug keypoint experiment."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    require_space(path, 4*1024**2)
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def protocol(path):
    p = read(path)
    if p['schema'] != 'talos.mug-keypoint-observer.v1':
        raise ValueError('Unexpected keypoint protocol.')
    for field in ('camera_protocol', 'training_recipe', 'training_recipe_manifest', 'pose_recipe_protocol'):
        if sha(ROOT/p[field]) != p[field+'_sha256']:
            raise ValueError('A declared original input changed: '+field)
    if sha(ROOT/p['network']['architecture_source']) != p['network']['architecture_source_sha256']:
        raise ValueError('The frozen network architecture or loss changed.')
    return p


def space(p, destination, expected):
    roots = [ROOT/p[name] for name in ('raw_root', 'model_package', 'training_package', 'evidence_package')]
    used = sum(path.stat().st_size for root in roots if root.exists() for path in root.rglob('*') if path.is_file())
    used += sum(path.stat().st_size for path in (ROOT/'.run/final-goal').glob(roots[0].name+'*') if path.is_file())
    if used+expected > p['budget']['maximum_raw_and_packaged_gib']*1024**3:
        raise ValueError('Declared keypoint data/storage budget would be exceeded.')
    return require_space(destination, expected, p['budget']['reserve_gib']*1024**3)


def summarize(rows, gate):
    present = [r for r in rows if r['scoring_only_present']]
    accepted = [r for r in present if r['observation']['status'] == 'observed']
    false = sum(not r['scoring_only_present'] and r['observation']['status'] == 'observed' for r in rows)
    errors, xy = [r['error_3d_mm'] for r in accepted], [r['error_xy_mm'] for r in accepted]
    result = {'states': len(rows), 'present': len(present), 'present_accepted': len(accepted),
        'coverage_fraction': len(accepted)/max(1, len(present)), 'absent': len(rows)-len(present), 'absent_false_accepts': false,
        'p95_3d_error_mm': float(np.quantile(errors, .95)) if errors else None,
        'maximum_3d_error_mm': max(errors) if errors else None,
        'p95_xy_error_mm': float(np.quantile(xy, .95)) if xy else None}
    result['gate_passed'] = bool(errors and result['coverage_fraction'] >= gate['minimum_present_coverage_fraction']
        and false <= gate['maximum_absent_false_accepts'] and result['p95_3d_error_mm'] <= gate['maximum_p95_3d_error_mm']
        and result['maximum_3d_error_mm'] <= gate['maximum_3d_error_mm'] and result['p95_xy_error_mm'] <= gate['maximum_p95_xy_error_mm'])
    return result
