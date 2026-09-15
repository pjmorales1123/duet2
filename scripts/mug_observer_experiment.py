"""Provenance, aggregate storage budget and honest perception scoring."""
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


def write_json(path, value):
    require_space(path, 4*1024**2)
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def load_protocol(path):
    p = read(path)
    if p['schema'] != 'talos.mug-visual-observer.v1':
        raise ValueError('Unexpected observer protocol.')
    if sha(ROOT/p['visibility_protocol']) != p['visibility_protocol_sha256']:
        raise ValueError('The fixed camera/feature reference changed.')
    if sha(ROOT/p['training_input']/'retrieval.npz') != p['training_input_sha256']:
        raise ValueError('The original mug training input changed.')
    return p


def space(p, destination, expected):
    roots = [ROOT/p[name] for name in ('raw_root', 'model_package', 'training_package', 'evidence_package')]
    used = sum(path.stat().st_size for root in roots if root.exists() for path in root.rglob('*') if path.is_file())
    used += sum(path.stat().st_size for path in (ROOT/'.run/final-goal').glob(roots[0].name+'*') if path.is_file())
    if used+expected > p['budget']['maximum_raw_and_packaged_mib']*1024**2:
        raise ValueError('Declared cumulative observer budget would be exceeded.')
    return require_space(destination, expected, p['budget']['reserve_gib']*1024**3)


def score(p, prediction_m, labels_m, present, usable):
    present, accepted = np.asarray(present, bool), np.asarray(usable) >= 2
    errors = np.linalg.norm(prediction_m-labels_m, axis=1)*1000
    xy = np.linalg.norm(prediction_m[:, :2]-labels_m[:, :2], axis=1)*1000
    target = accepted & present
    covered, count, false = int(target.sum()), int(present.sum()), int((accepted & ~present).sum())
    summary = {'states': len(present), 'present': count, 'present_accepted': covered,
        'coverage_fraction': covered/max(1, count), 'absent': int((~present).sum()), 'absent_false_accepts': false,
        'p95_3d_error_mm': float(np.quantile(errors[target], .95)) if covered else None,
        'maximum_3d_error_mm': float(errors[target].max()) if covered else None,
        'p95_xy_error_mm': float(np.quantile(xy[target], .95)) if covered else None}
    gate = p['perception_gate']
    summary['gate_passed'] = bool(covered and summary['coverage_fraction'] >= gate['minimum_present_coverage_fraction']
        and false <= gate['maximum_absent_false_accepts']
        and summary['p95_3d_error_mm'] <= gate['maximum_p95_3d_error_mm']
        and summary['maximum_3d_error_mm'] <= gate['maximum_3d_error_mm']
        and summary['p95_xy_error_mm'] <= gate['maximum_p95_xy_error_mm'])
    rows = [{'index': i, 'present': bool(present[i]), 'accepted': bool(accepted[i]), 'usable_views': int(usable[i]),
             'prediction_m': prediction_m[i].tolist() if accepted[i] else None,
             'scoring_only_label_m': labels_m[i].tolist() if present[i] else None,
             'error_3d_mm': float(errors[i]) if target[i] else None,
             'error_xy_mm': float(xy[i]) if target[i] else None} for i in range(len(present))]
    return {'summary': summary, 'rows': rows}
