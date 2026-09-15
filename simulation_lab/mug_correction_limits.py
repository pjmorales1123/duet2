"""Joint-limited neural local steps; no object state or inverse geometry input."""
import numpy as np
from .mug_correction_motor import assess_delta


def limited_delta(raw_delta, requested_translation, maximum_joint_delta):
    raw = np.asarray(raw_delta, dtype=float)
    requested = np.asarray(requested_translation, dtype=float)
    if raw.shape != (5,) or requested.shape != (3,) or not np.isfinite(raw).all() or not np.isfinite(requested).all():
        raise ValueError('Expected finite local joint and translation vectors.')
    if not 0 < maximum_joint_delta <= .02:
        raise ValueError('Unsupported local joint-step bound.')
    largest = float(np.max(abs(raw)))
    fraction = min(1., np.nextafter(float(maximum_joint_delta), 0.)/max(largest, 1e-30))
    return raw*fraction, requested*fraction, fraction


def assess_limited(geometry, joints, requested, raw_delta, support, guard, sizing):
    delta, effective, fraction = limited_delta(raw_delta, requested, sizing['maximum_joint_delta_rad'])
    row = assess_delta(geometry, joints, effective, delta, support, guard)
    row.update(original_requested_translation_m=np.asarray(requested).tolist(),
               raw_neural_joint_delta_rad=np.asarray(raw_delta).tolist(), step_fraction=fraction)
    row['accepted'] = bool(row['accepted'] and fraction >= sizing['minimum_fraction'])
    return row
