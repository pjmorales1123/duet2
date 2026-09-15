"""Explicit experimental target declarations; independent of robot actions."""
from copy import deepcopy
import numpy as np


def with_bottle_destination(layout, destination):
    point = np.asarray(destination, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite planar bottle destination.')
    result = deepcopy(layout)
    result['targets'] = [t for t in result['targets'] if t['object_id'] != 'bottle']
    result['targets'].append({'id': 'bottle_experiment_place', 'label': 'Experimental bottle destination',
                              'object_id': 'bottle', 'position_m': [*point.tolist(), result['table_z']], 'radius_m': .024})
    return result
