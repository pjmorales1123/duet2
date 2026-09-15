"""Known mug-axis geometry and a bounded visual correction request, RGB only."""
import numpy as np


def mug_origin(observation):
    if observation.get('status')!='observed':
        raise ValueError('The mug observer refused this image pair.')
    points=np.asarray(observation['points_m'],dtype=float)
    if points.shape!=(2,3) or not np.isfinite(points).all():
        raise ValueError('Two finite semantic mug points are required.')
    return points[0]-(points[1]-points[0])*(.003/.058)


def correction_request(observation,settings):
    origin=mug_origin(observation)
    shift=np.r_[np.asarray(settings['goal_xy_m'])-origin[:2],0.]*settings['feedback_gain']
    shift*=min(1.,settings['maximum_requested_translation_m']/max(float(np.linalg.norm(shift)),1e-30))
    return origin,shift
