"""Calibrated fixed workspace views; no object tracking or simulator pose input."""
import mujoco
import numpy as np
from .scene import TABLE_Z

VIEWS = {
    'table_left': {'lookat': [-.22, .025, TABLE_Z+.045], 'distance': .55, 'azimuth': 135., 'elevation': -55.},
    'table_right': {'lookat': [.16, -.025, TABLE_Z+.045], 'distance': .65, 'azimuth': 45., 'elevation': -55.},
}


def camera_argument(name):
    if name not in VIEWS: return name
    view = VIEWS[name]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = view['lookat']
    camera.distance = view['distance']
    camera.azimuth = view['azimuth']
    camera.elevation = view['elevation']
    return camera
