"""Fixed calibration for the experimental bottle observer, without object input."""
import mujoco
import numpy as np
from .scene import TABLE_Z

VIEWS = {
    'servo_overhead': {'azimuth': 90., 'elevation': -90., 'distance': .72},
    'servo_left': {'azimuth': 135., 'elevation': -55., 'distance': .72},
    'servo_right': {'azimuth': 45., 'elevation': -55., 'distance': .72},
}
KEYPOINTS = np.array([[0., 0., .025], [0., 0., .128]])


def camera_argument(name):
    settings = VIEWS[name]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.025, -.05, TABLE_Z + .085]
    for key, value in settings.items():
        setattr(camera, key, value)
    return camera


def calibration(renderer, width=320, height=240):
    cameras = renderer.scene.camera
    position = np.mean([c.pos for c in cameras], axis=0)
    forward = np.mean([c.forward for c in cameras], axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean([c.up for c in cameras], axis=0)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    c = cameras[0]
    scale = height * float(c.frustum_near) / float(c.frustum_top-c.frustum_bottom)
    rotation = np.stack([right, -up, forward])
    intrinsic = np.array([[scale, 0., width/2], [0., scale, height/2], [0., 0., 1.]])
    projection = intrinsic @ np.column_stack([rotation, -rotation@position])
    return {'projection': projection.tolist(), 'position': position.tolist(), 'image_shape': [height, width]}


def project(points, matrix):
    points = np.asarray(points)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def triangulate(pixels, matrices):
    """Linear calibrated triangulation. Inputs contain no simulator body state."""
    rows = []
    for (u, v), matrix in zip(pixels, matrices):
        matrix = np.asarray(matrix)
        rows.extend([u*matrix[2]-matrix[0], v*matrix[2]-matrix[1]])
    if len(rows) < 4:
        raise ValueError('At least two camera observations are required.')
    _, _, vh = np.linalg.svd(np.asarray(rows))
    if abs(vh[-1, 3]) < 1e-9:
        raise ValueError('Degenerate camera geometry.')
    return vh[-1, :3]/vh[-1, 3]
