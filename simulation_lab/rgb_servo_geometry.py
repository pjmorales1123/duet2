"""Known-shape image reprojection fit; no simulator object state input."""
import numpy as np
from .rgb_servo_cameras import project


def rigid_bottle_keypoints(points, pixels, matrices, iterations=8):
    points = np.asarray(points, dtype=float); pixels = np.asarray(pixels, dtype=float)
    axis = points[1]-points[0]
    if points.shape != (2, 3) or axis[2] <= .075:
        raise ValueError('Expected two upright bottle keypoints.')
    parameters = np.r_[points[0], axis[:2]/axis[2]]

    def geometry(values):
        direction = np.r_[values[3:], 1.]; direction /= np.linalg.norm(direction)
        return np.stack([values[:3], values[:3]+.103*direction])

    def residual(values):
        candidate = geometry(values)
        return np.concatenate([(project(candidate, matrix)-observed).reshape(-1) for matrix, observed in zip(matrices, pixels)])

    initial_error = sum(float(np.square(project(points, matrix)-observed).sum()) for matrix, observed in zip(matrices, pixels))
    current = residual(parameters); initial_rigid_error = float(current@current)
    for _ in range(iterations):
        steps = np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4])
        jacobian = np.column_stack([(residual(parameters+np.eye(5)[i]*steps[i])-current)/steps[i] for i in range(5)])
        normal = jacobian.T@jacobian
        damping = np.diag(np.maximum(np.diag(normal), 1.))*1e-4
        delta = np.linalg.solve(normal+damping, -jacobian.T@current)
        delta = np.clip(delta, -np.array([.005, .005, .005, .05, .05]), np.array([.005, .005, .005, .05, .05]))
        if np.linalg.norm(delta) < 1e-9:
            break
        trial = parameters+delta; error = residual(trial)
        if float(error@error) >= float(current@current):
            break
        parameters, current = trial, error
    result = geometry(parameters)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite rigid reprojection fit.')
    return result, {'initial_unconstrained_squared_pixel_error': initial_error,
                    'initial_rigid_squared_pixel_error': initial_rigid_error,
                    'final_rigid_squared_pixel_error': float(current@current),
                    'known_keypoint_separation_m': .103}
