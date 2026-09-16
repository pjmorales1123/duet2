"""Experimental learned differential motor map with a robot-only refusal guard."""
import mujoco
import numpy as np
import torch
from torch import nn


class MugDifferentialMotorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center', torch.zeros(5))
        self.register_buffer('scale', torch.ones(5))
        self.register_buffer('minimum', torch.zeros(5))
        self.register_buffer('maximum', torch.zeros(5))
        self.layers = nn.Sequential(nn.Linear(5, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 15))

    def coefficients(self, joints):
        return self.layers((joints-self.center)/self.scale).reshape(-1, 5, 3)*20

    def forward(self, joints, translation):
        return torch.bmm(self.coefficients(joints), translation.unsqueeze(2)).squeeze(2)


class MugRobotGeometry:
    """Forward geometry from explicitly supplied robot joints, never object state."""
    def __init__(self, model):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.body = model.body('right_gripper').id
        self.tool = np.array([-.006, 0, -.092])

    def pose(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (5,) or not np.isfinite(joints).all():
            raise ValueError('Expected five finite measured right-arm joints.')
        self.scratch.qpos[6:11] = joints
        mujoco.mj_kinematics(self.model, self.scratch)
        rotation = self.scratch.xmat[self.body].reshape(3, 3).copy()
        return self.scratch.xpos[self.body]+rotation@self.tool, rotation


def assess_delta(geometry, joints, requested, delta, support, guard):
    """A forward-only assessment. Never adjust a network output to pass."""
    joints, requested, delta = (np.asarray(value, dtype=float) for value in (joints, requested, delta))
    if joints.shape != (5,) or requested.shape != (3,) or delta.shape != (5,) or not all(np.isfinite(a).all() for a in (joints, requested, delta)):
        raise ValueError('Malformed local motor inputs/output.')
    point, rotation = geometry.pose(joints)
    endpoint = joints+delta
    actual, final_rotation = geometry.pose(endpoint)
    error = float(np.linalg.norm(actual-point-requested)*1000)
    axis_error = float(np.linalg.norm(final_rotation[:, 2]-rotation[:, 2]))
    limits = geometry.model.actuator_ctrlrange[6:11]
    valid = (np.linalg.norm(requested) <= guard['maximum_translation_norm_m']+1e-9 and abs(requested[2]) < 1e-12
        and np.all(joints >= np.asarray(support['minimum'])) and np.all(joints <= np.asarray(support['maximum']))
        and np.max(abs(delta)) <= guard['maximum_joint_step_rad'] and error <= guard['maximum_position_error_mm']
        and axis_error <= guard['maximum_axis_error'] and np.all(endpoint >= limits[:, 0]) and np.all(endpoint <= limits[:, 1]))
    return {'accepted': bool(valid), 'position_error_mm': error, 'axis_error': axis_error,
            'joints': joints.tolist(), 'requested_translation_m': requested.tolist(), 'joint_delta_rad': delta.tolist(),
            'actual_translation_m': (actual-point).tolist()}
