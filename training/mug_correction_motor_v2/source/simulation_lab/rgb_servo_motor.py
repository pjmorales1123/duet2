"""Experimental neural Cartesian motor map; no object state or inverse solver.

Forward kinematics reads a scratch robot configuration for a refusal/progress
guard. It never optimizes or repairs a prediction. All motor endpoints originate
in the neural map; callers supply observed goals and measured robot joints.
"""
import numpy as np
import mujoco
import torch
from torch import nn
from safetensors.torch import load_file


class CartesianMotorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center', torch.tensor([.04, -.055, .931]))
        self.register_buffer('scale', torch.tensor([.20, .155, .048]))
        self.register_buffer('joint_center', torch.zeros(5))
        self.register_buffer('joint_scale', torch.ones(5))
        self.layers = nn.Sequential(nn.Linear(4, 256), nn.SiLU(), nn.Linear(256, 256),
                                    nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 5))

    def forward(self, points, sides):
        x = torch.cat(((points-self.center)/self.scale, sides.reshape(-1, 1)), dim=1)
        return self.layers(x)*self.joint_scale+self.joint_center


class RobotGeometry:
    """Robot-only FK from explicitly supplied joint coordinates."""
    def __init__(self, model):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.bodies = {side: model.body(side+'_gripper').id for side in ('left', 'right')}

    def pose(self, side, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (5,) or not np.isfinite(joints).all():
            raise ValueError('Five finite robot joint coordinates are required.')
        offset = 0 if side == 'left' else 6
        self.scratch.qpos[offset:offset+5] = joints
        mujoco.mj_kinematics(self.model, self.scratch)
        body = self.bodies[side]
        rotation = self.scratch.xmat[body].reshape(3, 3).copy()
        point = self.scratch.xpos[body]+rotation@np.array([.003, 0., -.092])
        return point.copy(), rotation


class CartesianMotorPolicy:
    def __init__(self, checkpoint, model, device='cpu'):
        self.device = torch.device(device)
        self.network = CartesianMotorNet().to(self.device).eval()
        self.network.load_state_dict(load_file(str(checkpoint), device=device))
        self.geometry = RobotGeometry(model)
        self.ranges = np.asarray(model.actuator_ctrlrange).copy()

    def predict(self, side, point):
        if side not in ('left', 'right'):
            raise ValueError('Select a known arm.')
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Three finite Cartesian coordinates are required.')
        center = self.network.center.cpu().numpy(); scale = self.network.scale.cpu().numpy()
        if np.any(np.abs((point-center)/scale) > 1.00001):
            raise ValueError('Cartesian goal lies outside the declared motor training box.')
        with torch.inference_mode():
            result = self.network(torch.tensor(point[None], device=self.device, dtype=torch.float32),
                                  torch.tensor([-1. if side == 'left' else 1.], device=self.device)).cpu().numpy()[0]
        offset = 0 if side == 'left' else 6
        limits = self.ranges[offset:offset+5]
        if not np.isfinite(result).all() or np.any(result < limits[:, 0]) or np.any(result > limits[:, 1]):
            raise ValueError('Neural motor prediction exceeds joint limits.')
        actual, rotation = self.geometry.pose(side, result)
        error = float(np.linalg.norm(actual-point))
        axis_error = float(np.linalg.norm(rotation[:, 1]-[0., 0., 1.]))
        if error > .0015 or axis_error > .025:
            raise ValueError(f'Neural motor geometry guard refused prediction ({error*1000:.3f} mm, axis {axis_error:.4f}).')
        return result.astype(float)

    def action(self, side, point, measured_joints, maximum_delta=.08):
        joints = np.asarray(measured_joints, dtype=float)
        if joints.shape != (5,) or not np.isfinite(joints).all():
            raise ValueError('Five measured joint positions are required.')
        if not 0 < maximum_delta <= .15:
            raise ValueError('Invalid joint step limit.')
        prediction = self.predict(side, point)
        return joints+np.clip(prediction-joints, -maximum_delta, maximum_delta)
