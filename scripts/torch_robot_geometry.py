"""Differentiable robot-only forward geometry for offline motor training.

Local body/joint transforms follow MuJoCo's kinematic tree conventions:
https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
This module is never imported by a deployed controller.
"""
import mujoco
import numpy as np
import torch
from torch import nn


class TorchRobotGeometry(nn.Module):
    def __init__(self, model, side, dtype=torch.float64):
        super().__init__()
        self.side = side
        offset = 0 if side == 'left' else 6
        body = model.body(side + '_gripper').id
        chain = []
        while body:
            chain.append(body)
            body = int(model.body_parentid[body])
        self.steps = []
        for index, body in enumerate(reversed(chain)):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[body])
            for name, value in [('position', model.body_pos[body]), ('rotation', rotation.reshape(3, 3))]:
                self.register_buffer(f'{name}_{index}', torch.tensor(value.copy(), dtype=dtype))
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError('This offline helper expects at most one hinge per body.')
            if count:
                joint = int(model.body_jntadr[body])
                coordinate = int(model.jnt_qposadr[joint]) - offset
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not 0 <= coordinate < 5:
                    raise ValueError('Expected a five-hinge SO-101 gripper chain.')
                axis = model.jnt_axis[joint]
                x, y, z = axis
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                for name, value in [('anchor', model.jnt_pos[joint]), ('skew', skew), ('outer', np.outer(axis, axis))]:
                    self.register_buffer(f'{name}_{index}', torch.tensor(value.copy(), dtype=dtype))
                reference = float(model.qpos0[model.jnt_qposadr[joint]])
            else:
                coordinate, reference = None, 0.
            self.steps.append((index, coordinate, reference))
        self.register_buffer('identity', torch.eye(3, dtype=dtype))
        self.register_buffer('tool', torch.tensor([.003, 0., -.092], dtype=dtype))

    def forward(self, joints):
        if joints.ndim != 2 or joints.shape[1] != 5:
            raise ValueError('Expected a batch of five joint coordinates.')
        rotation = self.identity.expand(len(joints), 3, 3)
        position = torch.zeros((len(joints), 3), device=joints.device, dtype=joints.dtype)
        for index, coordinate, reference in self.steps:
            position = position + torch.matmul(rotation, getattr(self, f'position_{index}'))
            rotation = torch.matmul(rotation, getattr(self, f'rotation_{index}'))
            if coordinate is not None:
                anchor = getattr(self, f'anchor_{index}')
                pivot = position + torch.matmul(rotation, anchor)
                angle = (joints[:, coordinate] - reference)[:, None, None]
                cosine, sine = angle.cos(), angle.sin()
                hinge = (cosine*self.identity + sine*getattr(self, f'skew_{index}')
                         + (1-cosine)*getattr(self, f'outer_{index}'))
                rotation = torch.matmul(rotation, hinge)
                position = pivot - torch.matmul(rotation, anchor)
        return position + torch.matmul(rotation, self.tool), rotation
