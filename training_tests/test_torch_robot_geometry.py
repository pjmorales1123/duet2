import unittest
import mujoco
import numpy as np
import torch
from scripts.torch_robot_geometry import TorchRobotGeometry
from simulation_lab.rgb_servo_motor import RobotGeometry
from simulation_lab.scene import build_scene


class DifferentiableGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])

    def test_matches_mujoco_across_both_robot_joint_ranges(self):
        rng = np.random.default_rng(2026096302)
        reference = RobotGeometry(self.model)
        for side, offset in [('left', 0), ('right', 6)]:
            limits = self.model.actuator_ctrlrange[offset:offset+5]
            q = rng.uniform(limits[:, 0], limits[:, 1], (256, 5))
            points, rotations = TorchRobotGeometry(self.model, side)(torch.tensor(q))
            actual = [reference.pose(side, row) for row in q]
            np.testing.assert_allclose(points.detach().numpy(), np.array([p for p, _ in actual]), atol=1e-12, rtol=0)
            np.testing.assert_allclose(rotations.detach().numpy(), np.array([r for _, r in actual]), atol=1e-12, rtol=0)

    def test_gradients_match_independent_mujoco_finite_differences(self):
        reference = RobotGeometry(self.model)
        for side in ('left', 'right'):
            q = torch.tensor([[.3, -.6, .8, .2, -.4]], dtype=torch.float64, requires_grad=True)
            geometry = TorchRobotGeometry(self.model, side)
            point, rotation = geometry(q)
            loss = point[0] @ torch.tensor([.4, -.2, .7], dtype=torch.float64) + rotation[0, 2, 1]
            loss.backward()
            finite = []
            for joint in range(5):
                values = []
                for delta in (-1e-6, 1e-6):
                    row = q.detach().numpy()[0].copy()
                    row[joint] += delta
                    p, r = reference.pose(side, row)
                    values.append(p @ np.array([.4, -.2, .7]) + r[2, 1])
                finite.append((values[1]-values[0]) / 2e-6)
            np.testing.assert_allclose(q.grad.numpy()[0], finite, atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
