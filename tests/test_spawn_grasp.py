"""Regression tests for spawn-pose-aware gripper orientation."""
import math
import unittest

import numpy as np

from simulation_lab.spawn_grasp import world_grasp_axis


class SpawnGraspTests(unittest.TestCase):
    def test_rotates_a_local_fork_grasp_axis_with_the_spawn_yaw(self):
        yaw = math.pi / 2
        rotation = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.],
            [math.sin(yaw), math.cos(yaw), 0.],
            [0., 0., 1.],
        ])

        axis = world_grasp_axis(rotation, np.array([-1., 0., 0.]))

        np.testing.assert_allclose(axis, [0., -1., 0.], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
