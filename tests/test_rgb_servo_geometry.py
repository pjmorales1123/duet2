import unittest
import numpy as np
from simulation_lab.rgb_servo_cameras import project
from simulation_lab.rgb_servo_geometry import rigid_bottle_keypoints


class RigidBottleGeometryTests(unittest.TestCase):
    def test_known_shape_fit_recovers_calibrated_points_without_object_state(self):
        intrinsic = np.array([[500., 0., 160.], [0., 500., 120.], [0., 0., 1.]])
        matrices = [intrinsic@np.column_stack([np.eye(3), -np.asarray(center)])
                    for center in ([0., 0., 0.], [.2, 0., 0.], [0., .2, 0.])]
        axis = np.array([.03, -.02, 1.]); axis /= np.linalg.norm(axis)
        truth = np.array([[.03, -.06, 1.], np.array([.03, -.06, 1.])+.103*axis])
        pixels = [project(truth, matrix) for matrix in matrices]
        initial = truth+[[.001, -.001, .003], [-.001, .001, -.001]]
        recovered, details = rigid_bottle_keypoints(initial, pixels, matrices)
        self.assertLess(np.max(np.abs(recovered-truth)), 1e-6)
        self.assertAlmostEqual(np.linalg.norm(recovered[1]-recovered[0]), .103, places=10)
        self.assertLessEqual(details['final_rigid_squared_pixel_error'], details['initial_rigid_squared_pixel_error'])


if __name__ == '__main__':
    unittest.main()
