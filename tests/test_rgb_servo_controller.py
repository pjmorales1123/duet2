"""Observation and motor-feedback boundaries without optional training imports."""
import unittest
import numpy as np
from simulation_lab.rgb_servo_controller import RgbServoBottle
from simulation_lab.scene import HOME


class GeometryStub:
    def pose(self, side, joints):
        return np.array([.08, -.14, .94]), np.eye(3)


class MotorStub:
    geometry = GeometryStub()
    def predict(self, side, point):
        return np.array([point[0], point[1], point[2], 0., 0.])


class RgbServoContractTests(unittest.TestCase):
    def test_frozen_ablation_keeps_initial_image_and_live_accepts_new_image(self):
        first = {'status': 'observed', 'grasp_point_m': [.08, -.14, .888]}
        shifted = {'status': 'observed', 'grasp_point_m': [.092, -.14, .888]}
        live = RgbServoBottle(MotorStub(), [.1, -.115], image_mode='live')
        frozen = RgbServoBottle(MotorStub(), [.1, -.115], image_mode='frozen')
        for controller in (live, frozen):
            controller.accept_observation(first, 0.)
            controller.accept_observation(shifted, 2.)
        self.assertGreater(live.observation[0], first['grasp_point_m'][0])
        self.assertLessEqual(live.observation[0], shifted['grasp_point_m'][0])
        np.testing.assert_allclose(frozen.observation, first['grasp_point_m'])
        np.testing.assert_allclose(live.initial_observation, first['grasp_point_m'])

    def test_counterfactual_changes_action_without_advancing_task_or_mutating_joints(self):
        controller = RgbServoBottle(MotorStub(), [.1, -.115])
        controller.side = 'left'; controller.offset = 0
        q = np.array(HOME*2); controller.enter('approach', 0., q)
        q[:5] = [.08, -.14, .94, 0., 0.]
        before = q.copy(); history = list(controller.history)
        a = controller.preview_image_action([.08, -.14, .888], 3., q)
        b = controller.preview_image_action([.092, -.14, .888], 3., q)
        self.assertGreater(np.max(np.abs(a-b)), .01)
        np.testing.assert_array_equal(q, before)
        self.assertEqual(controller.history, history)
        self.assertEqual(controller.stage, 'approach')

    def test_bad_robot_observations_are_refused_before_action(self):
        controller = RgbServoBottle(MotorStub(), [.1, -.115])
        for positions in ([0.]*11, [float('nan')]*12):
            with self.assertRaises(ValueError):
                controller.update(0., positions, [0.]*12)


if __name__ == '__main__':
    unittest.main()
