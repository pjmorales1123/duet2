"""The independent scorer must not mistake an untouched scene for task success."""
import unittest
import mujoco
import numpy as np
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.scene import HOME, build_scene


class DinnerMonitorTests(unittest.TestCase):
    def test_unmoved_scene_and_manually_open_reset_do_not_count_as_manipulation(self):
        for skill in ('bottle', 'plate', 'mug', 'fork', 'spoon'):
            with self.subTest(skill=skill):
                xml, layout = build_scene(seed=42, scenario='dinner', dinner_preset='task')
                model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
                data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
                mujoco.mj_forward(model, data)
                monitor = DinnerPhysicalMonitor(model, data, layout, skill, 'left')
                for _ in range(240):
                    before, velocity, controls = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
                    monitor.update()
                    np.testing.assert_array_equal(before, data.qpos)
                    np.testing.assert_array_equal(velocity, data.qvel)
                    np.testing.assert_array_equal(controls, data.ctrl)
                    mujoco.mj_step(model, data)
                self.assertFalse(monitor.succeeded)
                self.assertEqual(monitor.report()['teacher_updates'], 0)


if __name__ == '__main__': unittest.main()
