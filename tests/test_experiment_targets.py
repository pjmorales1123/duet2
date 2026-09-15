import unittest
import mujoco
import numpy as np
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.experiment_targets import with_bottle_destination
from simulation_lab.scene import HOME, build_scene


class ExperimentalTargetTests(unittest.TestCase):
    def test_requested_goal_reaches_physical_monitor_when_scene_has_no_bottle_target(self):
        xml, original = build_scene(seed=42, scenario='dinner', dinner_preset='task')
        self.assertFalse(any(t['object_id'] == 'bottle' for t in original['targets']))
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        data.qpos[:12] = HOME*2; mujoco.mj_forward(model, data)
        for destination in ([.10, -.115], [.16, -.06], [.20, .05]):
            changed = with_bottle_destination(original, destination)
            monitor = DinnerPhysicalMonitor(model, data, changed, 'bottle', 'left')
            np.testing.assert_allclose(monitor.destination[:2], destination, atol=1e-12)
        self.assertFalse(any(t['object_id'] == 'bottle' for t in original['targets']))


if __name__ == '__main__':
    unittest.main()
