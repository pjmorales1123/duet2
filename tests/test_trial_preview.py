"""Display images survive policy renderer disposal without stepping physics."""
import unittest
import mujoco
import numpy as np
from simulation_lab.scene import build_scene, HOME
from simulation_lab.trial_preview import TrialPreview


class TrialPreviewContracts(unittest.TestCase):
    def test_preview_survives_sequential_policy_contexts_without_state_writes(self):
        xml, _ = build_scene(seed=42, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:12] = HOME*2
        mujoco.mj_forward(model, data)
        before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        preview = TrialPreview(xml, width=320, height=240)
        try:
            initial = preview.frame(data, 'opposite')
            self.assertGreater(int(initial.max()), 0)
            for _ in range(3):
                with mujoco.Renderer(model, height=240, width=320) as policy:
                    policy.update_scene(data, camera='overhead')
                    self.assertGreater(int(policy.render().max()), 0)
                self.assertTrue(np.array_equal(preview.frame(data, 'opposite'), initial))
            for expected, actual in zip(before, (data.qpos, data.qvel, data.ctrl)):
                self.assertTrue(np.array_equal(expected, actual))
        finally:
            preview.close()
        self.assertFalse(preview.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
