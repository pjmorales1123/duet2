"""Physical-scene checks; run with python -m unittest discover -s tests."""
import math
import unittest

import mujoco
import numpy as np

from simulation_lab.scene import CAMERAS, HOME, JOINTS, RackPose, TABLE_Z, build_scene, random_layout, validate_layout


class SceneTests(unittest.TestCase):
    def test_seed_reproduces_poses_and_tube_occupancy(self):
        self.assertEqual(build_scene(42, 3), build_scene(42, 3))
        self.assertNotEqual(random_layout(42, 3), random_layout(43, 3))

    def test_layout_bounds_and_clearance(self):
        for seed in range(30):
            for count in (2, 3, 4):
                validate_layout(random_layout(seed, count))
        for invalid in [[RackPose(0, .1, 0)], [RackPose(0, .1, 0), RackPose(.01, .1, 0)], [RackPose(.4, .1, 0), RackPose(-.2, .1, 0)], [RackPose(float('nan'), .1, 0), RackPose(-.2, .1, 0)]]:
            with self.assertRaises(ValueError):
                validate_layout(invalid)

    def test_robot_contract_and_dynamic_props(self):
        xml, layout = build_scene(7, 2)
        model = mujoco.MjModel.from_xml_string(xml)
        original_visual_model = mujoco.MjModel.from_xml_string(xml.replace('file="lod/', 'file="'))
        np.testing.assert_allclose(model.body_mass, original_visual_model.body_mass, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(model.body_inertia, original_visual_model.body_inertia, rtol=1e-10, atol=1e-12)
        self.assertEqual(model.nu, 12)
        self.assertEqual(model.neq, 0)
        self.assertEqual(model.nq, 12 + 7 * len(layout['tubes']))
        for side in ('left', 'right'):
            for name in JOINTS:
                self.assertGreaterEqual(model.joint(f'{side}_{name}').id, 0)
        for camera in CAMERAS:
            self.assertGreaterEqual(model.camera(camera).id, 0)

    def test_tubes_settle_and_arm_targets_track(self):
        for seed, count in ((0, 2), (42, 3), (17, 4)):
            with self.subTest(seed=seed, racks=count):
                xml, layout = build_scene(seed, count)
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                data.qpos[:12] = HOME * 2
                data.ctrl[:] = HOME * 2
                data.ctrl[0] = 0.1
                for _ in range(600):
                    mujoco.mj_step(model, data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertTrue(np.isfinite(data.qvel).all())
                self.assertAlmostEqual(data.qpos[0], 0.1, delta=0.025)
                for tube in layout['tubes']:
                    body = data.body(tube['body'])
                    self.assertGreater(body.xpos[2], TABLE_Z + 0.025)
                    self.assertGreater(body.xmat[8], math.cos(math.radians(20)))
                # Verify these are free physical props, not a static animated image.
                body_id = model.body(layout['tubes'][0]['body']).id
                before = data.xpos[body_id, 2]
                data.xfrc_applied[body_id, 2] = 0.7
                for _ in range(70):
                    mujoco.mj_step(model, data)
                self.assertGreater(data.xpos[body_id, 2], before + 0.04)


if __name__ == '__main__':
    unittest.main()
