"""Physical contracts for the dinner assets and passive drawer."""
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.evaluate_dinner_scene import load, stability
from simulation_lab.dinner import DRAWER_TRAVEL
from simulation_lab.scene import HOME, build_scene


class DinnerTests(unittest.TestCase):
    def test_seeds_and_invalid_scene_combinations(self):
        self.assertEqual(build_scene(scenario="dinner"), build_scene(scenario="dinner"))
        self.assertNotEqual(build_scene(seed=0, scenario="dinner")[1]["objects"], build_scene(seed=1, scenario="dinner")[1]["objects"])
        for kwargs in ({"scenario": "unknown"}, {"scenario": "dinner", "practice": True},
                       {"scenario": "dinner", "dinner_preset": "reference", "drawer_open": True},
                       {"scenario": "dinner", "dinner_preset": "unknown"}):
            with self.assertRaises(ValueError):
                build_scene(**kwargs)

    def test_free_props_passive_drawer_and_stability(self):
        for preset, opened in (("task", False), ("task", True), ("reference", False)):
            with self.subTest(preset=preset, opened=opened):
                m, d, layout = load(42, preset, opened)
                self.assertEqual(m.nu, 12)
                self.assertEqual(m.neq, 0)
                self.assertEqual(m.nq, 12+1+7*len(layout["objects"]))
                for item in layout["objects"]:
                    self.assertEqual(m.joint(item["body"]+"_free").type[0], mujoco.mjtJoint.mjJNT_FREE)
                    self.assertAlmostEqual(m.body(item["body"]).mass[0], item["mass_kg"])
                result = stability(m, d, layout)
                self.assertTrue(result["passed"], result)

    def test_drawer_moves_only_by_physics_and_carries_loose_cutlery(self):
        m, d, layout = load()
        stability(m, d, layout, 1.)
        fork_start = d.body("fork").xpos.copy()
        dof = m.joint("drawer_slide").dofadr[0]
        # Apply a modest test force to the passive joint, not a position edit.
        d.qfrc_applied[dof] = .7
        for _ in range(300):
            mujoco.mj_step(m, d)
        self.assertGreater(d.joint("drawer_slide").qpos[0], .06)
        self.assertLess(d.joint("drawer_slide").qpos[0], DRAWER_TRAVEL+.002)
        self.assertLess(d.body("fork").xpos[1], fork_start[1]-.04)
        # Independently lift the fork: it is not parented/welded to the drawer.
        before = d.body("fork").xpos[2]
        d.xfrc_applied[m.body("fork").id, 2] = .20
        for _ in range(50):
            mujoco.mj_step(m, d)
        self.assertGreater(d.body("fork").xpos[2], before+.015)

    def test_vessels_have_open_collision_cavities(self):
        # Drop a 3 mm probe into each mouth; it must reach the real vessel base.
        xml, layout = build_scene(scenario="dinner")
        root = ET.fromstring(xml)
        world = root.find("worldbody")
        for name in ("mug", "glass", "bottle"):
            item = next(i for i in layout["objects"] if i["id"] == name)
            x, y, z = item["initial_position_m"]
            body = ET.SubElement(world, "body", name=name+"_probe", pos=f"{x} {y} {z+item['size_m'][2]+.015}")
            ET.SubElement(body, "freejoint")
            ET.SubElement(body, "geom", type="sphere", size=".003", mass=".001")
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")); d = mujoco.MjData(m)
        d.qpos[:12] = HOME*2; d.ctrl[:] = HOME*2
        for _ in range(400):
            mujoco.mj_step(m, d)
        for name in ("mug", "glass", "bottle"):
            delta = d.body(name+"_probe").xpos-d.body(name).xpos
            self.assertGreater(delta[2], .014)
            self.assertLess(delta[2], .022)
            self.assertLess(np.linalg.norm(delta[:2]), .020)


if __name__ == "__main__":
    unittest.main()
