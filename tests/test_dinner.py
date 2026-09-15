"""Physical contracts for the dinner assets and passive drawer."""
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.evaluate_dinner_scene import load, stability
from simulation_lab.dinner_layout import canonical_dinner_layout
from simulation_lab.scene import HOME, build_scene


class DinnerTests(unittest.TestCase):
    def test_seeds_and_invalid_scene_combinations(self):
        self.assertEqual(build_scene(scenario="dinner"), build_scene(scenario="dinner"))
        self.assertNotEqual(build_scene(seed=0, scenario="dinner")[1]["objects"], build_scene(seed=1, scenario="dinner")[1]["objects"])
        for kwargs in ({"scenario": "unknown"}, {"scenario": "dinner", "practice": True},
                       {"scenario": "dinner", "dinner_preset": "unknown"}):
            with self.assertRaises(ValueError):
                build_scene(**kwargs)

    def test_free_props_cabinet_source_and_stability(self):
        for preset, opened in (("task", False), ("reference", False)):
            with self.subTest(preset=preset, opened=opened):
                m, d, layout = load(42, preset, opened)
                self.assertEqual(m.nu, 12)
                self.assertEqual(m.neq, 0)
                self.assertEqual(m.nq, 12+7*len(layout["objects"]))
                for item in layout["objects"]:
                    self.assertEqual(m.joint(item["body"]+"_free").type[0], mujoco.mjtJoint.mjJNT_FREE)
                    self.assertAlmostEqual(m.body(item["body"]).mass[0], item["mass_kg"])
                result = stability(m, d, layout)
                self.assertTrue(result["passed"], result)

    def test_cabinet_zone_groups_canonical_sources_without_a_drawer_joint(self):
        m, d, layout = load()
        self.assertEqual(layout["source_zone"]["kind"], "cabinet_zone")
        with self.assertRaises(KeyError):
            m.joint("drawer_slide")
        for target in layout['targets']:
            pose = canonical_dinner_layout()['objects'][target['object_id']]
            np.testing.assert_allclose(target['position_m'][:2], pose['position_m'][:2], atol=1e-9)

    def test_vessels_have_open_collision_cavities(self):
        # Drop a 3 mm probe into each mouth; it must reach the real vessel base.
        for name in ("mug", "glass", "bottle"):
            xml, layout = build_scene(scenario="dinner")
            root = ET.fromstring(xml)
            world = root.find("worldbody")
            item = next(i for i in layout["objects"] if i["id"] == name)
            x, y, z = item["initial_position_m"]
            body = ET.SubElement(world, "body", name=name+"_probe", pos=f"{x} {y} {z+item['size_m'][2]+.015}")
            ET.SubElement(body, "freejoint")
            ET.SubElement(body, "geom", type="sphere", size=".003", mass=".001")
            m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")); d = mujoco.MjData(m)
            # A 3 mm microprobe needs finer integration than the full robot.
            m.opt.timestep = .001
            d.qpos[:12] = HOME*2; d.ctrl[:] = HOME*2
            for _ in range(2000):
                mujoco.mj_step(m, d)
            delta = d.body(name+"_probe").xpos-d.body(name).xpos
            self.assertGreater(delta[2], .014)
            self.assertLess(delta[2], .022)
            self.assertLess(np.linalg.norm(delta[:2]), .020)


if __name__ == "__main__":
    unittest.main()
