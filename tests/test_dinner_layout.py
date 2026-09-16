"""Contracts for exact dinner layout export and scene consumption."""
import tempfile
import unittest
from pathlib import Path

from simulation_lab.dinner_layout import canonical_dinner_layout, layout_document, load_dinner_layout, save_dinner_layout
from scripts.design_dinner_layout import current_poses
from simulation_lab.scene import TABLE_CENTER_Y, TABLE_Z, build_scene


class DinnerLayoutTests(unittest.TestCase):
    def test_canonical_layout_has_each_dinner_object_without_robot_state(self):
        layout = canonical_dinner_layout()
        self.assertEqual(set(layout["objects"]), {"bottle", "fork", "glass", "mug", "plate", "side_plate", "spoon"})
        self.assertNotIn("robots", layout)

    def test_layout_round_trip_preserves_world_pose(self):
        poses = {"spoon": {"position_m": [-.123456789, .234567891, TABLE_Z + .001], "yaw_rad": .812345678}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "layout.json"
            save_dinner_layout(path, poses)
            loaded = load_dinner_layout(path, ("plate", "spoon"))
        self.assertEqual(loaded["objects"], poses)
        document = layout_document(poses)
        self.assertEqual(document["coordinate_frame"]["table_center_m"][1], TABLE_CENTER_Y)

    def test_exact_pose_reaches_dinner_scene_body(self):
        document = layout_document({
            "spoon": {"position_m": [.123456789, -.234567891, TABLE_Z + .001], "yaw_rad": .812345678},
        })
        _, layout = build_scene(seed=42, scenario="dinner", dinner_layout=document)
        spoon = next(item for item in layout["objects"] if item["id"] == "spoon")
        self.assertEqual(spoon["initial_position_m"], document["objects"]["spoon"]["position_m"])
        self.assertEqual(spoon["initial_yaw_rad"], document["objects"]["spoon"]["yaw_rad"])
        self.assertEqual(layout["dinner_layout"]["objects"]["spoon"], document["objects"]["spoon"])

    def test_layout_rejects_unknown_objects_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            load_dinner_layout({"scene": "dinner", "objects": {"knife": {"position_m": [0, 0, TABLE_Z]}}}, ("spoon",))
        with self.assertRaises(ValueError):
            load_dinner_layout({"scene": "dinner", "objects": {"spoon": {"position_m": [0, 0, float("nan")]}}}, ("spoon",))

    def test_layout_can_store_robot_poses(self):
        document = layout_document(
            {"spoon": {"position_m": [0, 0, TABLE_Z], "yaw_rad": 0}},
            {"left": {"base_position_m": [-.25, -.235, .778], "gripper_position_m": [-.249, .073, .955]}},
        )
        self.assertEqual(document["robots"]["left"]["base_position_m"], [-.25, -.235, .778])

    def test_default_editor_poses_are_the_intended_table_arrangement(self):
        reference = current_poses(42)
        task = current_poses(42, "task")
        final = canonical_dinner_layout()['objects']
        self.assertAlmostEqual(reference["fork"]["position_m"][0], final['fork']['position_m'][0], places=3)
        self.assertAlmostEqual(reference["spoon"]["position_m"][1], final['spoon']['position_m'][1], places=3)
        self.assertGreater(task["fork"]["position_m"][2], TABLE_Z + .02)
        self.assertGreater(task["spoon"]["position_m"][2], TABLE_Z + .02)

    def test_mug_final_position_is_beside_bottle(self):
        final = canonical_dinner_layout()["objects"]
        bottle_x, bottle_y, _ = final["bottle"]["position_m"]
        mug_x, mug_y, _ = final["mug"]["position_m"]
        self.assertGreater(mug_x - bottle_x, .075)
        self.assertLess(mug_x - bottle_x, .105)
        self.assertAlmostEqual(mug_y, .02, places=3)


if __name__ == "__main__":
    unittest.main()
