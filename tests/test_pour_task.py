"""Contracts for the physical two-arm water-pouring demonstration."""
import unittest

import mujoco
import numpy as np

from scripts.evaluate_dinner_scene import load
from simulation_lab.dinner_autonomy import DinnerSequence
from simulation_lab.pour_task import PourWaterTask
from simulation_lab.scene import HOME


class PourWaterTests(unittest.TestCase):
    def test_starts_with_a_table_supported_bottle_relay_to_the_left_arm(self):
        model, data, layout = load(1000)
        for _ in range(200):
            mujoco.mj_step(model, data)
        task = PourWaterTask(model, data, layout)

        task.start()
        task.update(np.array(HOME * 2))

        snapshot = task.snapshot()
        self.assertEqual(snapshot["phase"], "bottle_relay")
        self.assertEqual(snapshot["object_id"], "bottle")
        self.assertEqual(snapshot["arm"], "right")
        self.assertIn("bottle is relayed", snapshot["cooperation"])
        self.assertIn("cup arm remains fixed", snapshot["cooperation"])

    def test_bottle_relay_physically_releases_before_left_arm_regrips(self):
        model, data, layout = load(1000)
        for _ in range(200):
            mujoco.mj_step(model, data)
        targets = np.array(HOME * 2)
        dinner = DinnerSequence(model, data, layout)
        dinner.start(kind="set_table")
        for _ in range(60_000):
            dinner.update(targets)
            if not dinner.active:
                break
            data.ctrl[:] = dinner.apply_gripper_limit(targets)
            mujoco.mj_step(model, data)
        self.assertEqual(dinner.status, "succeeded", dinner.message)

        task = PourWaterTask(model, data, layout)
        task.start()
        for _ in range(35_000):
            task.update(targets)
            if task.phase == "left_lift_bottle" or not task.active:
                break
            data.ctrl[:] = task.apply_gripper_limit(targets)
            mujoco.mj_step(model, data)
        self.assertTrue(task.active, task.message)
        self.assertEqual(task.phase, "left_lift_bottle")

    def test_complete_pour_keeps_cup_arm_fixed_during_tilt(self):
        model, data, layout = load(1000)
        for _ in range(200):
            mujoco.mj_step(model, data)
        targets = np.array(HOME * 2)
        dinner = DinnerSequence(model, data, layout)
        dinner.start(kind="set_table")
        for _ in range(60_000):
            dinner.update(targets)
            if not dinner.active:
                break
            data.ctrl[:] = dinner.apply_gripper_limit(targets)
            mujoco.mj_step(model, data)
        self.assertEqual(dinner.status, "succeeded", dinner.message)

        task = PourWaterTask(model, data, layout)
        task.start()
        held_qpos = None
        max_cup_motion = 0.0
        pour_stages = set()
        cup_grasp_verified = True
        min_cup_lift = float("inf")
        max_cup_table_force = 0.0
        min_mouth_to_cup_xy = float("inf")
        min_mouth_above_rim = float("inf")
        max_mouth_above_rim = float("-inf")
        closest_pour_geometry = None
        for _ in range(70_000):
            task.update(targets)
            if task.phase == "present_mug":
                pour_stages.add("pour_insert")
            if task.phase == "pour" and task.stage == "pour_hold" and held_qpos is None:
                held_qpos = data.qpos[6:12].copy()
            if task.phase == "pour":
                pour_stages.add(task.stage)
            if task.phase == "pour" and task.stage == "pour_hold" and held_qpos is not None:
                max_cup_motion = max(max_cup_motion, float(np.max(np.abs(data.qpos[6:12] - held_qpos))))
                forces, lift, _, both = task.mug_child._observe()
                cup_grasp_verified = cup_grasp_verified and both
                min_cup_lift = min(min_cup_lift, lift)
                max_cup_table_force = max(max_cup_table_force, forces["base"])
                if task.stage == "pour_hold":
                    bottle_rotation = data.xmat[task.bottle_child.body].reshape(3, 3)
                    # The physical neck proxy ends at local z=.160 m; that is
                    # the actual outlet which must enter the cup opening.
                    bottle_mouth = data.xpos[task.bottle_child.body] + bottle_rotation @ np.array([0.0, 0.0, 0.16])
                    cup_center = data.xpos[task.mug_child.body]
                    mouth_to_cup_xy = float(np.linalg.norm(bottle_mouth[:2] - cup_center[:2]))
                    if mouth_to_cup_xy < min_mouth_to_cup_xy:
                        min_mouth_to_cup_xy = mouth_to_cup_xy
                        closest_pour_geometry = (
                            bottle_mouth.copy(),
                            data.xpos[task.bottle_child.body].copy(),
                            cup_center.copy(),
                            bottle_rotation[:, 2].copy(),
                        )
                    mouth_above_rim = float(bottle_mouth[2] - (cup_center[2] + 0.064))
                    min_mouth_above_rim = min(min_mouth_above_rim, mouth_above_rim)
                    max_mouth_above_rim = max(max_mouth_above_rim, mouth_above_rim)
            if not task.active:
                break
            data.ctrl[:] = task.apply_gripper_limit(targets)
            mujoco.mj_step(model, data)
        self.assertEqual(
            task.status,
            "succeeded",
            f"{task.message}; bottle={data.xpos[model.body('bottle').id]}; "
            f"cup={data.xpos[model.body('mug').id]}; "
            f"left_wrist={data.xpos[model.body('left_gripper').id]}; "
            f"right_wrist={data.xpos[model.body('right_gripper').id]}",
        )
        self.assertTrue({"pour_tilt", "pour_insert", "pour_hold", "pour_return"}.issubset(pour_stages), pour_stages)
        self.assertLess(np.rad2deg(max_cup_motion), 0.5)
        self.assertTrue(cup_grasp_verified)
        self.assertGreater(min_cup_lift, 0.02)
        self.assertLess(max_cup_table_force, 0.02)
        self.assertLess(
            min_mouth_to_cup_xy,
            0.019,
            f"bottle mouth never entered cup opening; closest XY distance={min_mouth_to_cup_xy:.4f}m; "
            f"mouth/body/cup/axis={closest_pour_geometry}",
        )
        self.assertGreater(min_mouth_above_rim, 0.0)
        self.assertLess(max_mouth_above_rim, 0.06)


if __name__ == "__main__":
    unittest.main()
