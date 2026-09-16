"""Contracts for the physical two-arm water-pouring demonstration."""
import unittest

import mujoco
import numpy as np

from scripts.evaluate_dinner_scene import load
from simulation_lab.pour_task import PourWaterTask
from simulation_lab.scene import HOME


class PourWaterTests(unittest.TestCase):
    def test_starts_with_a_table_supported_cup_relay_to_the_left_arm(self):
        model, data, layout = load(1000)
        for _ in range(200):
            mujoco.mj_step(model, data)
        task = PourWaterTask(model, data, layout)

        task.start()
        task.update(np.array(HOME * 2))

        snapshot = task.snapshot()
        self.assertEqual(snapshot["phase"], "cup_relay")
        self.assertEqual(snapshot["object_id"], "mug")
        self.assertEqual(snapshot["arm"], "right")
        self.assertIn("cup is relayed", snapshot["cooperation"])
        self.assertIn("left arm holds the cup", snapshot["cooperation"])

    def test_cup_relay_physically_releases_before_left_arm_regrips(self):
        model, data, layout = load(1000)
        for _ in range(200):
            mujoco.mj_step(model, data)
        targets = np.array(HOME * 2)
        dinner = __import__("simulation_lab.dinner_autonomy", fromlist=["DinnerSequence"]).DinnerSequence(model, data, layout)
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
            if task.phase == "hold_mug" or not task.active:
                break
            data.ctrl[:] = task.apply_gripper_limit(targets)
            mujoco.mj_step(model, data)
        self.assertTrue(task.active, task.message)
        self.assertEqual(task.phase, "hold_mug")


if __name__ == "__main__":
    unittest.main()
