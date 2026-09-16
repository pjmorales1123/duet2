"""Physical success and fault-injection tests, without a renderer or HTTP server."""
import unittest

import mujoco
import numpy as np

from simulation_lab.autonomy import LiftReturn
from simulation_lab.engine import LabEngine
from simulation_lab.scene import HOME, RackPose, build_scene
from scripts.evaluate_autonomy import trial


def setup(racks=None):
    xml, layout = build_scene(42, racks=racks, practice=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    target = np.array(HOME*2)
    data.qpos[:12], data.ctrl[:] = target, target
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data, target, LiftReturn(model, data, layout)


def advance(model, data, target, task, until=None, count=15000, apply=True):
    for _ in range(count):
        before = data.qpos.copy()
        task.update(target)
        np.testing.assert_array_equal(data.qpos, before, err_msg="Controller teleported a physical coordinate")
        if not task.active or task.stage == until:
            return
        if apply:
            data.ctrl[:] = task.apply_gripper_limit(target)
        mujoco.mj_step(model, data)


class AutonomyTests(unittest.TestCase):
    def test_either_arm_physically_lifts_and_returns(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                result = trial(42, arm=side)
                self.assertEqual(result["task"]["status"], "succeeded", result["task"]["message"])
                metrics = result["task"]["metrics"]
                self.assertGreaterEqual(result["measured_hold_min_cm"], 5.)
                self.assertLessEqual(result["peak_gripper_torque_nm"], .151)
                self.assertGreaterEqual(metrics["hold_verified_s"], 1.5)
                self.assertGreaterEqual(metrics["placement_stable_s"], 1.)
                self.assertLess(metrics["placement_xy_error_mm"], 1.)
                self.assertLess(metrics["tilt_deg"], 1.)
                self.assertLess(metrics["other_arm_max_motion_deg"], .1)
                self.assertEqual(metrics["unexpected_collisions"], 0)

    def test_unreachable_goal_rejected_without_motion(self):
        model, data, target, task = setup([RackPose(-.29, .30, 0), RackPose(.29, .30, 0)])
        before = data.qpos.copy()
        task.start()
        task.update(target)
        self.assertEqual(task.status, "failed")
        self.assertIn("No clear reachable grasp", task.message)
        np.testing.assert_array_equal(data.qpos, before)

    def test_missing_grasp_has_one_retry_then_failure(self):
        model, data, target, task = setup()
        task.start("left", "A2")
        task.update(target)
        # Fault injection into this isolated test model: make the fingers unable
        # to contact anything. The real controller must not claim a lift.
        ids = list(task.jaw_geoms)
        model.geom_contype[ids] = 0
        model.geom_conaffinity[ids] = 0
        advance(model, data, target, task)
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.attempts, 2)
        self.assertIn("stable grasp", task.message)
        self.assertLess(task.metrics["max_lift_cm"], .1)
        self.assertNotIn("hold", [h["stage"] for h in task.history])

    def test_loss_of_grasp_cannot_pass_the_hold(self):
        model, data, target, task = setup()
        task.start("left", "A2")
        advance(model, data, target, task, until="hold")
        self.assertEqual(task.stage, "hold")
        ids = list(task.jaw_geoms)
        model.geom_contype[ids] = 0
        model.geom_conaffinity[ids] = 0
        advance(model, data, target, task, count=100)
        self.assertEqual(task.status, "failed")
        self.assertTrue(task.request_pause)
        self.assertLess(task.metrics["hold_verified_s"], 1.5)

    def test_stalled_arm_times_out(self):
        model, data, target, task = setup()
        task.start("left", "A2")
        advance(model, data, target, task, apply=False)
        self.assertEqual(task.status, "failed")
        self.assertIn("Timed out", task.message)

    def test_cancel_pauses_and_releases_control_ownership(self):
        engine = LabEngine()
        engine._reset(practice=True)
        engine._task_command({"action": "start", "arm": "left", "tube_id": "A2"})
        engine._step(100)
        with self.assertRaises(ValueError):
            engine._control({"preset": "home"})
        with self.assertRaises(ValueError):
            engine._task_command({"action": "start"})
        before = engine.data.qpos.copy()
        engine._task_command({"action": "cancel"})
        self.assertFalse(engine.running)
        self.assertEqual(engine.task.status, "cancelled")
        np.testing.assert_array_equal(engine.data.qpos, before)
        engine._control({"preset": "home"})
        np.testing.assert_allclose(engine.target, HOME*2)
        engine._reset(practice=True)
        self.assertEqual(engine.task.status, "idle")
        with self.assertRaises(ValueError):
            engine.task.start(tube_id="D6")


if __name__ == "__main__":
    unittest.main()
