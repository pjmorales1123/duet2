import unittest
import mujoco
import numpy as np
from scripts.evaluate_dinner_scene import load
from simulation_lab.scene import HOME
from simulation_lab.command_task import CommandSequence,ground_destination
from simulation_lab.language import parse_command,CommandError

class CommandTaskTests(unittest.TestCase):
    def test_relay_expands_and_physically_finishes(self):
        m,d,l=load(42)
        for _ in range(200):mujoco.mj_step(m,d)
        task=CommandSequence(m,d,l);target=np.array(HOME*2)
        task.start_plan(parse_command('pass the bottle to the right arm'))
        self.assertEqual([i['arm'] for i in task.intents],['left','right'])
        for _ in range(35000):
            before=d.qpos.copy();velocity=d.qvel.copy();task.update(target)
            np.testing.assert_array_equal(before,d.qpos);np.testing.assert_array_equal(velocity,d.qvel)
            self.assertEqual(m.neq,0);self.assertFalse(np.any(d.xfrc_applied))
            if not task.active:break
            d.ctrl[:]=task.apply_gripper_limit(target);mujoco.mj_step(m,d)
        self.assertEqual(task.status,'succeeded',task.message)
        self.assertEqual(len(task.results),2)
        for leg in task.results:
            self.assertGreater(leg['metrics']['max_lift_cm'],5)
            self.assertGreater(leg['metrics']['hold_verified_s'],1.7)
            self.assertLess(leg['metrics']['placement_xy_error_mm'],3)
        self.assertLess(np.linalg.norm(d.body('bottle').xpos[:2]-[.20,.05]),.003)

    def test_relay_refuses_unsupported_direction_and_cancels(self):
        m,d,l=load(42)
        for text in ['pass the plate to the right arm','pass the bottle to the left arm']:
            with self.subTest(text=text),self.assertRaises(CommandError):
                CommandSequence(m,d,l).start_plan(parse_command(text))
        task=CommandSequence(m,d,l);task.start_plan(parse_command('pass the bottle to the right arm'))
        target=np.array(HOME*2);before=d.qpos.copy();task.cancel(target)
        self.assertFalse(task.active);self.assertTrue(task.request_pause)
        np.testing.assert_array_equal(before,d.qpos)

    def test_language_bottle_runs_physical_baseline(self):
        m,d,l=load(42)
        for _ in range(200):mujoco.mj_step(m,d)
        task=CommandSequence(m,d,l);target=np.array(HOME*2)
        task.start_plan(parse_command('place the bottle'))
        for _ in range(18000):
            before=d.qpos.copy();task.update(target)
            np.testing.assert_array_equal(before,d.qpos)
            if not task.active:break
            d.ctrl[:]=task.apply_gripper_limit(target);mujoco.mj_step(m,d)
        self.assertEqual(task.status,'succeeded',task.message)
        self.assertEqual(task.snapshot()['policy_mode'],'programmed_physical_skills')
        self.assertEqual(m.neq,0)
    def test_drawer_dependency_and_occupied_destination(self):
        m,d,l=load(42)
        task=CommandSequence(m,d,l);task.start_plan(parse_command('take the spoon out of the drawer and put it in the left spot'))
        self.assertEqual(task.steps,['drawer','spoon'])
        positions={o['id']:d.body(o['body']).xpos.copy() for o in l['objects']}
        positions['mug']=np.array([.11,-.025,l['table_z']])
        with self.assertRaises(CommandError):ground_destination({'kind':'spot','side':'right'},'plate',l,positions)

if __name__=='__main__':unittest.main()
