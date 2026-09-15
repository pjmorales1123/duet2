"""Physical dinner task contracts, including refusal and interrupted execution."""
import unittest
import mujoco
import numpy as np
from scripts.evaluate_dinner_scene import load
from scripts.evaluate_dinner_autonomy import run
from simulation_lab.dinner_autonomy import DinnerSequence
from simulation_lab.scene import HOME


def setup():
    m,d,l=load()
    for _ in range(200):mujoco.mj_step(m,d)
    d.time=0.
    return m,d,DinnerSequence(m,d,l),np.array(HOME*2)


class DinnerAutonomyTests(unittest.TestCase):
    def test_complete_contact_only_sequence(self):
        result=run(42)
        self.assertEqual(result['status'],'succeeded',result['task']['message'])
        self.assertEqual(result['physical_state_writes'],0)
        self.assertEqual(result['external_forces'],0)
        self.assertEqual(result['equality_constraints'],0)
        self.assertEqual(result['actuators'],12)
        self.assertEqual(result['task']['completed_steps'],['bottle','plate','mug','fork','spoon'])
        for step in result['task']['results']:
            self.assertLess(step['metrics']['other_arm_max_motion_deg'],1.)
            self.assertGreaterEqual(step['metrics']['hold_verified_s'],1.5)
            self.assertLess(step['metrics']['placement_xy_error_mm'],6.)
            self.assertLess(step['metrics']['placement_z_error_mm'],3.)
            if step['skill'] in ('fork', 'spoon'):
                self.assertTrue(step['metrics']['open_gripper_destination_checked'])
                self.assertEqual(step['metrics']['grasp_local_m'], [0, 0, .008])

    def test_invalid_object_rejected_and_fork_has_no_drawer_dependency(self):
        m,d,t,target=setup()
        with self.assertRaises(ValueError):t.start(kind='dinner_place',object_id='glass')
        with self.assertRaises(ValueError):t.start(kind='drawer_open',side='right')
        with self.assertRaises(ValueError):t.start(kind='set_table',side='left')
        t.start(kind='dinner_place',object_id='fork')
        before=d.qpos.copy()
        t.update(target)
        self.assertEqual(t.status,'running',t.message)
        self.assertEqual(t.stage,'approach')
        np.testing.assert_array_equal(before,d.qpos)

    def test_cancel_and_servo_timeout(self):
        m,d,t,target=setup()
        t.start(kind='dinner_place',object_id='bottle')
        t.update(target)
        t.cancel(target)
        before=d.qpos.copy()
        t.update(target)
        self.assertEqual(t.status,'cancelled')
        self.assertTrue(t.request_pause)
        np.testing.assert_array_equal(before,d.qpos)
        t.start(kind='dinner_place',object_id='bottle')
        for _ in range(5000):
            t.update(target)
            if not t.active:break
            # Deliberately unresponsive servos: physics continues under Home targets.
            d.ctrl[:]=HOME*2
            mujoco.mj_step(m,d)
        self.assertEqual(t.status,'failed')
        self.assertIn('timed out',t.message)

    def test_external_disturbance_cannot_be_counted_as_success(self):
        m,d,t,target=setup()
        t.start(kind='dinner_place',object_id='bottle')
        disturbed=False
        for _ in range(15000):
            t.update(target)
            if not t.active:break
            if t.stage=='hold':
                # A test disturbance, never used by the controller or success evaluation.
                d.xfrc_applied[m.body('bottle').id,2]=-15.
                disturbed=True
            d.ctrl[:]=t.apply_gripper_limit(target)
            mujoco.mj_step(m,d)
        self.assertTrue(disturbed)
        self.assertEqual(t.status,'failed')
        self.assertTrue(t.request_pause)

    def test_gripper_fault_has_only_one_retry(self):
        m,d,t,target=setup()
        t.start(kind='dinner_place',object_id='bottle')
        for _ in range(15000):
            t.update(target)
            if not t.active:break
            applied=t.apply_gripper_limit(target)
            if t.child.side:
                applied[t.child.offset+5]=.4  # Test fault: gripper cannot close.
            d.ctrl[:]=applied
            mujoco.mj_step(m,d)
        self.assertEqual(t.status,'failed')
        self.assertEqual(t.child.attempts,2)
        self.assertEqual(sum(h['stage']=='retry' for h in t.child.history),1)
        self.assertIn('allowed attempts',t.message)

if __name__=='__main__':unittest.main()
