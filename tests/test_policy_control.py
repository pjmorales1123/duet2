import unittest
from types import SimpleNamespace
import numpy as np
from simulation_lab.policy_control import apply_targets

class PolicyControlTests(unittest.TestCase):
    def test_right_gripper_cap_preserves_left_commands(self):
        model=SimpleNamespace(actuator_ctrlrange=np.tile([-1.,1.],(12,1)),
            actuator_gainprm=np.tile([20.,0.,0.],(12,1)),actuator_biasprm=np.tile([0.,0.,-2.],(12,1)))
        data=SimpleNamespace(qpos=np.zeros(12),qvel=np.full(12,.1))
        target=np.full(12,.8)
        command=apply_targets(model,data,target,.35,arm_offset=6)
        self.assertLessEqual(abs(20*command[11]-2*.1),.35+1e-12)
        np.testing.assert_array_equal(command[:11],target[:11])
        np.testing.assert_array_equal(data.qpos,np.zeros(12))
        for offset in (True,1,-6,12):
            with self.assertRaises(ValueError):apply_targets(model,data,target,arm_offset=offset)

    def test_clamps_commands_and_limits_gripper_without_changing_state(self):
        model=SimpleNamespace(actuator_ctrlrange=np.tile([-1.,1.],(12,1)),
            actuator_gainprm=np.tile([20.,0.,0.],(12,1)),actuator_biasprm=np.tile([0.,0.,-2.],(12,1)))
        data=SimpleNamespace(qpos=np.zeros(12),qvel=np.full(12,.1))
        before=data.qpos.copy()
        for target in [np.full(12,3.),np.full(12,-3.)]:
            command=apply_targets(model,data,target)
            torque=20*(command[5]-data.qpos[5])-2*data.qvel[5]
            self.assertLessEqual(abs(torque),.25+1e-12)
            self.assertTrue(np.all(np.abs(command[np.arange(12)!=5])<=1))
            np.testing.assert_array_equal(data.qpos,before)
        for invalid in [np.zeros(11),np.full(12,np.nan)]:
            with self.assertRaises(ValueError):apply_targets(model,data,invalid)

if __name__=='__main__':unittest.main()
