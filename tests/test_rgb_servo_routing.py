"""Route decisions must retain arm reach constraints and the requested goal."""
import unittest
import numpy as np
from simulation_lab.rgb_servo_routing import RoutedRgbServoBottle, plan_bottle_route


CONFIG = {'route_samples':13,'lift_candidates_m':[.07,.065,.06,.055],
          'transport_height_above_table_m':[.198,.1875,.1775,.1675],
          'retract_candidates_m':[.065,.055,.045,.035],'shared_bottle_position_m':[0.,-.10]}


class Geometry:
    def pose(self, side, joints):
        return np.asarray(joints[:3]), np.eye(3)


class SplitReachMotor:
    geometry = Geometry()
    def predict(self, side, point):
        point = np.asarray(point)
        if (side == 'left' and point[0] < -.02) or (side == 'right' and point[0] > .02):
            raise ValueError('Requested point is outside this arm reach.')
        return np.r_[point,0.,0.]


class LimitedHeightMotor:
    geometry = Geometry()
    def predict(self, side, point):
        point = np.asarray(point)
        if point[0] > .14 and point[2] > .934:
            raise ValueError('High destination is outside reach.')
        return np.r_[point,0.,0.]


class RgbRouteTests(unittest.TestCase):
    def test_cross_arm_goal_uses_shared_release_point_without_changing_goal(self):
        route = plan_bottle_route(SplitReachMotor(),[.12,-.1,.888],[-.10,-.1],.76,'live',CONFIG)
        self.assertEqual(route['kind'],'table-supported relay')
        self.assertEqual([leg['side'] for leg in route['legs']],['left','right'])
        self.assertEqual(route['legs'][0]['destination'],[0.,-.1])
        self.assertEqual(route['legs'][-1]['destination'],[-.10,-.1])

    def test_transport_can_descend_after_supported_lift_without_relaxing_reach(self):
        controller = RoutedRgbServoBottle(LimitedHeightMotor(),[.16,-.06],.76,'live',CONFIG,'left')
        result = controller.plan_side('left',np.array([.10,-.14,.888]))
        self.assertGreater(result['lift_m'],.05)
        self.assertEqual(result['transport_height_above_table_m'],.1675)

    def test_visual_checkpoint_updates_held_offset_only_in_live_condition(self):
        config={**CONFIG,'carry_visual_updates':True,'maximum_carry_observation_age_s':.3}
        offsets=[]
        q=np.array([.08,-.1,.94,0.,0.,.1]*2)
        for mode in ('live','frozen'):
            c=RoutedRgbServoBottle(LimitedHeightMotor(),[.10,-.115],.76,mode,config,'left')
            c.side='left';c.offset=0;c.route_details={'lift_m':.065}
            c.grasp_offset=np.array([.01,0.,0.]);c.last_observation_time=16.
            c.observation=np.array([.081,-.1,.94])
            c.enter('align',16.1,q)
            offsets.append(c.grasp_offset.copy())
        np.testing.assert_allclose(offsets[0],[.001,0.,0.])
        np.testing.assert_allclose(offsets[1],[.01,0.,0.])


if __name__ == '__main__':unittest.main()
