"""Regression tests for carried-object planar detours."""
import unittest

import numpy as np

from simulation_lab.carry_routing import plan_disc_detour, segment_clear_of_disc


class CarryRoutingTests(unittest.TestCase):
    def test_routes_a_carried_fork_around_a_plate_that_blocks_direct_transit(self):
        start = np.array([-.280, .045])
        end = np.array([.080, -.045])
        plate = np.array([-.060, -.025])
        clearance = .092

        self.assertFalse(segment_clear_of_disc(start, end, plate, clearance))

        route = plan_disc_detour(start, end, plate, clearance)

        self.assertGreaterEqual(len(route), 3)
        self.assertTrue(all(
            segment_clear_of_disc(a, b, plate, clearance)
            for a, b in zip(route, route[1:])
        ))


if __name__ == '__main__':
    unittest.main()
