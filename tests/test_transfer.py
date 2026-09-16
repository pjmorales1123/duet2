"""Rack transfer, payload collision checks and live slot identity invariants."""
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simulation_lab.autonomy import LiftReturn
from simulation_lab.scene import HOME, build_scene
from simulation_lab.slots import slot_state
from scripts.evaluate_autonomy import trial
from test_autonomy import advance


def setup(obstacle=False):
    xml, layout = build_scene(42, transfer_side="left")
    if obstacle:
        root = ET.fromstring(xml)
        a, b = layout["slots"][1], layout["slots"][7]
        xy = (np.array(a["position_m"][:2])+b["position_m"][:2])/2
        ET.SubElement(root.find("worldbody"), "geom", name="payload_obstacle", type="box", size=".013 .017 .018", pos=f"{xy[0]} {xy[1]} .870")
        xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    targets = np.array(HOME*2)
    data.qpos[:12], data.ctrl[:] = targets, targets
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data, targets, LiftReturn(model, data, layout)


class TransferTests(unittest.TestCase):
    def test_physical_transfer_with_either_arm(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                result = trial(42, side, kind="transfer", transfer_side=side)
                self.assertEqual(result["task"]["status"], "succeeded", result["task"]["message"])
                self.assertEqual(result["task"]["source_slot"], "A2")
                self.assertEqual(result["task"]["destination_slot"], "B2")
                self.assertLess(result["task"]["metrics"]["placement_xy_error_mm"], 1.)
                self.assertGreater(result["task"]["metrics"]["hold_verified_s"], 1.49)
                self.assertLess(result["peak_gripper_torque_nm"], .151)

    def test_occupied_and_same_rack_destinations_rejected(self):
        for destination, reason in (("B4", "occupied"), ("A1", "different rack")):
            model, data, targets, task = setup()
            before = data.qpos.copy()
            task.start("left", "A2", "transfer", destination)
            task.update(targets)
            self.assertEqual(task.status, "failed")
            self.assertIn(reason, task.message)
            np.testing.assert_array_equal(data.qpos, before)

    def test_collision_checks_include_carried_tube(self):
        model, data, targets, task = setup(obstacle=True)
        before = data.qpos.copy()
        task.start("left", "A2", "transfer", "B2")
        task.update(targets)
        self.assertEqual(task.status, "failed")
        self.assertIn("carried tube's path intersects payload_obstacle", task.message)
        np.testing.assert_array_equal(data.qpos, before)

    def test_tube_identity_and_occupancy_support_a_second_transfer(self):
        model, data, targets, task = setup()
        for source, destination in (("A2", "B2"), ("B2", "A2")):
            task.start("left", "A2", "transfer", destination)
            advance(model, data, targets, task)
            self.assertEqual(task.status, "succeeded", task.message)
            self.assertEqual(task.source_slot["id"], source)
            occupied = {slot["id"]: slot["occupants"] for slot in slot_state(data, task.layout)}
            self.assertEqual(occupied[source], [])
            self.assertEqual(occupied[destination], ["A2"])

    def test_destination_is_rechecked_after_pickup(self):
        model, data, targets, task = setup()
        task.start("left", "A2", "transfer", "B2")
        advance(model, data, targets, task, until="hold")
        self.assertEqual(task.stage, "hold")
        # Isolated fault injection: another object appears in the reserved slot.
        address = model.joint("tube_B4_free").qposadr[0]
        data.qpos[address:address+2] = task.destination["position_m"][:2]
        mujoco.mj_forward(model, data)
        advance(model, data, targets, task, count=400)
        self.assertEqual(task.status, "failed")
        self.assertIn("reserved destination", task.message)


if __name__ == "__main__":
    unittest.main()
