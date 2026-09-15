"""Record observations/actions at the same simulation time and reject incomplete data."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simulation_lab.recording import EpisodeRecorder
from scripts.validate_demonstration import validate
from test_autonomy import setup


def wait_saved(recorder):
    deadline = time.monotonic()+30
    while recorder.snapshot()["busy"] and time.monotonic() < deadline:
        time.sleep(.05)
    assert not recorder.snapshot()["busy"], recorder.snapshot()
    return recorder.snapshot()


class RecordingTests(unittest.TestCase):
    def test_low_disk_space_prevents_recording(self):
        with tempfile.TemporaryDirectory() as root:
            recorder=EpisodeRecorder(root)
            try:
                with patch('simulation_lab.recording.require_space',side_effect=OSError('Insufficient disk headroom')):
                    with self.assertRaises(OSError):recorder.start('',{}, {},0.)
                self.assertFalse(recorder.capturing)
                self.assertFalse(recorder.snapshot()['busy'])
                self.assertEqual(list(Path(root).iterdir()),[])
            finally:recorder.close()

    def test_synchronized_actions_observations_and_images(self):
        with tempfile.TemporaryDirectory() as root:
            model, data, targets, task = setup()
            from simulation_lab.scene import build_scene
            xml, _ = build_scene(42, practice=True)
            recorder = EpisodeRecorder(root)
            try:
                task.start("left", "A2")
                episode_id = recorder.start(xml, task.layout, task.snapshot(), data.time, images=True)
                for _ in range(60):
                    task.update(targets)
                    ctrl = task.apply_gripper_limit(targets)
                    recorder.capture(data, targets, ctrl, task)
                    data.ctrl[:] = ctrl
                    mujoco.mj_step(model, data)
                task.cancel(targets)
                recorder.finish(task, data)
                status = wait_saved(recorder)
                self.assertEqual(status["status"], "completed", status)
                result = validate(Path(root)/episode_id)
                self.assertEqual(result["actions"], 60)
                self.assertEqual(result["observations"], 7)
                self.assertEqual(result["images"], 21)
                self.assertEqual(result["outcome"], "cancelled")
                self.assertFalse(result["training_eligible"])
            finally:
                recorder.close()

    def test_queue_overflow_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            model, data, targets, task = setup()
            from simulation_lab.scene import build_scene
            xml, _ = build_scene(42, practice=True)
            recorder = EpisodeRecorder(root, queue_size=2, writer_delay_s=.05)
            try:
                task.start()
                episode_id = recorder.start(xml, task.layout, task.snapshot(), data.time, images=False)
                for _ in range(50):
                    recorder.capture(data, targets, targets, task)
                    data.time += .005
                task.cancel(targets)
                recorder.finish(task, data)
                status = wait_saved(recorder)
                self.assertEqual(status["status"], "failed")
                manifest = json.loads((Path(root)/episode_id/"manifest.json").read_text())
                self.assertFalse(manifest["trajectory_complete"])
                self.assertFalse(manifest["training_eligible"])
                with self.assertRaises(ValueError):
                    validate(Path(root)/episode_id)
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
