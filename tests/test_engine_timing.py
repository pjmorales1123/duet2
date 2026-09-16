"""A deliberately stalled camera must not set the physics or control clock."""
import time
import unittest

from simulation_lab.engine import LabEngine


class TimingTests(unittest.TestCase):
    def test_slow_camera_does_not_pace_physics(self):
        engine = LabEngine(width=320, height=180, render_delay_s=.4)
        try:
            engine.start()
            first = engine.snapshot()
            start = time.perf_counter()
            time.sleep(2.)
            last = engine.snapshot()
            elapsed = time.perf_counter()-start
            simulated = last["simulation_time_s"]-first["simulation_time_s"]
            self.assertGreater(simulated/elapsed, .85)
            self.assertLess(simulated/elapsed, 1.15)
            self.assertGreaterEqual(last["frame_work_ms"], 400.)
            self.assertLess(last["frame_id"]-first["frame_id"], 7)
            paused = engine.submit("control", {"running": False})
            time.sleep(.15)
            self.assertEqual(engine.snapshot()["simulation_time_s"], paused["simulation_time_s"])
            stepped = engine.submit("control", {"step": True})
            self.assertAlmostEqual(stepped["simulation_time_s"]-paused["simulation_time_s"], .05, places=3)
        finally:
            engine.close()
        self.assertFalse(engine.render_process.is_alive())


if __name__ == "__main__":
    unittest.main()
