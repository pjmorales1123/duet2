"""Camera process: no access to the authoritative physics state or controller."""
from __future__ import annotations

import io
import multiprocessing
import queue
import time

import mujoco
from PIL import Image


def offer_latest(channel, item):
    """Bounded, nonblocking transport; outdated frames/states may be dropped."""
    try:
        channel.put_nowait(item)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            return
        try:
            channel.put_nowait(item)
        except queue.Full:
            pass


def render_worker(inputs, outputs, stopping, width, height, fps, delay_s=0.):
    renderer = None
    generation = -1
    previous = time.perf_counter()
    parent = multiprocessing.parent_process()
    try:
        while not stopping.is_set() and (parent is None or parent.is_alive()):
            try:
                job = inputs.get(timeout=.2)
            except queue.Empty:
                continue
            # A slow frame must never accumulate a backlog of obsolete poses.
            while True:
                try:
                    job = inputs.get_nowait()
                except queue.Empty:
                    break
            start = time.perf_counter()
            xml, version, qpos, qvel, ctrl, sim_time, camera, shadows, idle = job
            if generation != version:
                if renderer:
                    renderer.close()
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                renderer = mujoco.Renderer(model, height=height, width=width)
                option = mujoco.MjvOption()
                option.geomgroup[3:] = 0
                generation = version
            data.qpos[:], data.qvel[:], data.ctrl[:], data.time = qpos, qvel, ctrl, sim_time
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
            pixels = renderer.render()
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, format="JPEG", quality=85)
            if delay_s:
                # Used only by the timing regression test; never changes physics.
                stopping.wait(delay_s)
            elapsed = time.perf_counter()-start
            offer_latest(outputs, {"version": version, "jpeg": output.getvalue(), "stats": {
                "fps": round(1/max(start-previous, .001), 1), "frame_work_ms": round(elapsed*1000, 1),
                "frame_simulation_time_s": round(sim_time, 3), "frame_camera": camera, "frame_scene_version": version}})
            previous = start
            stopping.wait(max(0., (.5 if idle else 1/fps)-elapsed))
    except Exception as exc:
        offer_latest(outputs, {"error": f"Camera {type(exc).__name__}: {exc}"})
    finally:
        if renderer:
            renderer.close()
        # The parent may have exited abruptly (e.g. stop-lab.ps1).
        outputs.cancel_join_thread()
