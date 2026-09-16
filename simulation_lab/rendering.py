"""Camera process: no access to the authoritative physics state or controller."""
from __future__ import annotations

import io
import multiprocessing
import queue
import time

import mujoco
from PIL import Image


def render_profile(mode, width, height, shadows):
    """Choose presentation-only camera settings; physics is never affected."""
    if mode == "fast":
        return max(1, width // 2), max(1, height // 2), False, 0
    return width, height, shadows, 4


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
    profile = None
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
            xml, version, qpos, qvel, ctrl, sim_time, camera, shadows, preview_mode, idle = job
            frame_width, frame_height, frame_shadows, offsamples = render_profile(preview_mode, width, height, shadows)
            if generation != version or profile != preview_mode:
                if renderer:
                    renderer.close()
                model = mujoco.MjModel.from_xml_string(xml)
                model.vis.quality.offsamples = offsamples
                data = mujoco.MjData(model)
                renderer = mujoco.Renderer(model, height=frame_height, width=frame_width)
                option = mujoco.MjvOption()
                option.geomgroup[3:] = 0
                generation = version
                profile = preview_mode
            data.qpos[:], data.qvel[:], data.ctrl[:], data.time = qpos, qvel, ctrl, sim_time
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = frame_shadows
            pixels = renderer.render()
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, format="JPEG", quality=85)
            if delay_s:
                # Used only by the timing regression test; never changes physics.
                stopping.wait(delay_s)
            elapsed = time.perf_counter()-start
            offer_latest(outputs, {"version": version, "jpeg": output.getvalue(), "stats": {
                "fps": round(1/max(start-previous, .001), 1), "frame_work_ms": round(elapsed*1000, 1),
                "frame_simulation_time_s": round(sim_time, 3), "frame_camera": camera, "frame_scene_version": version,
                "preview_mode": preview_mode, "frame_resolution": [frame_width, frame_height]}})
            previous = start
            stopping.wait(max(0., (.5 if idle else 1/fps)-elapsed))
    except Exception as exc:
        offer_latest(outputs, {"error": f"Camera {type(exc).__name__}: {exc}"})
    finally:
        if renderer:
            renderer.close()
        # The parent may have exited abruptly (e.g. stop-lab.ps1).
        outputs.cancel_join_thread()
