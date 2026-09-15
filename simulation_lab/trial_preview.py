"""A passive display renderer with its own model and OpenGL-owning thread."""
from queue import Empty, Queue
from threading import Thread

import mujoco


class TrialPreview:
    """Keep display GL resources separate from the sequential policy renderers."""
    def __init__(self, xml, width=640, height=360):
        self.inputs, self.outputs = Queue(maxsize=1), Queue(maxsize=1)
        self.closed = False
        self.thread = Thread(target=self._run, args=(xml, width, height), daemon=True,
                             name='talos-passive-preview')
        self.thread.start()

    def _run(self, xml, width, height):
        renderer = None
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            model.vis.quality.offsamples = 0
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=height, width=width)
            option = mujoco.MjvOption()
            option.geomgroup[3:] = 0
            while True:
                values = self.inputs.get()
                if values is None:
                    return
                qpos, qvel, ctrl, simulation_time, camera = values
                data.qpos[:], data.qvel[:], data.ctrl[:], data.time = qpos, qvel, ctrl, simulation_time
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                self.outputs.put((True, renderer.render().copy()))
        except Exception as exc:
            self.outputs.put((False, type(exc).__name__))
        finally:
            if renderer is not None:
                renderer.close()

    def frame(self, data, camera):
        if self.closed:
            raise RuntimeError('The trial preview is closed.')
        self.inputs.put((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time), camera), timeout=20)
        try:
            ok, result = self.outputs.get(timeout=20)
        except Empty:
            raise RuntimeError('The trial preview did not return a frame.') from None
        if not ok:
            raise RuntimeError('The trial preview could not render: '+result)
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.thread.is_alive():
            # A timed-out render can leave a request in the one-slot queue.
            try:
                self.inputs.get_nowait()
            except Empty:
                pass
            self.inputs.put_nowait(None)
            self.thread.join(timeout=2)
