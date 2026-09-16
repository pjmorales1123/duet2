"""Render a fast HD end-to-end physical dinner and pour demonstration."""
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.evaluate_dinner_scene import load
from simulation_lab.dinner_autonomy import DinnerSequence
from simulation_lab.pour_task import PourWaterTask
from simulation_lab.scene import HOME


def caption(frame, phase, stage, message):
    """Keep the review clip honest about the controller's live state."""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 52), fill=(10, 12, 34, 220))
    font = ImageFont.load_default()
    draw.text((12, 8), "Duet 2 · fast HD physical dinner + two-arm pour", fill=(255, 202, 103), font=font)
    draw.text((12, 28), f"{phase or 'starting'} · {stage or 'planning'} · {message[:72]}", fill=(245, 243, 255), font=font)
    return np.asarray(image)


def run_task(task, model, data, targets, renderer, writer, *, limit, record=False):
    """Advance a physical task; recording begins only after the table is set."""
    for step in range(limit):
        task.update(targets)
        if record and step % 60 == 0:
            renderer.update_scene(data, camera="overview")
            writer.append_data(caption(renderer.render().copy(), getattr(task, "phase", None), task.stage, task.message))
        if not task.active:
            return
        data.ctrl[:] = task.apply_gripper_limit(targets)
        mujoco.mj_step(model, data)


def main():
    output = Path(".run/two-arm-pour-hd-fast.mp4")
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        output.unlink()
    model, data, layout = load(seed=1000)
    for _ in range(200):
        mujoco.mj_step(model, data)
    data.time = 0.0
    targets = np.array(HOME * 2)
    with mujoco.Renderer(model, height=720, width=1280) as renderer, imageio.get_writer(output, fps=24, codec="libx264", quality=8) as writer:
        dinner = DinnerSequence(model, data, layout)
        dinner.start(kind="set_table")
        run_task(dinner, model, data, targets, renderer, writer, limit=60_000, record=True)
        if dinner.status != "succeeded":
            raise RuntimeError(dinner.message)
        pour = PourWaterTask(model, data, layout)
        pour.start()
        run_task(pour, model, data, targets, renderer, writer, limit=35_000, record=True)
    print(f"{output.resolve()}\n{pour.status}: {pour.message}")


if __name__ == "__main__":
    main()
