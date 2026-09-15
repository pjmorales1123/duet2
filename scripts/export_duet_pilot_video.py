"""Render a transparent video summary from a recorded Duet 2 demonstration batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SKILL_TITLES = {
    "bottle": "Place the plum carafe",
    "plate": "Place the coral dinner plate",
    "mug": "Place the cobalt cup",
    "drawer": "Open the service drawer",
    "fork": "Set the service fork",
    "spoon": "Set the service spoon",
}


def parse_args() -> argparse.Namespace:
    """Read paths for an existing recorded batch and a new MP4 output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_text(image: Image.Image, lines: list[str], *, title: bool = False) -> Image.Image:
    """Add a high-contrast caption band without altering the recorded pixels elsewhere."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    line_height = 18 if title else 15
    band_height = 76 if title else 46
    draw.rectangle((0, 0, canvas.width, band_height), fill=(10, 12, 34, 210))
    y = 10
    for index, line in enumerate(lines):
        color = (255, 202, 103) if index == 0 else (245, 243, 255)
        draw.text((14, y), line, fill=color, font=font, stroke_width=1, stroke_fill=(10, 12, 34))
        y += line_height
    return canvas


def title_frame(lines: list[str], size: tuple[int, int]) -> np.ndarray:
    """Create an explicit context card so the video is not mistaken for a live UI trace."""
    image = Image.new("RGB", size, (17, 17, 53))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 170
    for index, line in enumerate(lines):
        color = (255, 139, 120) if index == 0 else (245, 243, 255)
        draw.text((70, y), line, fill=color, font=font)
        y += 34
    return np.asarray(image)


def recorded_frames(observations_path: Path) -> list[np.ndarray]:
    """Load the overhead camera frames saved by the physical demonstration collector."""
    with np.load(observations_path) as observations:
        overhead = observations["overhead"]
    return [np.asarray(frame) for frame in overhead]


def main() -> None:
    """Render all recorded skill clips in collection order as one MP4."""
    args = parse_args()
    batch = args.batch.resolve()
    output = args.output.resolve()
    if not batch.is_dir():
        raise FileNotFoundError(f"Recorded batch not found: {batch}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    clips = sorted(path for path in batch.iterdir() if path.is_dir() and (path / "observations.npz").exists())
    # Plain skill folder names must follow physical execution, not alphabetic order.
    order = {skill: index for index, skill in enumerate(SKILL_TITLES)}
    clips.sort(key=lambda path: order.get(path.name.split('-', 1)[-1], len(order)))
    if not clips:
        raise RuntimeError(f"No recorded skill clips found in {batch}")

    size = (960, 720)
    frames: list[np.ndarray] = []
    frames.extend([title_frame([
        "DUET 2 — seed 1000 physical demonstration",
        "Task context: set the table",
        "Recorded overhead-camera trajectories; not a live text-command UI trace.",
    ], size)] * 25)

    for index, clip in enumerate(clips, start=1):
        skill = clip.name.split("-", 1)[-1]
        caption = SKILL_TITLES.get(skill, skill.replace("_", " ").title())
        raw_frames = recorded_frames(clip / "observations.npz")
        for raw in raw_frames:
            image = Image.fromarray(raw).resize(size, Image.Resampling.NEAREST)
            image = add_text(image, [f"{index}/{len(clips)}  {caption}", "Seed 1000 · recorded physical teacher · overhead camera"])
            frames.append(np.asarray(image))

    with imageio.get_writer(output, fps=10, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Rendered {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
