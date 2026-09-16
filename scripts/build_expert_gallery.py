"""Render one MP4 per seed in a collected dataset (all 5 skills back to back),
plus a manifest.json the demo dashboard reads to build its seed gallery.

Usage:
  python scripts/build_expert_gallery.py --input datasets/duet-micro-v1 \
      --output simulation_lab/web/media/expert
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ponytail: shell out to the ffmpeg already on PATH instead of adding an
# imageio/imageio-ffmpeg dependency just to wrap the same binary.
FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    raise SystemExit("ffmpeg not found on PATH.")

SKILL_ORDER = ["bottle", "plate", "mug", "fork", "spoon"]
INSTRUCTIONS = {
    "bottle": "Place the bottle on the table.",
    "plate": "Place the dinner plate.",
    "mug": "Place the mug above the plate.",
    "fork": "Place the fork beside the plate.",
    "spoon": "Place the spoon beside the plate.",
}
SIZE = (640, 480)


def add_caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 34), fill=(10, 12, 34, 210))
    draw.text((10, 6), lines[0], fill=(255, 202, 103), font=font, stroke_width=1, stroke_fill=(10, 12, 34))
    if len(lines) > 1:
        draw.text((10, 20), lines[1], fill=(245, 243, 255), font=font, stroke_width=1, stroke_fill=(10, 12, 34))
    return np.asarray(image)


def summarize_variation(randomization: dict) -> str:
    """One line describing how this seed's spawn differs from a neutral layout,
    so the gallery doesn't just look like 10 copies of the same table."""
    offsets = randomization["object_offsets_m"]
    magnitudes_cm = {obj: (dx**2 + dy**2) ** 0.5 * 100 for obj, (dx, dy) in offsets.items()}
    worst_obj = max(magnitudes_cm, key=magnitudes_cm.get)
    max_yaw_deg = max(abs(np.degrees(v)) for v in randomization["object_yaw_rad"].values())
    light_pct = round(randomization["light_multiplier"] * 100)
    return (f"{worst_obj} spawn shifted {magnitudes_cm[worst_obj]:.1f} cm, rotations up to "
            f"{max_yaw_deg:.0f}°, lighting at {light_pct}% brightness, unique table tint.")


def title_card(lines: list[str]) -> np.ndarray:
    image = Image.new("RGB", SIZE, (17, 17, 53))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 200
    for index, line in enumerate(lines):
        color = (255, 139, 120) if index == 0 else (245, 243, 255)
        draw.text((40, y), line, fill=color, font=font)
        y += 22
    return np.asarray(image)


def build_seed_video(seed_dir: Path, output_path: Path) -> dict:
    seed_label = seed_dir.name.removeprefix("seed-")
    skills_present, changes = [], ""
    frames: list[np.ndarray] = []
    for skill in SKILL_ORDER:
        skill_dir = seed_dir / skill
        manifest_path, obs_path = skill_dir / "manifest.json", skill_dir / "observations.npz"
        if not (manifest_path.exists() and obs_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("training_eligible"):
            continue
        if not changes:
            changes = summarize_variation(manifest["layout"]["randomization"])
        instruction = manifest["language"]["instruction"]
        with np.load(obs_path) as obs:
            raw_frames = obs["overhead"]
        for raw in raw_frames:
            resized = np.asarray(Image.fromarray(raw).resize(SIZE, Image.Resampling.NEAREST))
            frames.append(add_caption(resized, [instruction, f"skill: {skill}"]))
        skills_present.append(skill)

    frames = [title_card([f"DUET 2 - seed {seed_label}", "Scripted teacher policy - overview camera", changes])] * 20 + frames
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(Path(tmp) / f"f{index:05d}.png")
        output_path.unlink(missing_ok=True)
        subprocess.run(
            [FFMPEG, "-y", "-framerate", "10", "-i", str(Path(tmp) / "f%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
            check=True, capture_output=True,
        )
    return {"seed": seed_label, "skills": skills_present, "video": output_path.name,
            "changes": changes, "instructions": {skill: INSTRUCTIONS[skill] for skill in skills_present}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for seed_dir in sorted(args.input.glob("seed-*")):
        video_path = args.output / f"{seed_dir.name}.mp4"
        entries.append(build_seed_video(seed_dir, video_path))
        print(f"built {video_path}", flush=True)

    (args.output / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} seed videos + manifest to {args.output}")


if __name__ == "__main__":
    main()
