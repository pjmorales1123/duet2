r"""Interactive exact-coordinate editor for the Duet 2 dinner table.

Usage:
    .venv\Scripts\python.exe scripts\design_dinner_layout.py --seed 42
"""
from __future__ import annotations

import argparse
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco

from simulation_lab.dinner import OBJECTS
from simulation_lab.dinner_layout import load_dinner_layout, save_dinner_layout
from simulation_lab.scene import HOME, JOINTS, TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_HALF_X, TABLE_HALF_Y, TABLE_Z, build_scene


COLORS = {
    "plate": "#cf5047",
    "side_plate": "#f2b74d",
    "mug": "#5b7dd1",
    "glass": "#b5c9ff",
    "bottle": "#702c5d",
    "fork": "#c5cad6",
    "spoon": "#aeb8cd",
}


def current_scene(seed: int, arrangement: str = "reference") -> tuple[dict, dict]:
    """Load a settled dinner arrangement and robot pose without starting a server."""
    if arrangement not in ("task", "reference"):
        raise ValueError("Choose task or reference for the dinner arrangement.")
    xml, layout = build_scene(seed=seed, scenario="dinner", dinner_preset=arrangement)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME * 2
    data.ctrl[:] = HOME * 2
    for _ in range(200):
        mujoco.mj_step(model, data)

    objects = {}
    for item in layout["objects"]:
        body = data.body(item["body"])
        objects[item["id"]] = {
            "position_m": [float(value) for value in body.xpos],
            "yaw_rad": float(math.atan2(body.xmat[3], body.xmat[0])),
        }
    robots = {}
    for side in ("left", "right"):
        base = data.body(f"{side}_base")
        gripper = data.site(f"{side}_gripperframe")
        joint_values = []
        for joint_name in JOINTS:
            joint = model.joint(f"{side}_{joint_name}")
            joint_values.append(float(data.qpos[joint.qposadr[0]]))
        robots[side] = {
            "base_position_m": [float(value) for value in base.xpos],
            "gripper_position_m": [float(value) for value in gripper.xpos],
            "joint_names": list(JOINTS),
            "joint_positions_rad": joint_values,
        }
    return objects, robots


def current_poses(seed: int, arrangement: str = "reference") -> dict:
    """Compatibility helper returning only the selected tableware poses."""
    return current_scene(seed, arrangement)[0]


class DinnerLayoutEditor(tk.Tk):
    """Drag dinner objects on a true-to-scale tabletop and export body poses."""

    def __init__(self, seed: int, arrangement: str, output: Path, loaded: dict | None = None):
        super().__init__()
        self.title("Duet 2 exact dinner layout")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.output = output
        self.seed = seed
        self.arrangement = arrangement
        self.poses, self.robots = current_scene(seed, arrangement)
        if loaded:
            for object_id, pose in loaded["objects"].items():
                self.poses[object_id] = pose
            self.robots.update(loaded.get("robots", {}))
        self.selected = "plate"
        self.active_drag: str | None = None
        self._field_vars = {key: tk.StringVar() for key in ("x", "y", "z", "yaw")}
        self._build_widgets()
        self._redraw()
        self._select_object(self.selected)

    @property
    def canvas_scale(self) -> float:
        return min((self.canvas.winfo_width() - 44) / (2 * TABLE_HALF_X),
                   (self.canvas.winfo_height() - 64) / (2 * TABLE_HALF_Y))

    def _build_widgets(self) -> None:
        self.canvas = tk.Canvas(self, background="#172034", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=12)
        self.canvas.bind("<Button-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        panel = ttk.Frame(self, padding=12)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(panel, text="Duet 2 dinner layout", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            panel,
            text=f"Standalone editor · {self.arrangement} arrangement · seed {self.seed}\n"
                 f"Table: {2 * TABLE_HALF_X:.3f} m x {2 * TABLE_HALF_Y:.3f} m\n"
                 f"Center: ({TABLE_CENTER_X:.3f}, {TABLE_CENTER_Y:.3f})\n"
                 f"Top surface z: {TABLE_Z:.3f} m",
        ).pack(anchor="w", pady=(6, 12))
        ttk.Label(panel, text="Select an item").pack(anchor="w")
        self.item_list = tk.Listbox(panel, height=7, exportselection=False)
        for object_id in OBJECTS:
            self.item_list.insert(tk.END, object_id)
        self.item_list.pack(fill=tk.X, pady=(3, 12))
        self.item_list.bind("<<ListboxSelect>>", self._list_select)
        robot_text = "\n".join(
            f"{side.title()} gripper: ({pose['gripper_position_m'][0]:.4f}, "
            f"{pose['gripper_position_m'][1]:.4f}, {pose['gripper_position_m'][2]:.4f})"
            for side, pose in self.robots.items()
        )
        ttk.Label(panel, text="Robots · current settled pose").pack(anchor="w")
        ttk.Label(panel, text=robot_text, justify=tk.LEFT, font=("Consolas", 9)).pack(anchor="w", pady=(3, 12))

        ttk.Label(panel, text="Exact body pose (world coordinates)").pack(anchor="w")
        for key, label in (("x", "x (m)"), ("y", "y (m)"), ("z", "z (m)"), ("yaw", "yaw (rad)")):
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=self._field_vars[key], width=16).pack(side=tk.RIGHT)
        ttk.Button(panel, text="Apply coordinates", command=self._apply_fields).pack(fill=tk.X, pady=(8, 4))
        rotate_row = ttk.Frame(panel)
        rotate_row.pack(fill=tk.X)
        ttk.Button(rotate_row, text="Rotate -15 deg", command=lambda: self._rotate(-15)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(rotate_row, text="Rotate +15 deg", command=lambda: self._rotate(15)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(panel, text="Drag with the mouse for x/y. Edit z and yaw for exact placement.", wraplength=260).pack(anchor="w", pady=(12, 10))

        ttk.Label(panel, text="Output JSON").pack(anchor="w")
        self.output_var = tk.StringVar(value=str(self.output))
        ttk.Entry(panel, textvariable=self.output_var, width=38).pack(fill=tk.X, pady=(3, 5))
        ttk.Button(panel, text="Export exact coordinates", command=self._save).pack(fill=tk.X)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(panel, textvariable=self.status, wraplength=260).pack(anchor="w", pady=(12, 0))

    def _world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        scale = self.canvas_scale
        left = (self.canvas.winfo_width() - 2 * TABLE_HALF_X * scale) / 2
        bottom = (self.canvas.winfo_height() + 2 * TABLE_HALF_Y * scale) / 2
        return left + (x - TABLE_CENTER_X + TABLE_HALF_X) * scale, bottom - (y - TABLE_CENTER_Y + TABLE_HALF_Y) * scale

    def _canvas_to_world(self, px: float, py: float) -> tuple[float, float]:
        scale = self.canvas_scale
        left = (self.canvas.winfo_width() - 2 * TABLE_HALF_X * scale) / 2
        bottom = (self.canvas.winfo_height() + 2 * TABLE_HALF_Y * scale) / 2
        return TABLE_CENTER_X - TABLE_HALF_X + (px - left) / scale, TABLE_CENTER_Y + TABLE_HALF_Y - (bottom - py) / scale

    def _redraw(self) -> None:
        if not hasattr(self, "canvas") or self.canvas.winfo_width() < 100:
            return
        self.canvas.delete("all")
        x0, y0 = self._world_to_canvas(TABLE_CENTER_X - TABLE_HALF_X, TABLE_CENTER_Y - TABLE_HALF_Y)
        x1, y1 = self._world_to_canvas(TABLE_CENTER_X + TABLE_HALF_X, TABLE_CENTER_Y + TABLE_HALF_Y)
        self.canvas.create_rectangle(x0, y1, x1, y0, fill="#8799ae", outline="#d7e1ec", width=2)
        for index in range(-4, 5):
            x = self._world_to_canvas(index * .1, TABLE_CENTER_Y)[0]
            self.canvas.create_line(x, y1, x, y0, fill="#71869d")
            y = self._world_to_canvas(TABLE_CENTER_X, TABLE_CENTER_Y + index * .1)[1]
            self.canvas.create_line(x0, y, x1, y, fill="#71869d")
        self.canvas.create_text(x0 + 6, y1 + 14, anchor="w", text="x/y in metres; table top z = 0.760", fill="#f4f7fb")
        for object_id, pose in self.poses.items():
            self._draw_object(object_id, pose)
        for side, pose in self.robots.items():
            self._draw_robot(side, pose)

    def _draw_robot(self, side: str, pose: dict) -> None:
        base_x, base_y = self._world_to_canvas(*pose["base_position_m"][:2])
        grip_x, grip_y = self._world_to_canvas(*pose["gripper_position_m"][:2])
        color = "#d95d52" if side == "left" else "#4f82d9"
        self.canvas.create_rectangle(base_x - 18, base_y - 14, base_x + 18, base_y + 14,
                                     fill=color, outline="#f6f8fb", width=2,
                                     tags=("robot", f"robot:{side}"))
        self.canvas.create_line(base_x, base_y, grip_x, grip_y, fill=color, width=7,
                                tags=("robot", f"robot:{side}"))
        self.canvas.create_oval(grip_x - 6, grip_y - 6, grip_x + 6, grip_y + 6,
                                fill="#f6f8fb", outline=color, width=2,
                                tags=("robot", f"robot:{side}"))
        self.canvas.create_text(base_x, base_y, text=f"{side} arm", fill="#101521")

    def _draw_object(self, object_id: str, pose: dict) -> None:
        x, y = pose["position_m"][:2]
        px, py = self._world_to_canvas(x, y)
        size_x, size_y = OBJECTS[object_id]["size_m"][:2]
        radius = max(size_x, size_y) * self.canvas_scale / 2
        color = COLORS[object_id]
        tags = ("object", f"object:{object_id}")
        selected = object_id == self.selected
        outline = "#ffffff" if selected else "#263344"
        width = 3 if selected else 1
        if object_id in ("plate", "side_plate", "glass", "bottle"):
            self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline=outline, width=width, tags=tags)
        elif object_id == "mug":
            self.canvas.create_oval(px - radius * 1.15, py - radius * .8, px + radius * 1.15, py + radius * .8, fill=color, outline=outline, width=width, tags=tags)
        else:
            angle = pose["yaw_rad"]
            length = size_y * self.canvas_scale
            dx, dy = math.sin(angle) * length / 2, -math.cos(angle) * length / 2
            self.canvas.create_line(px - dx, py - dy, px + dx, py + dy, fill=color, width=max(5, int(size_x * self.canvas_scale)), tags=tags)
        self.canvas.create_text(px, py, text=object_id, fill="#101521", tags=tags)

    def _list_select(self, _event=None) -> None:
        selection = self.item_list.curselection()
        if selection:
            self._select_object(self.item_list.get(selection[0]))

    def _select_object(self, object_id: str) -> None:
        self.selected = object_id
        index = list(OBJECTS).index(object_id)
        self.item_list.selection_clear(0, tk.END)
        self.item_list.selection_set(index)
        self.item_list.activate(index)
        pose = self.poses[object_id]
        self._field_vars["x"].set(f"{pose['position_m'][0]:.9f}")
        self._field_vars["y"].set(f"{pose['position_m'][1]:.9f}")
        self._field_vars["z"].set(f"{pose['position_m'][2]:.9f}")
        self._field_vars["yaw"].set(f"{pose['yaw_rad']:.9f}")
        self._redraw()

    def _canvas_press(self, event) -> None:
        item = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(item)
        object_tags = [tag for tag in tags if tag.startswith("object:")]
        if object_tags:
            self._select_object(object_tags[0].split(":", 1)[1])
            self.active_drag = self.selected

    def _canvas_drag(self, event) -> None:
        if self.active_drag is None:
            return
        x, y = self._canvas_to_world(event.x, event.y)
        self.poses[self.active_drag]["position_m"][:2] = [x, y]
        self._select_object(self.active_drag)

    def _canvas_release(self, _event) -> None:
        self.active_drag = None

    def _apply_fields(self) -> bool:
        try:
            x = float(self._field_vars["x"].get())
            y = float(self._field_vars["y"].get())
            z = float(self._field_vars["z"].get())
            yaw = float(self._field_vars["yaw"].get())
            if not all(math.isfinite(value) for value in (x, y, z, yaw)):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid coordinates", "Enter finite numeric values for x, y, z, and yaw.")
            return False
        self.poses[self.selected] = {"position_m": [x, y, z], "yaw_rad": yaw}
        self._redraw()
        self.status.set(f"Updated {self.selected}.")
        return True

    def _rotate(self, degrees: float) -> None:
        pose = self.poses[self.selected]
        pose["yaw_rad"] += math.radians(degrees)
        self._select_object(self.selected)

    def _save(self) -> None:
        if not self._apply_fields():
            return
        destination = Path(self.output_var.get()).expanduser()
        if not destination.is_absolute():
            destination = ROOT / destination
        save_dinner_layout(destination, self.poses, self.robots)
        self.output = destination
        self.status.set(f"Saved exact coordinates to {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Scene seed used for the arrangement.")
    parser.add_argument("--arrangement", choices=("reference", "task"), default="reference",
                        help="Load the intended target arrangement (default) or the task start.")
    parser.add_argument("--output", type=Path, default=Path(".run/dinner-layout.json"))
    parser.add_argument("--load", type=Path, help="Load an existing dinner layout before editing.")
    args = parser.parse_args()
    loaded = load_dinner_layout(args.load, tuple(OBJECTS)) if args.load else None
    app = DinnerLayoutEditor(args.seed, args.arrangement, args.output, loaded)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
