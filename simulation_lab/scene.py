"""Compose licensed SO-101 robots and a seeded, rigid-body laboratory scene."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

ASSETS = Path(__file__).parent / "assets" / "so101"
TABLE_Z = 0.76
HOME = [0.0, -0.70, 0.80, 0.20, 0.0, 0.65]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_LABELS = ["Base turn", "Shoulder", "Elbow", "Wrist bend", "Wrist rotation", "Gripper"]
COLORS = [(0.03, 0.62, 0.65), (0.25, 0.43, 0.88), (0.88, 0.43, 0.19), (0.64, 0.36, 0.81)]
CAMERAS = {"center": "Between the arms", "overview": "Overview", "overhead": "Overhead", "opposite": "Opposite side", "left_wrist_cam": "Left wrist", "right_wrist_cam": "Right wrist"}


def vec(values) -> str:
    return " ".join(f"{float(x):.8g}" for x in values)


@dataclass(frozen=True)
class RackPose:
    x: float
    y: float
    yaw_deg: float


def validate_layout(racks: list[RackPose]) -> None:
    if not 2 <= len(racks) <= 4:
        raise ValueError("Choose between two and four racks.")
    for i, rack in enumerate(racks):
        if not all(math.isfinite(v) for v in (rack.x, rack.y, rack.yaw_deg)):
            raise ValueError("Rack positions must be finite numbers.")
        if abs(rack.x) > 0.30 or not -0.015 <= rack.y <= 0.30 or abs(rack.yaw_deg) > 180:
            raise ValueError("Keep rack centers inside the marked tabletop workspace.")
        for other in racks[:i]:
            if math.hypot(rack.x - other.x, rack.y - other.y) < 0.20:
                raise ValueError("Rack centers must be at least 20 cm apart. Move a rack farther away.")


def random_layout(seed: int, count: int) -> list[RackPose]:
    if not 2 <= count <= 4:
        raise ValueError("Choose between two and four racks.")
    anchors = [(-0.18, 0.075), (0.18, 0.075), (0.0, 0.265)] if count == 3 else [(-0.18, 0.065), (0.18, 0.065), (-0.18, 0.270), (0.18, 0.270)][:count]
    rng = np.random.default_rng(seed)
    for _ in range(100):
        racks = [RackPose(float(x + rng.uniform(-0.013, 0.013)), float(y + rng.uniform(-0.012, 0.012)), float(rng.uniform(-55, 55))) for x, y in anchors]
        try:
            validate_layout(racks)
            return racks
        except ValueError:
            continue
    raise ValueError("Could not generate a separated layout.")


def practice_layout(seed: int) -> list[RackPose]:
    """Two accessible holders; small seed variations for the first manipulation skill."""
    rng = np.random.default_rng(seed)
    return [RackPose(x + float(rng.uniform(-.004, .004)),
                     .090 + float(rng.uniform(-.004, .004)),
                     yaw + float(rng.uniform(-4, 4)))
            for x, yaw in ((-.18, -15), (.18, 15))]


def transfer_layout(seed: int, side: str) -> list[RackPose]:
    """Two holders in one arm's workspace, leaving room for a carried tube."""
    if side not in ("left", "right"):
        raise ValueError("Choose left or right for the transfer practice layout.")
    rng = np.random.default_rng(seed)
    mirror = 1 if side == "left" else -1
    return [RackPose(mirror*(x + float(rng.uniform(-.003, .003))),
                     .09 + float(rng.uniform(-.003, .003)),
                     mirror*(yaw + float(rng.uniform(-3, 3))))
            for x, yaw in ((-.28, -15), (-.055, 15))]


def rack_slot_xy(pose, slot: int):
    theta = math.radians(pose.yaw_deg)
    dx, dy = (slot % 3 - 1)*.045, (-.5 if slot < 3 else .5)*.046
    return [pose.x + math.cos(theta)*dx - math.sin(theta)*dy,
            pose.y + math.sin(theta)*dx + math.cos(theta)*dy]


def add_camera(world, name, position, target, fovy=52):
    direction = np.asarray(target) - np.asarray(position)
    direction = direction / np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(direction, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    ET.SubElement(world, "camera", name=name, pos=vec(position), xyaxes=vec([*right, *up]), fovy=str(fovy))


def geom(parent, name, kind, size, pos=(0, 0, 0), **attrs):
    return ET.SubElement(parent, "geom", name=name, type=kind, size=vec(size), pos=vec(pos), **{k: str(v) for k, v in attrs.items()})


def build_scene(seed: int = 42, count: int = 3, racks: list[RackPose] | None = None, practice: bool = False, transfer_side: str | None = None, scenario: str = "chemistry", dinner_preset: str = "task", drawer_open: bool = False, dinner_variation: str = "duet_v1") -> tuple[str, dict]:
    if scenario not in ("chemistry", "dinner"):
        raise ValueError("Choose the dinner or chemistry scene.")
    if scenario == "dinner" and (racks is not None or practice or transfer_side is not None):
        raise ValueError("Rack layouts and tube practice belong to the chemistry scene.")
    if transfer_side not in (None, "left", "right"):
        raise ValueError("Unknown transfer practice side.")
    practice = practice or transfer_side is not None
    racks = [] if scenario == "dinner" else racks or (transfer_layout(seed, transfer_side) if transfer_side else practice_layout(seed) if practice else random_layout(seed, count))
    if scenario == "chemistry":
        validate_layout(racks)
    rng = np.random.default_rng(seed ^ 0x51A7)
    source = ET.parse(ASSETS / "so101.xml").getroot()
    root = ET.Element("mujoco", model="benchlab_dual_so101")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str((ASSETS / "assets").resolve()), autolimits="true")
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast", cone="elliptic", iterations="30", ls_iterations="20", gravity="0 0 -9.81")
    ET.SubElement(root, "size", njmax="4000", nconmax="1000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "quality", shadowsize="1024", offsamples="1")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.50 0.50 0.50", specular="0.20 0.20 0.20")
    ET.SubElement(visual, "rgba", haze="0.16 0.20 0.26 1")
    root.append(deepcopy(source.find("default")))
    asset = deepcopy(source.find("asset"))
    # Use source meshes for presentation and RGB training data. The optional LOD
    # assets make the robot visibly faceted and are reserved for a later speed mode.
    root.append(asset)
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.15 0.21 0.29", rgb2="0.04 0.07 0.12", width="512", height="3072")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", width="512", height="512", rgb1="0.18 0.22 0.28", rgb2="0.20 0.24 0.30")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance="0.05")
    ET.SubElement(asset, "material", name="table_mat", rgba="0.72 0.79 0.81 1", specular="0.25", shininess="0.3")
    ET.SubElement(asset, "material", name="edge_mat", rgba="0.07 0.12 0.17 1", specular="0.45")
    ET.SubElement(asset, "material", name="tube_mat", rgba="0.71 0.88 0.94 0.78", specular="0.6", shininess="0.8")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-0.7 -0.3 2.5", dir="0.2 0.2 -1", diffuse="0.75 0.77 0.8", castshadow="true")
    ET.SubElement(world, "light", pos="0.8 0.6 1.8", dir="-0.4 -0.3 -1", diffuse="0.5 0.55 0.6", castshadow="false")
    geom(world, "floor", "plane", (3, 3, 0.1), material="floor_mat")
    geom(world, "table", "box", (0.48, 0.39, 0.022), (0, 0.07, TABLE_Z - 0.022), material="table_mat", friction="0.8 0.01 0.001")
    geom(world, "table_edge", "box", (0.483, 0.393, 0.009), (0, 0.07, TABLE_Z - 0.048), material="edge_mat")
    for x in [-0.42, 0.42]:
        for y in [-0.24, 0.38]:
            geom(world, f"leg_{x}_{y}", "box", (0.02, 0.02, 0.34), (x, y, 0.36), material="edge_mat")
    # Thin reference grid is visual only, so it cannot interfere with contacts.
    for axis in (0, 1):
        for index in range(-4, 5):
            coordinate = index * 0.10
            size = (0.0005, 0.35, 0.0001) if axis == 0 else (0.45, 0.0005, 0.0001)
            position = (coordinate, 0.07, TABLE_Z + 0.0001) if axis == 0 else (0, coordinate + 0.07, TABLE_Z + 0.0001)
            geom(world, f"grid_{axis}_{index}", "box", size, position, rgba="0.28 0.42 0.47 0.25", contype="0", conaffinity="0")

    actuator = ET.SubElement(root, "actuator")
    for side, x, color in [("left", -0.25, (0.88, 0.89, 0.90)), ("right", 0.25, (0.88, 0.89, 0.90))]:
        geom(world, f"{side}_base_pad", "box", (0.058, 0.055, 0.009), (x, -0.235, TABLE_Z + 0.009), material="edge_mat")
        body = deepcopy(source.find("./worldbody/body"))
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", f"{side}_{element.get('name')}")
            if element.tag == "geom" and element.get("class") == "visual":
                mat = element.get("material", "")
                if not mat.startswith("sts3215"):
                    element.attrib.pop("material", None)
                    element.set("rgba", vec([*color, 1]))
        body.set("pos", vec((x, -0.235, TABLE_Z + 0.018)))
        body.set("quat", vec((math.sqrt(0.5), 0, 0, math.sqrt(0.5))))
        world.append(body)
        for source_actuator in source.find("actuator"):
            item = deepcopy(source_actuator)
            item.set("name", f"{side}_{item.get('name')}")
            item.set("joint", f"{side}_{item.get('joint')}")
            actuator.append(item)

    tube_records = []
    rack_records = []
    slot_records = []
    for index, pose in enumerate(racks):
        label = chr(65 + index)
        name = f"rack_{label}"
        color = COLORS[index]
        theta = math.radians(pose.yaw_deg)
        rack = ET.SubElement(world, "body", name=name, pos=vec((pose.x, pose.y, TABLE_Z)), euler=vec((0, 0, theta)))
        rgba = vec([*color, 1])
        geom(rack, name + "_base", "box", (0.077, 0.055, 0.005), (0, 0, 0.005), rgba=rgba, friction="0.9 0.01 0.001")
        for px in (-0.071, 0.071):
            for py in (-0.049, 0.049):
                geom(rack, f"{name}_post_{px}_{py}", "cylinder", (0.004, 0.025), (px, py, 0.032), rgba=rgba)
        for layer_z in (0.022, 0.057):
            for py in (-0.046, 0, 0.046):
                geom(rack, f"{name}_rail_y_{layer_z}_{py}", "box", (0.077, 0.012 if practice else 0.010, 0.003), (0, py, layer_z), rgba=rgba)
            for px in (-0.0675, -0.0225, 0.0225, 0.0675):
                geom(rack, f"{name}_rail_x_{layer_z}_{px}", "box", (0.0115 if practice else 0.0095, 0.055, 0.003), (px, 0, layer_z), rgba=rgba)
        # Exposed front tube, two rear tubes, and 22 mm guide openings in practice.
        # All props remain free rigid bodies with ordinary contacts.
        occupied = [1, 3, 5] if practice else sorted(int(n) for n in rng.choice(6, size=int(rng.integers(4, 7)), replace=False))
        if transfer_side and index == 1:
            occupied = [3, 5]  # Exposed empty front row in the destination rack.
        rack_records.append({"id": label, **asdict(pose), "color": "#" + "".join(f"{int(c * 255):02x}" for c in color), "slots": occupied})
        for slot in range(6):
            slot_records.append({"id": f"{label}{slot+1}", "rack": label, "index": slot,
                                 "position_m": [*rack_slot_xy(pose, slot), TABLE_Z+.010]})
        for slot in occupied:
            px, py = rack_slot_xy(pose, slot)
            height = float(rng.uniform(0.102, 0.106) if practice else rng.uniform(0.088, 0.108))
            tube_name = f"tube_{label}{slot + 1}"
            center_z = TABLE_Z + 0.011 + height / 2
            tube = ET.SubElement(world, "body", name=tube_name, pos=vec((px, py, center_z)))
            ET.SubElement(tube, "freejoint", name=tube_name + "_free")
            # Flat-bottom training tubes stand on their own; rounded test tubes
            # remain available in the general randomized scene.
            geom(tube, tube_name + "_shell", "cylinder" if practice else "capsule", (0.009, height / 2 if practice else height / 2 - 0.009), material="tube_mat", mass="0.018", friction="0.8 0.01 0.001", condim="4", solref="0.012 1")
            sample_color = COLORS[(slot + index) % len(COLORS)]
            geom(tube, tube_name + "_sample", "cylinder", (0.0085, height * 0.21), (0, 0, -height * 0.13), rgba=vec([*sample_color, 0.90]), contype="0", conaffinity="0", mass="0")
            geom(tube, tube_name + "_cap", "cylinder", (0.0105, 0.006), (0, 0, height / 2 - 0.003), rgba=vec([*sample_color, 1]), mass="0.002")
            geom(tube, tube_name + "_label", "cylinder", (0.00915, 0.010), (0, 0, height * 0.14), rgba="0.96 0.98 0.99 1", contype="0", conaffinity="0", mass="0")
            tube_records.append({"id": f"{label}{slot + 1}", "body": tube_name, "rack": label, "slot": slot, "height": height, "initial_position": [px, py, center_z]})

    add_camera(world, "center", (0, -0.57, 1.14), (0, 0.10, 0.83), 59)
    add_camera(world, "overview", (0.83, -0.77, 1.62), (0, 0.08, 0.82), 46)
    add_camera(world, "overhead", (0, 0.07, 1.75), (0, 0.07, TABLE_Z), 48)
    add_camera(world, "opposite", (0, .78, 1.48), (0, -.035, .82), 55)
    layout = {"seed": seed, "scenario": scenario, "racks": rack_records, "slots": slot_records, "tubes": tube_records, "table_z": TABLE_Z, "practice": practice, "transfer_side": transfer_side}
    if scenario == "dinner":
        from .dinner import add_dinner_scene
        layout.update(add_dinner_scene(root, seed, dinner_preset, drawer_open, dinner_variation))
    return ET.tostring(root, encoding="unicode"), layout
