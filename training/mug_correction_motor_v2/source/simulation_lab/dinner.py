"""Original procedural dinner assets and a passive cutlery drawer.

All lengths are metres and masses are kilograms. Hollow vessels use separate
wall segments: no convex collision hull closes their mouths or handles.
No runtime object placement, attachments, or liquid dynamics are implemented.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scene import TABLE_Z, geom, vec

DRAWER_TRAVEL = .120
OBJECTS = {
    "plate": {"label": "Blue-rim plate", "kind": "plate", "mass_kg": .065, "size_m": [.132, .132, .024], "color": "#719acb", "grasp_local_m": [0, -.061, .010], "grasp_width_m": .009},
    "side_plate": {"label": "Gold-rim side plate", "kind": "plate", "mass_kg": .045, "size_m": [.106, .106, .012], "color": "#dfac65", "grasp_local_m": [0, -.048, .009], "grasp_width_m": .009},
    "mug": {"label": "Teal mug", "kind": "mug", "mass_kg": .045, "size_m": [.077, .050, .064], "color": "#54b9b1", "grasp_local_m": [.047, 0, .041], "grasp_width_m": .008},
    "glass": {"label": "Clear drinking glass", "kind": "glass", "mass_kg": .035, "size_m": [.048, .048, .074], "color": "#b3dcdf", "grasp_local_m": [0, 0, .045], "grasp_width_m": .048},
    "bottle": {"label": "Amber water bottle", "kind": "bottle", "mass_kg": .080, "size_m": [.048, .048, .140], "color": "#cf995b", "grasp_local_m": [0, 0, .128], "grasp_width_m": .022},
    "fork": {"label": "Fork", "kind": "fork", "mass_kg": .012, "size_m": [.017, .110, .016], "color": "#cad5de", "grasp_local_m": [0, 0, .008], "grasp_width_m": .009},
    "spoon": {"label": "Spoon", "kind": "spoon", "mass_kg": .014, "size_m": [.022, .110, .016], "color": "#cad5de", "grasp_local_m": [0, 0, .008], "grasp_width_m": .009},
}


def left_reach_bottle_pose(seed):
    """Declared practice reset; real motion begins only after this setup."""
    rng=np.random.default_rng(seed)
    return {'x':float(-.0734067766+rng.uniform(-.006,.006)),
            'y':float(-.1215734485+rng.uniform(-.006,.006)),
            'yaw':float(.3096408+rng.uniform(-.08,.08))}


def surface(parent, name, kind, size, pos=(0, 0, 0), **attrs):
    """Explicit mass is assigned once per body; geometry carries collisions."""
    return geom(parent, name, kind, size, pos, mass="0", friction=".8 .005 .0001",
                condim="4", solref=".012 1", **attrs)


def ring(parent, name, radius, thickness, height, z, material, segments=24):
    # Tangential boxes overlap at their corners, forming a closed hollow wall.
    for i in range(segments):
        a = 2*math.pi*i/segments
        surface(parent, f"{name}_{i}", "box", (thickness/2, (radius+thickness/2)*math.tan(math.pi/segments), height/2),
                (radius*math.cos(a), radius*math.sin(a), z), euler=vec((0, 0, a)), material=material)


def make_object(world, object_id, position, yaw=0., mass_scale=1., friction_scale=1.):
    spec = OBJECTS[object_id]
    body = ET.SubElement(world, "body", name=object_id, pos=vec(position), euler=vec((0, 0, yaw)))
    ET.SubElement(body, "freejoint", name=f"{object_id}_free")
    x, y, z = spec["size_m"]
    mass = spec["mass_kg"]*mass_scale
    # Conservative box inertia, with COM midway up each object. A mug handle
    # makes the true inertia asymmetric; this first asset set approximates it.
    inertia = mass/12*np.array([y*y+z*z, x*x+z*z, x*x+y*y])
    ET.SubElement(body, "inertial", pos=vec((0, 0, z/2)), mass=str(mass), diaginertia=vec(inertia))
    kind = spec["kind"]
    if kind == "plate":
        radius = x/2
        surface(body, f"{object_id}_base", "cylinder", (radius-.008, .003), (0, 0, .003), material="ceramic")
        ring(body, f"{object_id}_rim", radius-.005, .010, z-.004, (z+.004)/2,
             "plate_blue" if object_id == "plate" else "plate_gold")
        # The raised rim is supported by the base with ordinary contact geometry.
    elif kind in ("mug", "glass"):
        radius = .025 if kind == "mug" else .024
        mat = "mug_teal" if kind == "mug" else "clear_glass"
        surface(body, f"{object_id}_base", "cylinder", (radius, .007), (0, 0, .007), material=mat)
        ring(body, f"{object_id}_wall", radius-.002, .004, z-.006, (z+.006)/2, mat)
        ring(body, f"{object_id}_lip", radius-.002, .0045, .003, z-.0015,
             "mug_teal" if kind == "mug" else "glass_rim")
        if kind == "mug":
            for h in (.019, .055):
                surface(body, f"mug_handle_{h}", "capsule", (.004, .012), (.035, 0, h),
                        quat=vec((math.sqrt(.5), 0, math.sqrt(.5), 0)), material=mat)
            surface(body, "mug_handle_grip", "capsule", (.004, .018), (.047, 0, .037), material=mat)
    elif kind == "bottle":
        surface(body, "bottle_base", "cylinder", (.024, .007), (0, 0, .007), material="amber_glass")
        ring(body, "bottle_wall", .022, .004, .079, .0455, "amber_glass")
        # A stepped shoulder and hollow narrow neck, including a real outlet.
        ring(body, "bottle_shoulder_lower", .019, .008, .010, .089, "amber_glass")
        ring(body, "bottle_shoulder_upper", .014, .006, .010, .097, "amber_glass")
        ring(body, "bottle_neck", .0095, .003, .040, .120, "amber_glass", segments=16)
        ring(body, "bottle_label", .0242, .0006, .027, .047, "bottle_label")
        # Label does not alter collisions.
        for g in body.findall("geom"):
            if g.get("name", "").startswith("bottle_label"):
                g.set("contype", "0"); g.set("conaffinity", "0")
    else:
        surface(body, f"{object_id}_handle", "box", (.0045, .034, .008), (0, -.019, .008), material="steel")
        surface(body, f"{object_id}_neck", "box", (.003, .010, .004), (0, .020, .004), material="steel")
        if kind == "fork":
            surface(body, "fork_head", "box", (.0085, .007, .004), (0, .032, .004), material="steel")
            for i in range(4):
                surface(body, f"fork_tine_{i}", "box", (.00125, .011, .0015),
                        ((i-1.5)*.0048, .046, .004), material="steel")
        else:
            surface(body, "spoon_bowl", "ellipsoid", (.011, .019, .003), (0, .038, .004), material="steel")
    for g in body.findall("geom"):
        g.set("friction", vec((.8*friction_scale, .005, .0001)))
    ET.SubElement(body, "site", name=f"{object_id}_grasp", pos=vec(spec["grasp_local_m"]), size=".002", group="4", rgba="0 1 0 1")
    return {"id": object_id, "body": object_id, **spec, "mass_kg": mass,
            "friction": .8*friction_scale, "initial_position_m": list(position),
            "initial_yaw_rad": yaw, "grasp_site": f"{object_id}_grasp",
            "grasp_status": "candidate_only_not_a_verified_grasp"}


def add_drawer(world, opened):
    """A tabletop cabinet with one unactuated slider; no hidden drawer motor."""
    base = (-.280, .180, TABLE_Z)
    cabinet = ET.SubElement(world, "body", name="cutlery_cabinet", pos=vec(base))
    for sign in (-1, 1):
        surface(cabinet, f"cabinet_side_{sign}", "box", (.006, .083, .045), (sign*.104, 0, .045), material="drawer_wood")
    surface(cabinet, "cabinet_roof", "box", (.110, .083, .006), (0, 0, .096), material="drawer_wood")
    surface(cabinet, "cabinet_back", "box", (.098, .005, .045), (0, .078, .045), material="drawer_wood")
    tray = ET.SubElement(world, "body", name="cutlery_drawer", pos=vec(base))
    ET.SubElement(tray, "joint", name="drawer_slide", type="slide", axis="0 -1 0", range=vec((0, DRAWER_TRAVEL)), damping="2", frictionloss=".15", armature=".01")
    ET.SubElement(tray, "inertial", pos="0 0 .025", mass=".18", diaginertia=".00045 .0006 .0009")
    surface(tray, "drawer_floor", "box", (.094, .072, .004), (0, 0, .015), material="drawer_lining")
    for sign in (-1, 1):
        surface(tray, f"drawer_side_{sign}", "box", (.003, .072, .015), (sign*.094, 0, .034), material="drawer_wood")
    surface(tray, "drawer_back", "box", (.094, .003, .015), (0, .069, .034), material="drawer_wood")
    surface(tray, "drawer_front", "box", (.108, .005, .020), (0, -.089, .030), material="drawer_wood")
    for x in (-.036, .036):
        surface(tray, f"drawer_handle_post_{x}", "box", (.004, .034, .004), (x, -.124, .052), material="steel")
    surface(tray, "drawer_handle", "capsule", (.005, .036), (0, -.160, .052), material="steel", quat=vec((math.sqrt(.5), 0, math.sqrt(.5), 0)))
    # Supports expose the utensil handles; these are fixed to the moving tray.
    for y in (-.047, .006):
        surface(tray, f"cutlery_support_{y}", "box", (.078, .003, .005), (0, y, .024), material="drawer_lining")
    ET.SubElement(tray, "site", name="drawer_handle_grasp", pos="0 -.160 .052", size=".002", group="4")
    return {"body": "cutlery_drawer", "joint": "drawer_slide", "travel_m": DRAWER_TRAVEL,
            "initial_open_m": DRAWER_TRAVEL if opened else 0., "handle_site": "drawer_handle_grasp",
            "actuated": False, "closed_origin_m": list(base)}


def add_dinner_scene(root, seed, preset="task", drawer_open=False):
    if preset not in ("task", "reference"):
        raise ValueError("Choose the task start or reference dinner layout.")
    if preset == "reference" and drawer_open:
        raise ValueError("The target example keeps the drawer closed to clear the glass setting. Choose Task start to inspect the open drawer.")
    world, asset = root.find("worldbody"), root.find("asset")
    root.set("model", "talos_dual_so101_dinner")
    rng = np.random.default_rng(seed ^ 0xD177E2)
    materials = {
        "ceramic": (".95 .95 .90 1", ".3"), "plate_blue": (".33 .51 .76 1", ".35"),
        "plate_gold": (".81 .56 .28 1", ".35"), "mug_teal": (".06 .53 .49 1", ".4"),
        "clear_glass": (".66 .87 .92 .30", ".75"), "glass_rim": (".62 .84 .89 .8", ".6"),
        "amber_glass": (".55 .32 .13 .78", ".6"), "bottle_label": (".96 .91 .72 1", ".1"),
        "steel": (".72 .78 .83 1", ".85"), "drawer_wood": (".37 .21 .12 1", ".2"),
        "drawer_lining": (".20 .28 .27 1", ".1"),
    }
    for name, (rgba, specular) in materials.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular=specular, shininess=".65")
    asset.find("material[@name='table_mat']").set("rgba", ".69 .53 .37 1")
    for grid in list(world.findall("geom")):
        if grid.get("name", "").startswith("grid_"):
            world.remove(grid)
    # Visual placemats and target outlines have no contact or mass.
    geom(world, "linen_runner", "box", (.455, .145, .00015), (0, -.005, TABLE_Z+.0002), rgba=".84 .83 .73 1", contype="0", conaffinity="0")
    targets = [
        {"id": "plate_place", "label": "Plate setting", "object_id": "plate", "position_m": [-.060, -.025, TABLE_Z], "radius_m": .074},
        {"id": "side_plate_place", "label": "Side plate setting", "object_id": "side_plate", "position_m": [.110, -.025, TABLE_Z], "radius_m": .060},
        {"id": "mug_place", "label": "Mug setting", "object_id": "mug", "position_m": [.265, .005, TABLE_Z], "radius_m": .033},
        {"id": "glass_place", "label": "Glass setting", "object_id": "glass", "position_m": [.075, .100, TABLE_Z], "radius_m": .033},
        {"id": "fork_place", "label": "Fork setting", "object_id": "fork", "position_m": [-.160, -.045, TABLE_Z], "radius_m": .012},
        {"id": "spoon_place", "label": "Spoon setting", "object_id": "spoon", "position_m": [-.020, -.155, TABLE_Z], "radius_m": .014},
    ]
    for target in targets:
        x, y, z = target["position_m"]
        if target["object_id"] in ("spoon", "fork"):
            geom(world, target["id"], "box", ((.061, .014, .00015) if target["object_id"] == "spoon" else (.014, .061, .00015)), (x, y, z+.0006), rgba=".50 .51 .43 .55", contype="0", conaffinity="0")
        else:
            marker = ET.SubElement(world, "body", name=target["id"], pos=vec((x, y, z+.0006)))
            ring(marker, target["id"]+"_outline", target["radius_m"], .0015, .0003, 0, "drawer_lining", segments=40)
            for g in marker.findall("geom"):
                g.set("contype", "0"); g.set("conaffinity", "0")
    drawer = add_drawer(world, drawer_open)
    positions = {"plate": (-.110, .015), "side_plate": (.120, .252),
                 "mug": (.250, -.065), "glass": (.285, .245), "bottle": (.025, -.075)}
    records = []
    for object_id, spec in OBJECTS.items():
        target = next((t for t in targets if t["object_id"] == object_id), None)
        if preset == "reference" and target:
            position = np.array(target["position_m"], dtype=float)
            position[2] += .001
            yaw = -math.pi/2 if object_id == "spoon" else 0.
        elif preset == "reference" and object_id == "bottle":
            position = np.array([.095,-.115,TABLE_Z+.001])
            yaw = 0.
        elif object_id in ("fork", "spoon"):
            position = np.array([-.280 + (-.023 if object_id == "fork" else .023), .165-drawer["initial_open_m"], TABLE_Z+.030])
            position[:2] += rng.uniform(-.003, .003, 2)
            yaw = float(rng.uniform(-.07, .07))
        else:
            position = np.array([*positions[object_id], TABLE_Z+.001])
            position[:2] += rng.uniform(-.009, .009, 2)
            if drawer_open and object_id == "plate":
                position[0] += .020
            yaw = float(rng.uniform(-.12, .12))
        records.append(make_object(world, object_id, position, yaw, float(rng.uniform(.92, 1.08)), float(rng.uniform(.9, 1.1))))
    # Mild reproducible lighting perturbation; geometry/dimensions stay fixed.
    brightness = float(rng.uniform(.92, 1.06))
    world.find("light").set("diffuse", vec(np.array([.75, .77, .8])*brightness))
    return {"objects": records, "targets": targets, "drawer": drawer,
            "dinner_preset": preset, "drawer_open": drawer_open,
            "randomization": {"object_xy_m": .009, "mass_fraction": .08, "friction_fraction": .10, "light_multiplier": brightness},
            "challenge": {"title": "Set the dinner table", "autonomy_available": True,
                          "instruction": "Open the drawer, place the plate and cutlery, then coordinate both arms to serve a drink.",
                          "next_skill": "Learn camera/language control from verified demonstrations."}}


def dinner_state(model, data, layout):
    """Ground truth for inspection, never a learned policy observation contract."""
    if layout.get("scenario") != "dinner":
        return {}
    objects = []
    for item in layout["objects"]:
        body = data.body(item["body"])
        objects.append({**item, "position_m": body.xpos.tolist(),
                        "upright": bool(body.xmat[8] > math.cos(math.radians(15))),
                        "above_table": bool(body.xpos[2] >= TABLE_Z-.003),
                        "grasp_position_m": data.site(item["grasp_site"]).xpos.tolist()})
    joint = model.joint(layout["drawer"]["joint"])
    opening = float(data.qpos[joint.qposadr[0]])
    return {"objects": objects, "object_count": len(objects),
            "stable_object_count": sum(o["upright"] and o["above_table"] for o in objects),
            "targets": layout["targets"], "challenge": layout["challenge"],
            "dinner_preset": layout["dinner_preset"], "drawer_open": layout["drawer_open"],
            "bottle_start":layout.get('bottle_start','upright'),
            "drawer": {**layout["drawer"], "open_m": opening, "open_fraction": opening/DRAWER_TRAVEL,
                       "handle_position_m": data.site("drawer_handle_grasp").xpos.tolist()}}
