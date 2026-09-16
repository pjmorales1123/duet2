"""Duet 2 dinner objects with seeded cabinet starts and declared final poses.

Object bodies use two geom layers:
  Visual geom  — type="mesh", contype="0", conaffinity="0", mass="0".
                 References OBJ file in assets/dinner/meshes/.  Rendered.
  Collision proxy — simple primitive (cylinder/capsule/box), normal contact,
                 geom group 3 (invisible to presentation cameras).

Masses and collision proxies retain the procedural physical specification.
Source poses, grasp candidates and final targets implement cabinet service.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scene import TABLE_Z, geom, vec
from .dinner_layout import canonical_dinner_layout, load_dinner_layout
from .cabinet_source import cabinet_source_poses
from .duet_variation import sample_dinner_variation

OBJECTS = {
    "plate": {"label": "Coral dinner plate", "kind": "plate", "mass_kg": .065, "size_m": [.132, .132, .024], "color": "#cf5047", "grasp_local_m": [0, -.061, .010], "grasp_width_m": .009},
    "side_plate": {"label": "Sunset side plate", "kind": "plate", "mass_kg": .045, "size_m": [.106, .106, .012], "color": "#f2b74d", "grasp_local_m": [0, -.048, .009], "grasp_width_m": .009},
    "mug": {"label": "Cobalt cup", "kind": "mug", "mass_kg": .045, "size_m": [.077, .050, .064], "color": "#5b7dd1", "grasp_local_m": [.047, 0, .041], "grasp_width_m": .008},
    "glass": {"label": "Moonlit tumbler", "kind": "glass", "mass_kg": .035, "size_m": [.048, .048, .074], "color": "#c4d3ff", "grasp_local_m": [0, 0, .045], "grasp_width_m": .048},
    "bottle": {"label": "Plum carafe", "kind": "bottle", "mass_kg": .080, "size_m": [.048, .048, .140], "color": "#5b294c", "grasp_local_m": [0, 0, .128], "grasp_width_m": .022},
    "fork": {"label": "Service fork", "kind": "fork", "mass_kg": .012, "size_m": [.017, .110, .016], "color": "#ff2fb0", "grasp_local_m": [0, 0, .008], "grasp_candidates_local_m": [[0, -.030, .008], [0, -.019, .008], [0, 0, .008]], "grasp_width_m": .009},
    "spoon": {"label": "Service spoon", "kind": "spoon", "mass_kg": .014, "size_m": [.022, .110, .016], "color": "#4caf50", "grasp_local_m": [0, 0, .008], "grasp_candidates_local_m": [[0, -.030, .008], [0, -.019, .008], [0, 0, .008]], "grasp_width_m": .009},
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


def collision_ring(parent, name, radius, thickness, height, z, segments=24):
    """Build a hidden, contact-enabled hollow vessel wall from tangent boxes."""
    for index in range(segments):
        angle = 2 * math.pi * index / segments
        _proxy(
            parent, f"{name}_{index}", "box",
            (thickness / 2, (radius + thickness / 2) * math.tan(math.pi / segments), height / 2),
            (radius * math.cos(angle), radius * math.sin(angle), z),
            euler=(0, 0, angle),
        )


def _mesh_geom(parent, name: str, mesh_name: str, rgba: str, euler=(0,0,0), group=2):
    """Visual-only mesh geom: no collision, no mass."""
    attrs = dict(name=name, type="mesh", mesh=mesh_name,
                 contype="0", conaffinity="0", mass="0",
                 rgba=rgba, group=str(group))
    if any(euler):
        attrs["euler"] = vec(euler)
    ET.SubElement(parent, "geom", **attrs)


def _proxy(parent, name: str, kind: str, size, pos=(0,0,0), euler=(0,0,0), quat=None):
    """Collision-only primitive: contact-enabled, hidden from presentation.

    group="3" keeps these invisible in the overhead/wrist cameras used for
    training images; they still participate in physics.
    """
    attrs = dict(name=name, type=kind, size=vec(size) if not isinstance(size, str) else size,
                 pos=vec(pos), mass="0",
                 friction=".8 .005 .0001", condim="4", solref=".012 1",
                 contype="1", conaffinity="1", group="3")
    if any(euler):
        attrs["euler"] = vec(euler)
    if quat is not None:
        attrs["quat"] = vec(quat)
    ET.SubElement(parent, "geom", **attrs)


def make_object(world, object_id, position, yaw=0., mass_scale=1., friction_scale=1.):
    spec = OBJECTS[object_id]
    body = ET.SubElement(world, "body", name=object_id,
                         pos=vec(position), euler=vec((0, 0, yaw)))
    ET.SubElement(body, "freejoint", name=f"{object_id}_free")
    x, y, z = spec["size_m"]
    mass = spec["mass_kg"] * mass_scale
    # Box inertia approximation with COM at half-height.  A mug handle makes
    # the true inertia asymmetric; this conservative estimate is unchanged
    # from the original procedural spec.
    inertia = mass / 12 * np.array([y*y+z*z, x*x+z*z, x*x+y*y])
    ET.SubElement(body, "inertial", pos=vec((0, 0, z/2)),
                  mass=str(mass), diaginertia=vec(inertia))

    kind = spec["kind"]
    # ── hex colour to MuJoCo RGBA string (alpha=1) ────────────────────────
    hx = spec["color"].lstrip("#")
    r8, g8, b8 = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
    rgba = f"{r8/255:.3f} {g8/255:.3f} {b8/255:.3f} 1"
    mesh_id = object_id if object_id != "bottle" else "carafe"
    visible = object_id not in {"side_plate", "glass"}

    if kind == "plate":
        radius = x / 2
        # Keep support props in the physical scene, but omit them from camera
        # presentation so the task view contains one plate and one cup.
        _mesh_geom(body, f"{object_id}_vis", f"{mesh_id}_vis", rgba,
                   group=2 if visible else 3)
        # Preserve the original low base and raised rim contact envelope.
        _proxy(body, f"{object_id}_col", "cylinder",
               (radius - .008, .003), (0, 0, .003))
        collision_ring(body, f"{object_id}_rim", radius - .005, .010,
                       z - .004, (z + .004) / 2)

    elif kind == "mug":
        _mesh_geom(body, f"{object_id}_vis", f"{mesh_id}_vis", rgba)
        # Preserve the original hollow vessel contact envelope. A solid
        # cylinder would close the mouth and break both probe and grasp tests.
        _proxy(body, f"{object_id}_base", "cylinder", (.025, .007), (0, 0, .007))
        collision_ring(body, f"{object_id}_wall", .023, .004, z-.006, (z+.006)/2)
        collision_ring(body, f"{object_id}_lip", .023, .0045, .003, z-.0015)
        for height in (.019, .055):
            _proxy(body, f"{object_id}_handle_{height}", "capsule", (.004, .012),
                   (.035, 0, height), quat=(math.sqrt(.5), 0, math.sqrt(.5), 0))
        _proxy(body, f"{object_id}_handle_grip", "capsule", (.004, .018),
               (.047, 0, .037), quat=(math.sqrt(.5), 0, math.sqrt(.5), 0))

    elif kind in ("glass",):
        # Keep the validation tumbler physically present but camera-hidden.
        radius = 0.024
        mat_g = "clear_glass"
        surface(body, f"{object_id}_base", "cylinder", (radius, .007), (0,0,.007), material=mat_g)
        ring(body, f"{object_id}_wall", radius-.002, .004, z-.006, (z+.006)/2, mat_g)
        ring(body, f"{object_id}_lip", radius-.002, .0045, .003, z-.0015, "glass_rim")
        for geom_element in body.findall("geom"):
            geom_element.set("group", "3")

    elif kind == "bottle":
        _mesh_geom(body, f"{object_id}_vis", "carafe_vis", rgba)
        # A flat base and hollow neck retain the stable physical behavior of
        # the former procedural asset; only its presentation is now a mesh.
        _proxy(body, f"{object_id}_base", "cylinder", (.024, .007), (0, 0, .007))
        collision_ring(body, f"{object_id}_wall", .022, .004, .079, .0455)
        collision_ring(body, f"{object_id}_shoulder_lower", .019, .008, .010, .089)
        collision_ring(body, f"{object_id}_shoulder_upper", .014, .006, .010, .097)
        collision_ring(body, f"{object_id}_neck", .0095, .003, .040, .120, segments=16)

    elif kind == "fork":
        _mesh_geom(body, f"{object_id}_vis", "fork_vis", rgba)
        _proxy(body, f"{object_id}_handle", "box", (.0045, .034, .008), (0, -.019, .008))
        _proxy(body, f"{object_id}_neck", "box", (.003, .010, .004), (0, .020, .004))
        _proxy(body, f"{object_id}_head", "box", (.0085, .007, .004), (0, .032, .004))
        for index in range(4):
            _proxy(body, f"{object_id}_tine_{index}", "box", (.00125, .011, .0015),
                   ((index - 1.5) * .0048, .046, .004))

    elif kind == "spoon":
        _mesh_geom(body, f"{object_id}_vis", "spoon_vis", rgba)
        _proxy(body, f"{object_id}_handle", "box", (.0045, .034, .008), (0, -.019, .008))
        _proxy(body, f"{object_id}_neck", "box", (.003, .010, .004), (0, .020, .004))
        _proxy(body, f"{object_id}_bowl", "ellipsoid", (.011, .019, .003), (0, .038, .004))

    else:
        # Fallback for any unrecognised kind: procedural box stack
        surface(body, f"{object_id}_handle", "box", (.0045,.034,.008), (0,-.019,.008), material="steel")
        surface(body, f"{object_id}_neck",   "box", (.003,.010,.004),  (0, .020,.004), material="steel")

    # Apply per-object friction scaling to all contact-enabled geoms.
    if friction_scale != 1.0:
        for g_ in body.findall("geom"):
            if g_.get("contype","1") != "0":
                g_.set("friction", vec((.8*friction_scale, .005, .0001)))

    # Grasp site — unchanged from original spec.
    ET.SubElement(body, "site", name=f"{object_id}_grasp",
                  pos=vec(spec["grasp_local_m"]), size=".002",
                  group="4", rgba="0 1 0 1")
    return {"id": object_id, "body": object_id, **spec, "mass_kg": mass,
            "friction": .8*friction_scale, "initial_position_m": list(position),
            "initial_yaw_rad": yaw, "grasp_site": f"{object_id}_grasp",
            "grasp_status": "candidate_only_not_a_verified_grasp"}




def add_cabinet_zone(world, object_poses):
    """Draw a non-colliding open source cabinet while the table supports objects."""
    points = np.array([pose["position_m"][:2] for pose in object_poses.values()])
    low, high = points.min(axis=0)-.035, points.max(axis=0)+.035
    center = (low+high)/2
    size = (high-low)/2
    zone = ET.SubElement(world, "body", name="cabinet_source_zone", pos=vec((center[0], center[1], TABLE_Z+.001)))
    for name, width, position in (("back", (size[0], .004, .002), (0, size[1], .002)),
                                  ("left", (.004, size[1], .002), (-size[0], 0, .002)),
                                  ("right", (.004, size[1], .002), (size[0], 0, .002))):
        geom(zone, f"cabinet_zone_{name}", "box", width, position, rgba=".17 .10 .21 .65", contype="0", conaffinity="0", group="2")
    return {"kind": "cabinet_zone", "bounds_m": [*low.tolist(), *high.tolist()], "physical_support": "table"}


def add_dinner_scene(root, seed, preset="task", drawer_open=False, variation_profile="duet_v1", dinner_layout=None):
    if preset not in ("task", "reference"):
        raise ValueError("Choose the task start or reference dinner layout.")
    world, asset = root.find("worldbody"), root.find("asset")
    root.set("model", "duet_2_dual_so101_dinner")
    variation = sample_dinner_variation(seed, tuple(OBJECTS), variation_profile)
    final_layout = canonical_dinner_layout()
    exact_layout = load_dinner_layout(dinner_layout, tuple(OBJECTS)) if dinner_layout is not None else None
    sources = cabinet_source_poses(final_layout, variation, TABLE_Z, profile=variation_profile)
    if exact_layout is not None:
        sources.update(exact_layout['objects'])
    materials = {
        "ceramic": (".98 .94 .87 1", ".24"), "plate_blue": (".81 .31 .28 1", ".28"),
        "plate_gold": (".95 .72 .30 1", ".30"), "mug_teal": (".16 .34 .69 1", ".38"),
        "clear_glass": (".74 .82 .98 .30", ".72"), "glass_rim": (".80 .86 1 .82", ".55"),
        "amber_glass": (".36 .16 .28 .82", ".56"), "bottle_label": (".97 .60 .25 1", ".1"),
        "steel": (".66 .69 .79 1", ".80"), "drawer_wood": (".17 .10 .21 1", ".18"),
        "drawer_lining": (".10 .12 .26 1", ".08"),
    }
    for name, (rgba, specular) in materials.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular=specular, shininess=".65")
    # Register original Duet 2 dinner meshes (visual geoms; collision proxies are primitives).
    # Meshes are loaded from assets/dinner/meshes/ at scene-build time.
    _mesh_dir = Path(__file__).parent / "assets" / "dinner" / "meshes"
    _mesh_registry = {
        "plate_vis":  "plate.obj",
        # Reuse the authored plate with a declared non-uniform presentation
        # scale; its independent collision proxy remains the true side-plate
        # physical envelope.
        "side_plate_vis": "plate.obj",
        "mug_vis":    "mug.obj",
        "carafe_vis": "carafe.obj",
        "fork_vis":   "fork.obj",
        "spoon_vis":  "spoon.obj",
    }
    for mesh_name, obj_file in _mesh_registry.items():
        obj_path = _mesh_dir / obj_file
        if obj_path.is_file():
            attrs = {"name": mesh_name, "file": str(obj_path)}
            if mesh_name == "side_plate_vis":
                attrs["scale"] = ".803030303 .803030303 .5"
            ET.SubElement(asset, "mesh", **attrs)
        else:
            import warnings
            warnings.warn(
                f"Dinner mesh not found: {obj_path}. "
                "Run scripts/author_dinner_meshes.py to generate assets.",
                stacklevel=3,
            )

    # Tint the table with the seed's backdrop color so scene variation is visible
    # in the overhead training/demo frame, not just off-camera in the sky texture.
    base = np.array([.78, .72, .68])
    tint = np.array(variation.backdrop_rgba[:3])
    table_rgba = np.clip(base*.55 + tint*.9, 0, 1)
    asset.find("material[@name='table_mat']").set("rgba", vec(table_rgba)+" 1")
    asset.find("texture[@name='sky']").set("rgb1", vec(variation.backdrop_rgba[:3]))
    for grid in list(world.findall("geom")):
        if grid.get("name", "").startswith("grid_"):
            world.remove(grid)
    targets = [
        {"id": "plate_place", "label": "Plate setting", "object_id": "plate", "position_m": [-.060, -.025, TABLE_Z], "radius_m": .074},
        {"id": "side_plate_place", "label": "Side plate setting", "object_id": "side_plate", "position_m": [.110, -.025, TABLE_Z], "radius_m": .060},
        {"id": "mug_place", "label": "Mug setting", "object_id": "mug", "position_m": [.265, .015, TABLE_Z], "radius_m": .033},
        {"id": "glass_place", "label": "Glass setting", "object_id": "glass", "position_m": [.075, .100, TABLE_Z], "radius_m": .033},
        {"id": "fork_place", "label": "Fork setting", "object_id": "fork", "position_m": [-.160, -.045, TABLE_Z], "radius_m": .012},
        {"id": "spoon_place", "label": "Spoon setting", "object_id": "spoon", "position_m": [-.020, -.155, TABLE_Z], "radius_m": .014},
    ]
    targets.append({'id': 'bottle_place', 'object_id': 'bottle', 'label': 'Carafe setting', 'radius_m': .026})
    for target in targets:
        pose = final_layout['objects'][target['object_id']]
        target['position_m'] = [*pose['position_m'][:2], TABLE_Z]
        target['yaw_rad'] = pose['yaw_rad']
    for target in targets:
        x, y, z = target["position_m"]
        if target["object_id"] in ("spoon", "fork"):
            geom(world, target["id"], "box", ((.061, .014, .00015) if target["object_id"] == "spoon" else (.014, .061, .00015)), (x, y, z+.0006), rgba=".50 .51 .43 .55", contype="0", conaffinity="0", group="4")
        else:
            marker = ET.SubElement(world, "body", name=target["id"], pos=vec((x, y, z+.0006)))
            ring(marker, target["id"]+"_outline", target["radius_m"], .0015, .0003, 0, "plate_gold", segments=40)
            for g in marker.findall("geom"):
                g.set("contype", "0"); g.set("conaffinity", "0"); g.set("group", "4")
    source_zone = add_cabinet_zone(world, sources)
    positions = {"plate": (-.110, .015), "side_plate": (.120, .252),
                 "mug": (.250, -.065), "glass": (.285, .245), "bottle": (.025, -.075)}
    records = []
    for object_id, spec in OBJECTS.items():
        target = next((t for t in targets if t["object_id"] == object_id), None)
        exact_pose = sources.get(object_id) if preset == 'task' else None
        if exact_pose is not None:
            position = np.array(exact_pose["position_m"], dtype=float)
            yaw = exact_pose["yaw_rad"]
        elif preset == "reference" and target:
            position = np.array(target["position_m"], dtype=float)
            position[2] += .001
            yaw = target['yaw_rad']
        elif preset == "reference" and object_id == "bottle":
            position = np.array([.095,-.115,TABLE_Z+.001])
            yaw = 0.
        else:
            position = np.array([*positions[object_id], TABLE_Z+.001])
            position[:2] += variation.object_offsets_m[object_id]
            yaw = variation.object_yaw_rad[object_id]
        records.append(make_object(world, object_id, position, yaw, variation.mass_scales[object_id], variation.friction_scales[object_id]))
        if preset == 'task' and object_id in ('fork', 'spoon'):
            # Two real ledges expose the handle for both fingers without a drawer.
            support = ET.SubElement(world, 'body', name=object_id+'_source_support', pos=vec((position[0], position[1], TABLE_Z)), euler=vec((0, 0, yaw)))
            for y in (-.047, .006):
                surface(support, object_id+f'_source_ledge_{y}', 'box', (.014, .003, .0145), (0, y, .0145), material='drawer_lining')
    world.find("light").set("diffuse", vec(np.array([.75, .77, .8])*variation.light_multiplier))
    return {"objects": records, "targets": targets, "source_zone": source_zone,
            "dinner_preset": preset,
            "randomization": variation.manifest(),
            "dinner_layout": exact_layout,
            "challenge": {"title": "Duet dinner service", "autonomy_available": True,
                          "instruction": "Pick from the cabinet source zone, then place the table setting.",
                          "next_skill": "Train from verified, split-aware demonstrations across randomized scenes."},
            "duet_scene": {"theme": "clean studio service", "asset_revision": "duet-clean-studio-v2"}}


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
    return {"objects": objects, "object_count": len(objects),
            "stable_object_count": sum(o["upright"] and o["above_table"] for o in objects),
            "targets": layout["targets"], "challenge": layout["challenge"],
            "dinner_preset": layout["dinner_preset"],
            "bottle_start":layout.get('bottle_start','upright'),
            "source_zone": layout["source_zone"]}
