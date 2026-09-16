"""Create seeded, reachable cabinet starts independently of final table poses."""
import math
import numpy as np

# Rough bounding-circle radius per movable item (m), used only to keep the
# free-placement sampler below from overlapping objects. Not the true
# collision geometry - that's handled by physics after placement.
_FOOTPRINT_RADIUS_M = {'bottle': .03, 'plate': .07, 'mug': .04, 'fork': .055, 'spoon': .055}
_FREE_ZONE_X = (-.27, .27)
_FREE_ZONE_Y = (-.16, .16)
_FREE_YAW_RANGE_RAD = math.radians(60)


def cabinet_source_poses(final_layout, variation, table_z, profile="duet_v1"):
    """Choose source poses with open gripper clearance; retain fixed support props."""
    if profile == "duet_free_v1":
        return _free_cabinet_source_poses(final_layout, variation.seed, table_z)
    positions = {'bottle': (.025, -.075), 'plate': (-.110, .015),
                 'mug': (.250, -.065), 'fork': (.180, -.060),
                 'spoon': (-.257, .045)}
    poses = {}
    for item, final in final_layout['objects'].items():
        if item not in positions:
            poses[item] = final
            continue
        xy = np.array(positions[item]) + variation.object_offsets_m[item]
        poses[item] = {'position_m': [*xy.tolist(), table_z + (.030 if item in ('fork', 'spoon') else .001)],
                       'yaw_rad': variation.object_yaw_rad[item]}
    return poses


def _free_cabinet_source_poses(final_layout, seed, table_z):
    """Drop every pick/place item at a random, collision-free spot anywhere in
    the shared reachable zone - not a fixed per-item anchor with small jitter.
    Lets an item that always spawned on one side land on the other seed to
    seed. Rejection sampling, not a real packing solver: fine for 5 small
    items in a wide zone; raise the attempt budget if items start blocking
    each other more (more items, smaller zone)."""
    rng = np.random.default_rng(seed ^ 0xC001)
    order = list(_FOOTPRINT_RADIUS_M)
    rng.shuffle(order)
    placed = {}
    for item in order:
        radius = _FOOTPRINT_RADIUS_M[item]
        for _ in range(2000):
            x = rng.uniform(_FREE_ZONE_X[0] + radius, _FREE_ZONE_X[1] - radius)
            y = rng.uniform(_FREE_ZONE_Y[0] + radius, _FREE_ZONE_Y[1] - radius)
            if all(math.hypot(x - px, y - py) >= radius + _FOOTPRINT_RADIUS_M[other] + .015
                   for other, (px, py) in placed.items()):
                placed[item] = (x, y)
                break
        else:
            raise ValueError(f"Could not place '{item}' without overlap after 2000 tries.")
    poses = {}
    for item, final in final_layout['objects'].items():
        if item not in placed:
            poses[item] = final
            continue
        x, y = placed[item]
        yaw = float(rng.uniform(-_FREE_YAW_RANGE_RAD, _FREE_YAW_RANGE_RAD))
        poses[item] = {'position_m': [x, y, table_z + (.030 if item in ('fork', 'spoon') else .001)],
                       'yaw_rad': yaw}
    return poses
