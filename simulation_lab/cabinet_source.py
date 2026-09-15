"""Create seeded, reachable cabinet starts independently of final table poses."""
import numpy as np


def cabinet_source_poses(final_layout, variation, table_z):
    """Choose source poses with open gripper clearance; retain fixed support props."""
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
