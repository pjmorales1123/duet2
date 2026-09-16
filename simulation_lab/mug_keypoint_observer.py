"""Experimental mug image-point inference with fixed calibrated geometry.

The network sees RGB only. The decoder uses known cameras/object dimensions;
ground-truth object pose, segmentation and robot state are absent from its API.
"""
import numpy as np
import torch
from .rgb_servo_cameras import project, triangulate
from .rgb_servo_network import BottleKeypointNet, decode_heatmaps


class MugImagePointNet(BottleKeypointNet):
    """Reuse the architecture, with separately initialized and trained weights."""


def decoded_outputs(outputs):
    heatmaps, masks, visibility = outputs
    pixels, peaks, variances = decode_heatmaps(heatmaps)
    return {'pixels': pixels, 'peaks': peaks, 'variances': variances,
            'visibility': visibility.sigmoid(), 'foreground_pixels': (masks.sigmoid()[:, 0] >= .5).sum((1, 2))*4}


def reconstruct(decoded, calibrations, settings):
    """Decode one pair, without accepting any scoring/privileged argument."""
    if len(calibrations) != 2 or np.asarray(decoded['pixels']).shape != (2, 2, 2):
        raise ValueError('The mug observer requires exactly two fixed calibrated views.')
    if any(not np.isfinite(np.asarray(value)).all() for value in decoded.values()):
        return {'status': 'refused', 'reason': 'nonfinite_prediction'}
    accepted = ((np.asarray(decoded['visibility']) >= settings['minimum_visibility_probability'])
        & (np.asarray(decoded['peaks']).min(1) >= settings['minimum_heatmap_peak'])
        & (np.asarray(decoded['variances']).max(1) <= settings['maximum_heatmap_variance_px2'])
        & (np.asarray(decoded['foreground_pixels']) >= settings['minimum_predicted_foreground_pixels']))
    result = {'status': 'refused', 'reason': 'view_confidence', 'accepted_views': np.flatnonzero(accepted).tolist(),
              'views': {key: np.asarray(value).tolist() for key, value in decoded.items()}}
    if not accepted.all():
        return result
    matrices = [r['projection'] for r in calibrations]
    pixels = np.asarray(decoded['pixels'])
    try:
        points = np.asarray([triangulate(pixels[:, key], matrices) for key in range(2)])
    except ValueError:
        result['reason'] = 'degenerate_geometry'
        return result
    residual = max(float(np.linalg.norm(project(points, matrix)-pixels[slot], axis=1).max()) for slot, matrix in enumerate(matrices))
    axis = points[1]-points[0]
    length = float(np.linalg.norm(axis))
    midpoint = points.mean(0)
    low, high = np.asarray(settings['midpoint_workspace_box_m'])
    valid = bool(np.isfinite(points).all() and residual <= settings['maximum_reprojection_error_px']
        and abs(length-settings['axis_length_m']) <= settings['maximum_axis_length_error_m']
        and axis[2] >= settings['minimum_upward_axis_z_m'] and np.all(midpoint >= low) and np.all(midpoint <= high))
    result.update(status='observed' if valid else 'refused', reason=None if valid else 'calibrated_geometry',
                  points_m=points.tolist(), midpoint_m=midpoint.tolist(), axis_length_m=length,
                  reprojection_error_px=residual)
    return result


def predict_batch(network, images, device):
    network.eval()
    with torch.inference_mode():
        values = decoded_outputs(network(images.to(device).float()/255))
    return {key: value.detach().cpu().numpy() for key, value in values.items()}
