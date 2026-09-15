"""Learned local RGB refinement after a frozen coarse bottle point detector.

Inference accepts images, predicted image points and fixed camera calibration.
No simulator object state, depth, segmentation IDs or teacher actions are inputs.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .rgb_servo_network import block
from .rgb_servo_cameras import triangulate, project
from .rgb_servo_geometry import rigid_bottle_keypoints


class KeypointCropRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(block(4, 16), block(16, 32, 2), block(32, 48))
        self.heatmap = nn.Conv2d(48, 1, 1)
        self.visibility = nn.Linear(96, 1)

    def forward(self, crop):
        feature = self.encoder(crop)
        pooled = torch.cat([feature.mean((2, 3)), feature.amax((2, 3))], 1)
        return self.heatmap(feature)[:, 0], self.visibility(pooled)[:, 0]


def image_crops(images, points, size=64):
    """Bilinear RGB crops with explicit pixel-center coordinates and point role."""
    if images.ndim != 4 or images.shape[1] != 3 or points.shape != (len(images), 2, 2):
        raise ValueError('Expected N RGB images and two predicted image points each.')
    n, _, height, width = images.shape
    offsets = torch.arange(size, device=images.device, dtype=images.dtype)-(size-1)/2
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    local = torch.stack([xx, yy], -1)
    pixel = points.reshape(-1, 1, 1, 2)+local
    norm = torch.tensor([width, height], dtype=images.dtype, device=images.device)
    grid = 2*(pixel+.5)/norm-1
    crops = F.grid_sample(images.repeat_interleave(2, 0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    roles = torch.arange(2, device=images.device, dtype=images.dtype).repeat(n).reshape(-1, 1, 1, 1)
    return torch.cat([crops*2-1, (roles*2-1).expand(-1, 1, size, size)], 1)


def decode(outputs):
    logits, visibility = outputs
    height, width = logits.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    grid = torch.stack([xx*2+.5, yy*2+.5], -1).to(logits.dtype)
    probabilities = logits.flatten(1).softmax(-1).reshape_as(logits)
    point = (probabilities[..., None]*grid).sum((1, 2))
    variance = (probabilities*((grid[None]-point[:, None, None])**2).sum(-1)).sum((1, 2))
    return point, probabilities.flatten(1).amax(1), variance, visibility.sigmoid()


def refinement_loss(outputs, points, visible):
    logits, visibility = outputs
    height, width = logits.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    grid = torch.stack([xx*2+.5, yy*2+.5], -1).to(logits.dtype)
    squared = ((grid[None]-points[:, None, None])**2).sum(-1)
    target = torch.exp(-squared/(2*.9**2))
    target = target/(target.sum((1, 2), keepdim=True)+1e-9)
    cross_entropy = -(target*logits.flatten(1).log_softmax(-1).reshape_as(logits)).sum((1, 2))
    coordinates, _, _, _ = decode(outputs)
    coordinate = F.smooth_l1_loss(coordinates, points, reduction='none').mean(1)
    visible = visible.float()
    denominator = visible.sum().clamp_min(1)
    return ((cross_entropy+.25*coordinate)*visible).sum()/denominator+.4*F.binary_cross_entropy_with_logits(visibility, visible)


def reconstruct(points, peaks, variances, visibility, calibrations, settings):
    """Apply declared image-confidence and the existing 103 mm rigid geometry."""
    points, peaks, variances, visibility = [np.asarray(v) for v in (points, peaks, variances, visibility)]
    if points.shape != (3, 2, 2) or any(v.shape != (3, 2) for v in (peaks, variances, visibility)):
        raise ValueError('Three views with two semantic points are required.')
    if not all(np.isfinite(v).all() for v in (points, peaks, variances, visibility)):
        return {'status': 'refused', 'reason': 'nonfinite_image_prediction'}
    valid = ((visibility >= settings['minimum_visibility_probability']).all(1)
        & (peaks >= settings['minimum_heatmap_peak']).all(1)
        & (variances <= settings['maximum_heatmap_variance_px2']).all(1))
    selected = np.flatnonzero(valid)
    result = {'status': 'refused', 'reason': 'image_confidence', 'accepted_views': selected.tolist(),
        'keypoints_px': points.tolist(), 'peaks': peaks.tolist(), 'variance_px2': variances.tolist(),
        'visibility': visibility.tolist()}
    if len(selected) < settings['minimum_views']:
        return result
    matrices = [calibrations[i]['projection'] for i in selected]
    try:
        world = np.asarray([triangulate(points[selected, k], matrices) for k in range(2)])
        axis = world[1]-world[0]
        residual = max(float(np.linalg.norm(project(world, matrix)-points[i], axis=1).max()) for i, matrix in zip(selected, matrices))
        if not .097 <= np.linalg.norm(axis) <= .109 or axis[2] <= .075 or residual > 1.5:
            result['reason'] = 'inconsistent_geometry'
            return result
        world, fit = rigid_bottle_keypoints(world, points[selected], matrices)
        residual = max(float(np.linalg.norm(project(world, matrix)-points[i], axis=1).max()) for i, matrix in zip(selected, matrices))
        axis = world[1]-world[0]
        if residual > 1.5 or axis[2] <= .075:
            result['reason'] = 'rigid_geometry'
            return result
    except (ValueError, np.linalg.LinAlgError):
        result['reason'] = 'degenerate_geometry'
        return result
    low, high = np.asarray(settings['grasp_workspace_m'])
    if np.any(world[1] < low) or np.any(world[1] > high):
        result['reason'] = 'outside_declared_workspace'
        return result
    return {**result, 'status': 'observed', 'reason': None, 'keypoints_m': world.tolist(),
        'grasp_point_m': world[1].tolist(), 'axis': (axis/np.linalg.norm(axis)).tolist(),
        'reprojection_error_px': residual, 'rigid_geometry': fit, 'yaw_estimated': False}
