"""Experimental RGB silhouette geometry and learned held-mug midpoint estimator.

Classical color/component preprocessing is explicit. The small neural network
predicts position from those image features; no privileged state enters inference.
This module is not integrated into the selected dinner controller.
"""
import itertools
import numpy as np
from PIL import Image
import torch
from torch import nn
from .dinner_vision import components
from .rgb_servo_cameras import project

VIEW_DIMENSIONS = 264
FEATURE_DIMENSIONS = 3*VIEW_DIMENSIONS


def encode_view(image, projection, method):
    if image.shape != (240, 320, 3) or not np.isfinite(image).all():
        raise ValueError('Expected a finite calibrated 320x240 RGB image.')
    low, high = np.asarray(method['roi_world_box_m'])
    pixels = project(np.asarray(list(itertools.product(*zip(low, high)))), projection)
    margin = method['roi_pixel_margin']
    left, top = np.maximum(np.floor(pixels.min(0))-margin, 0).astype(int)
    right, bottom = np.minimum(np.ceil(pixels.max(0))+margin+1, [320, 240]).astype(int)
    roi = np.zeros((240, 320), bool)
    roi[top:bottom, left:right] = True
    r, g, b = image.astype(float).transpose(2, 0, 1)
    candidates = components(roi & (g > 1.5*r) & (g > 1.015*b) & (b > 1.3*r) & (g > 45))
    candidates.sort(key=lambda p: (-len(p), int(p[:, 0].min()), int(p[:, 1].min())))
    result = np.zeros(VIEW_DIMENSIONS, np.float32)
    count = len(candidates[0]) if candidates else 0
    if count < 60:
        return result, {'usable': False, 'rgb_pixels': count}
    ys, xs = candidates[0].T
    x0, x1, y0, y1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    mask[ys-y0, xs-x0] = 255
    small = np.asarray(Image.fromarray(mask).resize((16, 16), Image.Resampling.BOX), dtype=np.float32)/255
    result[:8] = [1, count/4096, xs.mean()/320, ys.mean()/240, x0/320, y0/240, x1/320, y1/240]
    result[8:] = small.flatten()
    return result, {'usable': True, 'rgb_pixels': count}


def encode_images(images, calibrations, method):
    if len(images) != 3 or len(calibrations) != 3:
        raise ValueError('Exactly three declared fixed views are required.')
    results = [encode_view(image, camera['projection'], method) for image, camera in zip(images, calibrations)]
    return np.concatenate([r[0] for r in results]), [r[1] for r in results]


class MugShapePoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIMENSIONS, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, features):
        return self.net(features)
