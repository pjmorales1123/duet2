"""CPU execution of the separate experimental coarse-to-fine bottle observer."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import openvino as ov
import torch
from .rgb_servo_cameras import VIEWS
from .rgb_servo_network import decode_heatmaps
from .bottle_refinement import image_crops, decode, reconstruct


class RefinedBottleObserver:
    def __init__(self, folder):
        folder = Path(folder)
        self.settings = json.loads((folder/'protocol.json').read_text(encoding='utf-8-sig'))['inference']
        artifacts = json.loads((folder/'artifacts.json').read_text(encoding='utf-8-sig'))
        for name, digest in artifacts['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise ValueError('A refined observer artifact changed: '+name)
        root = Path(__file__).resolve().parents[1]
        for name, digest in artifacts['runtime_sources'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError('A refined observer source changed: '+name)
        options = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2}
        core = ov.Core()
        self.coarse = core.compile_model(str(folder/'coarse.xml'), 'CPU', options)
        self.refiner = core.compile_model(str(folder/'refiner.xml'), 'CPU', options)

    def observe(self, images, calibrations):
        if set(images) != set(VIEWS) or set(calibrations) != set(VIEWS):
            raise ValueError('Three declared RGB views and calibrations are required.')
        array = np.stack([images[name] for name in VIEWS])
        if array.shape != (3, 240, 320, 3) or array.dtype != np.uint8:
            raise ValueError('Expected three uint8 RGB images.')
        started = time.perf_counter()
        rgb = array.transpose(0, 3, 1, 2).astype(np.float32)/255
        with torch.inference_mode():
            coarse = self.coarse([rgb])
            points = decode_heatmaps(torch.from_numpy(coarse[0].copy()))[0]
            crops = image_crops(torch.from_numpy(rgb), points).half().float()
            refined = self.refiner([crops.numpy()])
            decoded = decode(tuple(torch.from_numpy(refined[i].copy()) for i in range(2)))
            local, peaks, variance, visibility = [v.numpy() for v in decoded]
            pixels = local+points.numpy().reshape(-1, 2)-31.5
        observation = reconstruct(pixels.reshape(3, 2, 2), peaks.reshape(3, 2), variance.reshape(3, 2),
            visibility.reshape(3, 2), [calibrations[name] for name in VIEWS], self.settings)
        observation.update(source='Current RGB, frozen coarse CNN, learned local image refinement and fixed camera geometry',
            inference_runtime='OpenVINO CPU FP32', observer_wall_ms=(time.perf_counter()-started)*1000,
            required_views=2, used_views=[list(VIEWS)[i] for i in observation.get('accepted_views', [])])
        return observation
