"""OpenVINO execution adapters preserving the experimental policy guards.

Only the learned networks execute through OpenVINO. RGB decoding, calibration,
phase logic, proprioception and the separate physical monitor remain explicit.
"""
import numpy as np
import openvino as ov
import torch
from safetensors.torch import load_file
from .rgb_bottle_observer import RgbBottleObserver
from .rgb_servo_motor import CartesianMotorPolicy, RobotGeometry


class NetworkAdapter:
    def __init__(self, xml, device='CPU', checkpoint=None):
        self.compiled = ov.Core().compile_model(str(xml), device, {'PERFORMANCE_HINT': 'LATENCY',
                                               'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2} if device == 'CPU'
                                               else {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'})
        if checkpoint:
            values = load_file(str(checkpoint))
            self.center = values['center']; self.scale = values['scale']

    def __call__(self, *inputs):
        result = self.compiled([value.detach().cpu().numpy() for value in inputs])
        outputs = [torch.from_numpy(np.asarray(result[output]).copy()) for output in self.compiled.outputs]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class OpenVinoBottleObserver(RgbBottleObserver):
    def __init__(self, xml, minimum_views=2, device='CPU', rigid_geometry=False):
        if minimum_views not in (2, 3):
            raise ValueError('Require two or three calibrated views.')
        self.minimum_views = minimum_views; self.device = torch.device('cpu')
        self.rigid_geometry = bool(rigid_geometry)
        self.network = NetworkAdapter(xml, device)


class OpenVinoCartesianMotorPolicy(CartesianMotorPolicy):
    def __init__(self, xml, checkpoint, model, device='CPU'):
        self.device = torch.device('cpu'); self.network = NetworkAdapter(xml, device, checkpoint)
        self.geometry = RobotGeometry(model); self.ranges = np.asarray(model.actuator_ctrlrange).copy()
