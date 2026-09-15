"""Packaged CPU neural local corrections from measured joints and an XY request."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import openvino as ov
from .mug_correction_motor import MugRobotGeometry
from .mug_correction_limits import assess_limited


class LocalMugMotor:
    def __init__(self, folder, robot_model):
        self.folder=Path(folder)
        self.meta=json.loads((self.folder/'runtime.json').read_text())
        if self.meta['schema']!='talos.mug-joint-limited-runtime.v1':
            raise ValueError('Unexpected mug motor contract.')
        for name in ('motor.xml','motor.bin'):
            if hashlib.sha256((self.folder/'openvino'/name).read_bytes()).hexdigest()!=self.meta['ir_sha256'][name]:
                raise ValueError('Packaged motor IR checksum mismatch.')
        self.geometry=MugRobotGeometry(robot_model)
        self.core=ov.Core()
        self.compiled=self.core.compile_model(str(self.folder/'openvino/motor.xml'),'CPU',
            {'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':2})
        self.device=self.core.get_property('CPU','FULL_DEVICE_NAME')

    def predict(self, measured_joints, requested_translation):
        q=np.asarray(measured_joints,dtype=np.float32)
        d=np.asarray(requested_translation,dtype=np.float32)
        if q.shape!=(5,) or d.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(d).all():
            raise ValueError('Five measured joints and one finite XY correction are required.')
        if abs(float(d[2]))>1e-12 or np.linalg.norm(d)>self.meta['guard']['maximum_translation_norm_m']+1e-9:
            raise ValueError('The requested local correction exceeds the declared XY step range.')
        started=time.perf_counter()
        raw=self.compiled([q[None],d[None]])[0].copy()[0]
        latency=(time.perf_counter()-started)*1000
        result=assess_limited(self.geometry,q,d,raw,self.meta['support'],self.meta['guard'],self.meta['step_sizing'])
        result['neural_inference_ms']=latency
        return result
