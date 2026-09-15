"""Generate each image-conditioned trajectory on the hosted GPU once.

The physics worker has no CUDA context. Its existing progress guard samples the
fresh neural output at the same 20 Hz times. This caches current model output,
not demonstration actions. Each skill still observes its own initial images.
"""
import math
from pathlib import Path
from queue import Empty
import numpy as np
import torch
from .primitive_policy import PrimitivePolicy
from .policy_observation import ObservationRejected


class GeneratedTrajectoryNet(torch.nn.Module):
    def __init__(self, skill, max_seconds, request, response, stop):
        super().__init__()
        self.register_parameter('_device_anchor', torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        self.skill, self.max_seconds = skill, float(max_seconds)
        self.request, self.response, self.stop = request, response, stop
        self.context = self.generated = None
        self.execution_devices = []

    def forward(self, visual, seconds):
        if visual.ndim != 2 or visual.shape[1] != 32 or not len(visual) or not torch.isfinite(visual).all() or not torch.isfinite(seconds).all():
            raise ObservationRejected('Invalid hosted visual context or action time.')
        if not torch.equal(visual, visual[:1].expand_as(visual)):
            raise ObservationRejected('A hosted trajectory requires one shared initial visual context.')
        context = visual[0].detach().cpu().numpy()
        if self.generated is None:
            self.request.put(('inference', (self.skill, context, self.max_seconds)), timeout=5)
            while not self.stop.is_set():
                try:
                    reply = self.response.get(timeout=.25)
                    break
                except Empty:
                    continue
            else:
                raise ObservationRejected('Trial cancelled while waiting for neural inference.')
            if reply.get('error'):
                reasons = {
                    'quota': 'The hosting service reports that the current GPU usage allowance is exhausted. Try again after it resets.',
                    'timeout': 'The hosting service timed out while generating the neural trajectory. Start a fresh trial later.',
                    'configuration': 'The hosted model configuration could not be validated.',
                    'allocation': 'The hosting service could not allocate or run the GPU for this skill. Start a fresh trial later.',
                }
                raise ObservationRejected(reasons.get(reply['error'], 'Hosted neural inference failed. Start a fresh trial later.'))
            values = np.asarray(reply['actions'], dtype=np.float32)
            expected = math.ceil(self.max_seconds*20)+1
            if values.shape != (expected,12) or not np.isfinite(values).all():
                raise ObservationRejected('Invalid hosted neural trajectory.')
            self.generated = torch.from_numpy(values.copy())
            self.context = context.copy()
            self.execution_devices = [reply['device']]
        elif not np.array_equal(context, self.context):
            raise ObservationRejected('This hosted cache supports initial-image conditioning only.')
        # All queried times are the existing 20 Hz grid, with a final clamp.
        indices = torch.round(seconds.detach().cpu()*20).long().clamp(0,len(self.generated)-1)
        return self.generated[indices]


class HostedGpuPolicy(PrimitivePolicy):
    def details(self):
        result = super().details()
        result.update(neural_runtime='PyTorch CUDA via ZeroGPU',
                      neural_execution_devices=self.net.execution_devices,
                      trajectory_cache='Fresh network predictions from this skill\'s initial image; no demonstration retrieval')
        return result


def make_gpu_policy_factory(checkpoints, request, response, stop):
    names = {Path(folder).resolve(): name for name,folder in checkpoints.items()}
    def factory(folder, device):
        policy = HostedGpuPolicy(folder, device)
        if policy.feedback:
            raise ObservationRejected('Hosted trajectory generation does not support the experimental visual-refresh model.')
        policy.net = GeneratedTrajectoryNet(names[Path(folder).resolve()], policy.meta['max_seconds'], request, response, stop)
        return policy
    return factory
