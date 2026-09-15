"""Small, real CUDA trajectory inference for the ZeroGPU hosting tier."""
import json, math
import spaces  # Initialize ZeroGPU's CUDA emulation before constructing models.
import numpy as np
import torch
from safetensors.torch import load_file
from simulation_lab.primitive_policy import PrimitiveNet
from simulation_lab.public_trial import load_checkpoints

MODELS, METADATA = {}, {}
for name, folder in load_checkpoints().items():
    net = PrimitiveNet()
    net.load_state_dict(load_file(str(folder/'primitive.safetensors'), device='cpu'))
    MODELS[name] = net.eval().to('cuda')
    METADATA[name] = json.loads((folder/'primitive.json').read_text())


@spaces.GPU(duration=10)
def generate_trajectory(skill, context, max_seconds):
    if skill not in MODELS or float(max_seconds) != float(METADATA[skill]['max_seconds']):
        raise ValueError('Unknown hosted model configuration.')
    values = np.asarray(context,dtype=np.float32)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError('Invalid visual context.')
    count = math.ceil(max_seconds*20)+1
    seconds = (torch.arange(count,device='cuda',dtype=torch.float32)/20).clamp(max=max_seconds)
    visual = torch.as_tensor(values,device='cuda')[None].expand(count,-1)
    with torch.inference_mode(): actions = MODELS[skill](visual,seconds).cpu().numpy()
    return {'actions':actions,'device':torch.cuda.get_device_name(0)}
