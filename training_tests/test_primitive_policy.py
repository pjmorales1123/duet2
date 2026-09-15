import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from simulation_lab.primitive_policy import PrimitiveNet,PrimitivePolicy
from simulation_lab.retrieval_policy import KEYS,ObservationRejected

class PrimitiveContractTests(unittest.TestCase):
    def test_joint_guard_holds_progress_and_rejects_persistent_lag(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);net=PrimitiveNet()
            with torch.no_grad():
                for p in net.parameters():p.zero_()
                net.net[-1].bias.fill_(.1)
            save_file(net.state_dict(),str(folder/'primitive.safetensors'))
            np.savez(folder/'visual.npz',mean=np.zeros(6912),components=np.zeros((32,6912)),scale=np.ones(32))
            meta={'visual_mean':[0.]*32,'visual_std':[1.]*32,'action_std':[1.]*12,'action_mean':[0.]*12,
                  'max_seconds':60,'tracking_tolerance_rad':.005,'reconstruction_limit':1.}
            (folder/'primitive.json').write_text(json.dumps(meta));policy=PrimitivePolicy(folder,'cpu')
            batch={k:torch.zeros(1,3,240,320) for k in KEYS};batch['observation.state']=torch.zeros(1,24)
            with torch.inference_mode():
                first=policy.predict_action_chunk(batch)
                for _ in range(50):torch.testing.assert_close(policy.predict_action_chunk(batch),first)
                self.assertEqual(policy.progress,4)
                with self.assertRaises(ObservationRejected):policy.predict_action_chunk(batch)
                batch['observation.state'][0,:5]=.1;policy.predict_action_chunk(batch)
                self.assertEqual(policy.progress,8)
                batch['observation.state'][0,0]=float('nan')
                with self.assertRaises(ObservationRejected):policy.predict_action_chunk(batch)

if __name__=='__main__':unittest.main()
