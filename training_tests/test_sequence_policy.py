import unittest
import tempfile,json
from pathlib import Path
import numpy as np
import torch
from simulation_lab.sequence_policy import SequenceNet,SequencePolicy
from simulation_lab.retrieval_policy import KEYS,ObservationRejected
from safetensors.torch import save_file

class SequenceContractTests(unittest.TestCase):
    def test_tracking_guard_preserves_memory_and_has_bounded_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);net=SequenceNet()
            with torch.no_grad():
                for p in net.parameters():p.zero_()
                net.head[-1].bias.fill_(.1)
            save_file(net.state_dict(),str(folder/'sequence.safetensors'))
            np.savez(folder/'visual.npz',mean=np.zeros(6912),components=np.zeros((32,6912)),scale=np.ones(32))
            meta={'arguments':{},'state_mean':[0.]*24,'state_std':[1.]*24,'action_scale':[1.]*12,
                  'action_mean':[0.]*12,'action_mode':'absolute','reconstruction_limit':1.,
                  'tracking_tolerance_rad':.005,'tracking_replay_chunk':True}
            (folder/'sequence.json').write_text(json.dumps(meta))
            policy=SequencePolicy(folder,'cpu');batch={k:torch.zeros(1,3,240,320) for k in KEYS}
            batch['observation.state']=torch.zeros(1,24)
            with torch.inference_mode():
                first=policy.predict_action_chunk(batch);hidden=policy.hidden.clone()
                for _ in range(50):torch.testing.assert_close(first,policy.predict_action_chunk(batch))
                torch.testing.assert_close(policy.hidden,hidden)
                self.assertEqual(policy.calls,1)
                with self.assertRaises(ObservationRejected):policy.predict_action_chunk(batch)
                batch['observation.state'][0,:5]=.1
                policy.predict_action_chunk(batch)
                self.assertEqual(policy.calls,2)

    def test_initial_observation_context_matches_incremental_execution(self):
        torch.set_num_threads(2);torch.manual_seed(12)
        model=SequenceNet(initial_context=True).eval();x=torch.randn(2,9,56)
        with torch.inference_mode():
            full,_=model(x);hidden=None;steps=[]
            for i in range(x.shape[1]):
                output,hidden=model(x[:,i:i+1],hidden,x[:,:1]);steps.append(output)
            torch.testing.assert_close(full,torch.cat(steps,dim=1),atol=1e-6,rtol=1e-5)

    def test_incremental_inference_matches_training_sequence_and_is_causal(self):
        torch.set_num_threads(2);torch.manual_seed(12)
        model=SequenceNet().eval();x=torch.randn(2,9,56)
        with torch.inference_mode():
            full,_=model(x);hidden=None;steps=[]
            for i in range(x.shape[1]):
                output,hidden=model(x[:,i:i+1],hidden);steps.append(output)
            torch.testing.assert_close(full,torch.cat(steps,dim=1),atol=1e-6,rtol=1e-5)
            changed=x.clone();changed[:,5:]+=10
            alternate,_=model(changed)
            torch.testing.assert_close(full[:,:5],alternate[:,:5],atol=1e-6,rtol=1e-5)
            reset,_=model(x[:,:1])
            torch.testing.assert_close(reset,full[:,:1],atol=1e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
