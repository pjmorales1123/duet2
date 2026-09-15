import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from simulation_lab.act_learning import config,ACTPolicy,IMAGE_KEYS,learning_loss,episode_indices,normalize,denormalize,normalization
from simulation_lab.storage import require_space,GIB


class DiagnosticTests(unittest.TestCase):
    def test_relative_action_roundtrip_uses_current_state(self):
        import numpy as np
        stats=normalization({k:{'mean':np.zeros(n),'std':np.ones(n)} for k,n in [('observation.state',24),('action',12)]})
        stats['action_representation']='relative'
        state=torch.randn(2,24);actions=torch.randn(2,20,12)
        batch={'observation.state':state,'action':actions,**{k:torch.zeros(2,3,32,32) for k in IMAGE_KEYS}}
        normalized=normalize(batch,stats,'cpu')
        torch.testing.assert_close(denormalize(normalized['action'],stats,state),actions)
        with self.assertRaises(ValueError):denormalize(normalized['action'],stats)
        torch.testing.assert_close(denormalize(torch.zeros_like(actions),stats,state),state[:,None,:12].expand_as(actions))

    def test_inference_objective_matches_prediction_and_has_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        cfg=config('cpu',False);cfg.dim_model=32;cfg.n_heads=4
        cfg.dim_feedforward=64;cfg.n_encoder_layers=1;cfg.n_decoder_layers=1
        policy=ACTPolicy(cfg)
        batch={'observation.state':torch.randn(1,24),'action':torch.randn(1,20,12),
               'action_is_pad':torch.tensor([[False]*17+[True]*3]),
               **{key:torch.rand(1,3,32,32) for key in IMAGE_KEYS}}
        reference=policy.predict_action_chunk({k:v for k,v in batch.items() if k.startswith('observation.')})
        expected=(reference[:,:17]-batch['action'][:,:17]).abs().mean()
        loss,info=learning_loss(policy,batch,'inference-l1')
        torch.testing.assert_close(loss.detach(),expected)
        self.assertNotIn('kld_loss',info)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.parameters()))
        self.assertTrue(all(p.grad is None for name,p in policy.named_parameters() if 'vae_' in name))

    def test_subset_bounds_and_disk_reserve(self):
        self.assertEqual(episode_indices({'episodes':[{'source':'a','frames':3},{'source':'b','frames':2}]},'b'),[3,4])
        with patch('simulation_lab.storage.shutil.disk_usage',return_value=SimpleNamespace(free=8*GIB)):
            with self.assertRaises(OSError):require_space('.',GIB)


if __name__=='__main__':unittest.main()
