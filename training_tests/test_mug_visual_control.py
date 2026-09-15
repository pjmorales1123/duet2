"""No-physics contracts for corrective chunks, tracking waits and frozen images."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from simulation_lab.mug_visual_control import MugCorrectionPolicy
from simulation_lab.primitive_policy import PrimitivePolicy
from simulation_lab.retrieval_policy import ObservationRejected

ROOT=Path(__file__).resolve().parents[1]


class Observer:
    def __init__(self):self.images=[]
    def observe(self,images,calibrations):
        self.images.append(images.copy())
        return {'status':'observed','points_m':[[.263,.005,.763],[.263,.005,.821]]}


class Motor:
    def __init__(self,accepted=True):self.accepted=accepted
    def predict(self,q,request):return {'accepted':self.accepted,'joint_delta_rad':[.001]*5}


def policy(mode='live',motor=None):
    protocol=json.loads((ROOT/'docs/robotics/experiments/mug-visual-correction-v1.json').read_text())
    observer=Observer()
    model=SimpleNamespace(actuator_ctrlrange=np.tile([-3.,3.],(12,1)))
    p=MugCorrectionPolicy(ROOT/'models/dinner_suite/mug','cpu',model,protocol,mode,observer=observer,motor=motor or Motor())
    p.progress=496;p.force_wait=False
    return p,observer


def advance(self,batch):
    if self.force_wait:return self.previous
    self.progress+=4;self.previous=torch.zeros(1,20,12)
    return self.previous


def batch(pixel=0):
    return {'observation.state':torch.zeros(1,24),'mug.rgb':np.full((2,240,320,3),pixel,np.uint8),'mug.calibrations':[{},{}]}


class MugVisualControlContracts(unittest.TestCase):
    def test_tracking_wait_neither_accumulates_nor_doubles_offset(self):
        p,observer=policy()
        with patch.object(PrimitivePolicy,'predict_action_chunk',advance):
            first=p.predict_action_chunk(batch()).clone()
            p.force_wait=True
            waited=p.predict_action_chunk(batch(1))
            self.assertTrue(torch.equal(first,waited))
            self.assertEqual((len(p.corrections),len(observer.images)),(1,1))
            p.force_wait=False
            second=p.predict_action_chunk(batch(2))
            self.assertTrue(torch.allclose(second[0,:,6:11],torch.full((20,5),.002),rtol=0,atol=1e-8))
            self.assertTrue(torch.equal(p.previous,second))

    def test_image_control_uses_declared_live_or_copied_frozen_pixels(self):
        for mode,expected in (('live',7),('frozen',0)):
            with self.subTest(mode=mode):
                p,observer=policy(mode)
                with patch.object(PrimitivePolicy,'predict_action_chunk',advance):
                    initial=batch();p.predict_action_chunk(initial)
                    initial['mug.rgb'][:]=7;p.predict_action_chunk(initial)
                self.assertTrue(np.all(observer.images[-1]==expected))
                self.assertEqual(p.corrections[-1]['current_rgb_sha256']==p.corrections[-1]['used_rgb_sha256'],mode=='live')

    def test_motor_refusal_and_offset_limit_issue_no_correction(self):
        for motor,offset in ((Motor(False),0.),(Motor(),.12)):
            with self.subTest(offset=offset):
                p,_=policy(motor=motor);p.joint_offset[:]=offset
                with patch.object(PrimitivePolicy,'predict_action_chunk',advance),self.assertRaises(ObservationRejected):
                    p.predict_action_chunk(batch())
                self.assertTrue(np.array_equal(p.joint_offset,np.full(5,offset)))
                self.assertEqual(p.corrections[-1]['status'],'refused')

    def test_taper_returns_exact_nominal_targets_before_parking(self):
        p,_=policy();p.joint_offset[:]=.05;p.progress=int(35.6*20)
        with patch.object(PrimitivePolicy,'predict_action_chunk',advance):
            result=p.predict_action_chunk(batch())
        self.assertTrue(torch.equal(result,torch.zeros(1,20,12)))
        self.assertFalse(p.corrections)

    def test_exact_cancellation_clears_the_stored_offset_and_records_query(self):
        p,_=policy()
        with patch.object(PrimitivePolicy,'predict_action_chunk',advance):
            p.predict_action_chunk(batch())
            p.motor=SimpleNamespace(predict=lambda q,d:{'accepted':True,'joint_delta_rad':[-.001]*5})
            result=p.predict_action_chunk(batch(1))
        self.assertTrue(torch.equal(result,torch.zeros(1,20,12)))
        self.assertTrue(np.array_equal(p.joint_offset,np.zeros(5)))
        self.assertEqual(len(p.corrections),2)
        self.assertEqual(p.corrections[-1]['status'],'issued')


if __name__=='__main__':unittest.main()
