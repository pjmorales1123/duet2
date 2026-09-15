"""Promotion and hosted-mode boundaries without executing a physics trial."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from simulation_lab import mug_visual_profile as profiles
from simulation_lab.public_trial import run_trial


class MugVisualProfileContracts(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.profile=self.root/'profile.json'
        self.root_patch=patch.object(profiles,'ROOT',self.root);self.root_patch.start()
        for name,value in (('model.bin',b'original model'),('runtime.py',b'original runtime'),('suite.json',b'{"mug":"mug"}')):
            (self.root/name).write_bytes(value)
        self.write('protocol.json',{'baseline_suite':'suite.json','baseline_suite_sha256':self.sha('suite.json')})
        for split in ('development','evaluation'):
            self.write(split+'-audit.json',{'gate_passed':True,'requirements':{'physical_gate':True}})
        self.metadata={'schema':'talos.promoted-mug-vision.v1','name':'unit-fixture',
            'protocol_sha256':self.sha('protocol.json'),
            'model_files':{'model.bin':self.sha('model.bin')},'runtime_sources':{'runtime.py':self.sha('runtime.py')}}
        self.refresh()

    def tearDown(self):
        self.root_patch.stop();self.temporary.cleanup()

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value),encoding='utf-8')

    def sha(self,name):return hashlib.sha256((self.root/name).read_bytes()).hexdigest()

    def refresh(self):
        self.metadata['evidence_files']={name:self.sha(name) for name in ('protocol.json','development-audit.json','evaluation-audit.json')}
        self.write('profile.json',self.metadata)

    def test_missing_profile_cannot_select_a_replacement(self):
        with self.assertRaisesRegex(ValueError,'unavailable'):
            profiles.make_sequence(None,None,None,{},[],self.root/'missing.json')

    def test_failed_development_or_final_evidence_is_refused(self):
        for split in ('development','evaluation'):
            with self.subTest(split=split):
                self.write(split+'-audit.json',{'gate_passed':False,'requirements':{'physical_gate':False}});self.refresh()
                with self.assertRaisesRegex(ValueError,'promotion gates'):profiles.load_profile(self.profile)
                self.write(split+'-audit.json',{'gate_passed':True,'requirements':{'physical_gate':True}});self.refresh()

    def test_changed_model_or_runtime_is_refused(self):
        for name in ('model.bin','runtime.py'):
            with self.subTest(name=name):
                original=(self.root/name).read_bytes();(self.root/name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'dependency changed'):profiles.load_profile(self.profile)
                (self.root/name).write_bytes(original)

    def test_partial_mug_plan_cannot_claim_full_workflow_verification(self):
        with self.assertRaisesRegex(ValueError,'full table setting'):
            profiles.make_sequence(None,None,None,{'mug':self.root/'mug'},['mug'],self.profile)

    def test_shared_gpu_visual_mode_is_refused_before_scene_creation(self):
        with patch('simulation_lab.public_trial.build_scene',side_effect=AssertionError('No scene should be created.')):
            trial=run_trial('set the table',controller='learned_visual',policy_factory=object())
            try:
                image,message,report=next(trial)
                self.assertIsNone(image);self.assertEqual(report['status'],'refused');self.assertIn('CPU',message)
            finally:trial.close()


if __name__=='__main__':unittest.main()
