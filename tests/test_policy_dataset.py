"""Policy schema must exclude simulator privilege and incomplete image exports."""
import gzip,json,tempfile,unittest
from pathlib import Path
from simulation_lab.dataset import load_policy_episode

class PolicyDatasetTests(unittest.TestCase):
    def test_privileged_state_cannot_enter_policy_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest={'training_eligible':True,'images':{'status':'completed','hz':20,'cameras':['overhead'],'index':'images.jsonl'},
                      'actions':'actions.gz','observations':'observations.gz'}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'frame.jpg').write_bytes(b'loader only resolves paths; dataset validator checks images')
            (root/'images.jsonl').write_text(json.dumps({'observation_index':0,'camera':'overhead','path':'frame.jpg'}))
            for name,row in [('actions',{'time_s':0,'target':[0]*12}),('observations',{'index':0,'action_index':0,'time_s':0,'terminal':False,
                              'robot_joint_position':[0]*12,'robot_joint_velocity':[0]*12,'qpos':[123]*40,'qvel':[456]*36,'actuator_force':[789]*12})]:
                with gzip.open(root/(name+'.gz'),'wt') as f:f.write(json.dumps(row)+'\n')
            sample=load_policy_episode(root)[0]
            self.assertEqual(set(sample),{'time_s','joint_position','joint_velocity','action','images'})
            manifest['images']['status']='pending'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_policy_episode(root)

if __name__=='__main__':unittest.main()
