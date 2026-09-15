"""Independently check all local-motor labels with separate Torch tree geometry."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from scripts.mug_correction_motor_v1 import read,sha,space,write,data_gate
from scripts.torch_robot_geometry import TorchRobotGeometry


def audit(args):
    p=read(args.protocol)
    folder=ROOT/p['raw_root']/args.split
    output=folder/'audit.json'
    if output.exists():raise FileExistsError('Preserve previous audits.')
    preflight=space(p,output,4*1024**2)
    manifest,arrays=data_gate(p,folder,args.protocol)
    model=mujoco.MjModel.from_xml_path(str(ROOT/p['scene']))
    torch.set_num_threads(2)
    geometry=TorchRobotGeometry(model,'right',dtype=torch.float64)
    geometry.tool.copy_(torch.tensor(p['tool_point_local_m'],dtype=torch.float64))
    started=time.perf_counter()
    q=torch.tensor(arrays['joints'].astype(float))
    # Match the actual float32 subtraction used by the saved label command.
    delta=arrays['labels']-arrays['joints']
    with torch.inference_mode():
        point,rotation=geometry(q)
        final,final_rotation=geometry(q+torch.tensor(delta.astype(float)))
    movement=(final-point).numpy()
    position=np.linalg.norm(movement-arrays['translation'],axis=1)*1000
    axis=torch.linalg.vector_norm(final_rotation[:,:,2]-rotation[:,:,2],dim=1).numpy()
    expected=np.asarray([r['actual_translation_m'] for r in manifest['rows']])
    displacement_difference=float(np.max(abs(movement-expected)))
    position_difference=float(np.max(abs(position-np.asarray([r['position_error_mm'] for r in manifest['rows']]))))
    axis_difference=float(np.max(abs(axis-np.asarray([r['axis_error'] for r in manifest['rows']]))))
    ids=np.unique(np.linspace(0,len(q)-1,64,dtype=int))
    epsilon=1e-6
    finite_difference=[]
    with torch.inference_mode():
        for index in ids:
            repeated=q[index:index+1].repeat(5,1)
            eye=torch.eye(5,dtype=torch.float64)*epsilon
            plus,rot_plus=geometry(repeated+eye)
            minus,rot_minus=geometry(repeated-eye)
            jp=((plus-minus)/(2*epsilon)).numpy().T
            ja=((rot_plus[:,:,2]-rot_minus[:,:,2])/(2*epsilon)).numpy().T*p['data']['jacobian_axis_weight_m']
            jac=np.vstack((jp,ja))
            matrix=np.linalg.solve(jac.T@jac+np.eye(5)*p['data']['damping'],jac.T[:,:3])
            finite_difference.append(float(np.max(abs(matrix-arrays['coefficients'][index]))))
    coefficient_difference=max(finite_difference)
    passed=displacement_difference<1e-9 and position_difference<1e-6 and axis_difference<1e-9 and coefficient_difference<.002
    write(p,output,{'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':sha(Path(__file__)),
        'geometry_source_sha256':sha(ROOT/'scripts/torch_robot_geometry.py'),'data_sha256':sha(folder/'data.npz'),
        'states':len(q),'separate_forward_geometry_maximum_translation_difference_m':displacement_difference,
        'maximum_score_difference_mm':position_difference,'maximum_axis_score_difference':axis_difference,
        'finite_difference_indices':ids.tolist(),'finite_difference_maximum_coefficient_difference_rad_per_m':coefficient_difference,
        'finite_difference_all_maximum_differences_rad_per_m':finite_difference,'preflight':preflight,
        'passed':passed,'wall_seconds':time.perf_counter()-started,'new_physical_trials':0,
        'scope':'All saved endpoint labels checked with separate forward tree geometry; 64 Jacobians checked with central finite differences. No trained accuracy or physical success claim.'})
    print({'split':args.split,'states':len(q),'passed':passed,'maximum_translation_difference_m':displacement_difference,
        'finite_difference_maximum_coefficient_difference_rad_per_m':coefficient_difference},flush=True)
    if not passed:raise ValueError('Independent geometry audit failed; preserve its result.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-correction-motor-v1.json')
    parser.add_argument('--split',choices=['training','development','evaluation'],required=True)
    audit(parser.parse_args())
