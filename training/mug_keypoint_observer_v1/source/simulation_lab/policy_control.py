"""Shared observation-independent actuator contract for learned bottle control."""
import numpy as np

GRIPPER_CAP_NM=.25

def apply_targets(model,data,target,cap=GRIPPER_CAP_NM,arm_offset=0):
    target=np.asarray(target,dtype=float)
    if target.shape!=(12,) or not np.isfinite(target).all():
        raise ValueError('Expected twelve finite joint position targets.')
    if not 0<cap<=.65:raise ValueError('Invalid gripper torque cap.')
    if type(arm_offset) is not int or arm_offset not in (0,6):raise ValueError('Choose the left or right arm offset.')
    target=np.clip(target,model.actuator_ctrlrange[:12,0],model.actuator_ctrlrange[:12,1])
    result=target.copy();i=arm_offset+5
    kp=model.actuator_gainprm[i,0];kv=-model.actuator_biasprm[i,2]
    torque=kp*(target[i]-data.qpos[i])-kv*data.qvel[i]
    result[i]=data.qpos[i]+(np.clip(torque,-cap,cap)+kv*data.qvel[i])/kp
    return result
