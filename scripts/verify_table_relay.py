"""Verify physical contact for both legs of a table-supported relay."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from scripts.evaluate_dinner_scene import load
from simulation_lab.relay_task import TableRelay
from simulation_lab.scene import HOME
from simulation_lab.autonomy import PlanningError
from simulation_lab.dinner_autonomy import CLOSED
from simulation_lab.storage import require_space

def main(a):
    require_space(a.output,32*1024**2);m,d,layout=load(a.seed)
    for _ in range(200):mujoco.mj_step(m,d)
    d.time=0.;task=TableRelay(m,d,layout);task.start(recipient=a.recipient,destination=a.destination);targets=np.array(HOME*2)
    for _ in range(50000):
        before=d.qpos.copy();velocity=d.qvel.copy();task.update(targets)
        assert np.array_equal(before,d.qpos) and np.array_equal(velocity,d.qvel)
        assert m.neq==0 and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
        if not task.active:break
        d.ctrl[:]=task.apply_gripper_limit(targets);mujoco.mj_step(m,d)
    if task.active:task.cancel(targets)
    report=task.snapshot();report.update(seed=a.seed,physical_state_writes=0,equality_constraints=int(m.neq))
    if a.probe and task.status=='failed' and task.child.stage=='hold':
        child=task.child;reference=child._carry_reference();candidates=[];before=d.qpos.copy()
        for x in [.08,.12,.16,.20,.24,.28]:
            for y in [-.15,-.10,-.05,0.,.05,.10]:
                try:
                    center=np.array([x,y,d.body('bottle').xpos[2]])
                    point,_=child._point_for_center(center,reference,d.qpos[child.offset:child.offset+5])
                    child._move_point('align',point,CLOSED,4.)
                    candidates.append([x,y])
                except PlanningError:pass
        assert np.array_equal(before,d.qpos)
        report['planning_only_receiver_candidates']=candidates;print('receiver candidates',candidates,flush=True)
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({'status':task.status,'message':task.message,
         'legs':[{'arm':r['arm'],'status':r['status'],'metrics':r['metrics']} for r in task.results]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--recipient',choices=['left','right'],default='right')
    p.add_argument('--destination',nargs=3,type=float)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--output',required=True);main(p.parse_args())
