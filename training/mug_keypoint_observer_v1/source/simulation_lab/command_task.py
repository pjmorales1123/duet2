"""Command sequencing for the explicitly labeled simulator-state skill baseline."""
from copy import deepcopy
import numpy as np
from .dinner_autonomy import DinnerSequence,DinnerTask,DrawerTask,SKILLS
from .language import CommandError

def ground_destination(request,item_id,layout,positions):
    """Resolve geometric relations from a supplied observation mapping.

    Caller identifies whether positions are camera estimates or simulator state.
    No body coordinates are changed here.
    """
    if not request or request['kind']=='default':return None
    if request['kind']=='handoff':raise CommandError('Relay instructions must be expanded into physical grasp and release steps.')
    z=layout['table_z']
    if request['kind']=='point':
        point=np.asarray(request['position'],dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all():raise CommandError('Invalid internal destination point.')
        point=point.copy();point[2]=z
    elif request['kind']=='spot':
        x={'left':-.060,'right':.110,'center':.025}[request['side']]
        if request.get('far'):x=-.28 if request['side']=='left' else .28 if request['side']=='right' else x
        point=np.array([x,-.025,z])
    elif request['kind']=='relative':
        ref=request['reference']
        if ref=='plate' and request.get('reference_side'):
            candidates=[name for name in ('plate','side_plate') if name in positions and name!=item_id]
            if not candidates:raise CommandError('No other plate is visible.')
            ref=sorted(candidates,key=lambda name:positions[name][0])[0 if request['reference_side']=='left' else -1]
        if ref not in positions:raise CommandError('The reference item is not visible.')
        if ref==item_id:raise CommandError('Cannot place an item relative to itself.')
        specs={o['id']:o for o in layout['objects']}
        offset=(max(specs[item_id]['size_m'][:2])+max(specs[ref]['size_m'][:2]))/2+.035
        dx,dy={'front':(0,-1),'behind':(0,1),'left':(-1,0),'right':(1,0),'beside':(1,0)}[request['relation']]
        point=np.array([positions[ref][0]+dx*offset,positions[ref][1]+dy*offset,z])
    else:raise CommandError('Unknown destination type.')
    if abs(point[0])>.40 or not -.22<=point[1]<=.32:raise CommandError('The destination lies outside the usable tabletop.')
    moving=next(o for o in layout['objects'] if o['id']==item_id)
    for other in layout['objects']:
        if other['id']==item_id or other['id'] not in positions:continue
        radius=(max(moving['size_m'][:2])+max(other['size_m'][:2]))/2+.005
        if np.linalg.norm(point[:2]-np.array(positions[other['id']])[:2])<radius:
            raise CommandError('The requested spot is occupied by '+other['label']+'. Choose a clear location.')
    return point

class CommandSequence(DinnerSequence):
    def start_plan(self,plan):
        if self.active:raise CommandError('A task is already running.')
        if self.layout.get('dinner_preset')!='task':raise CommandError('Load a dinner Task start scene first.')
        self.plan=deepcopy(plan);expanded=[]
        drawer_ready=self.data.joint('drawer_slide').qpos[0]>.105
        for intent in plan['steps']:
            if (intent.get('destination') or {}).get('kind')=='handoff':
                if intent['object_id']!='bottle' or intent['destination']['recipient']!='right':
                    raise CommandError('The verified relay currently transfers the bottle from left to right using a shared table area.')
                if intent.get('arm','auto') not in ('auto','left'):
                    raise CommandError('The left arm starts the bottle relay to the right arm.')
                positions={o['id']:self.data.body(o['body']).xpos.copy() for o in self.layout['objects']}
                for arm,point in [('left',[0.,-.10,self.layout['table_z']]),('right',[.20,.05,self.layout['table_z']])]:
                    destination={'kind':'point','position':point}
                    ground_destination(destination,'bottle',self.layout,positions)
                    expanded.append({'kind':'dinner_place','object_id':'bottle','arm':arm,'destination':destination,'relay':True})
                continue
            if intent['kind']=='set_table':
                expanded.extend({'kind':'drawer_open' if n=='drawer' else 'dinner_place','object_id':None if n=='drawer' else n,'arm':'auto','destination':None} for n in SKILLS)
                drawer_ready=True;continue
            if intent['kind']=='drawer_open':drawer_ready=True
            if intent['object_id'] in ('fork','spoon') and not drawer_ready:
                expanded.append({'kind':'drawer_open','object_id':None,'arm':'left','destination':None});drawer_ready=True
            if intent['object_id'] in ('glass','side_plate'):raise CommandError('That item is not a verified physical skill yet. Choose plate, bottle, mug, fork or spoon.')
            expanded.append(intent)
        self.intents=expanded;self.steps=['drawer' if p['kind']=='drawer_open' else p['object_id'] for p in expanded]
        self.requested_side='auto';self.kind='language_sequence';self.status='running';self._next()
    def _next(self):
        intent=self.intents[len(self.results)];name=self.steps[len(self.results)]
        layout=deepcopy(self.layout)
        if name!='drawer':
            positions={o['id']:self.data.body(o['body']).xpos.copy() for o in layout['objects']}
            point=ground_destination(intent.get('destination'),name,layout,positions)
            if point is not None:
                layout['targets']=[t for t in layout['targets'] if t['object_id']!=name]
                layout['targets'].append({'id':name+'_command_destination','object_id':name,'position_m':point.tolist(),'radius_m':.012})
        self.child=DrawerTask(self.model,self.data,layout) if name=='drawer' else DinnerTask(self.model,self.data,layout)
        self.child.start(side=intent.get('arm','auto'),object_id=name)
        self.stage=self.child.stage
    def update(self,targets):
        try:super().update(targets)
        except CommandError as exc:
            targets[:]=self.data.qpos[:12];self._finish('failed',str(exc),True)
    def snapshot(self):
        result=super().snapshot();result.update(instruction=self.plan['instruction'],language_interpreter='constrained_grammar',
            observation='exact_simulator_state',policy_mode='programmed_physical_skills',intent_plan=self.intents)
        if any(i.get('relay') for i in self.intents):
            result['cooperation']='Left arm releases the bottle on the table and parks; right arm regrips and places it. Table-supported relay.'
            if self.child and self.intents[min(len(self.results),len(self.intents)-1)].get('relay'):
                result['stage_label']=f'{self.child.side or "Planning"} arm · {self.child.snapshot()["stage_label"]}'
        return result
