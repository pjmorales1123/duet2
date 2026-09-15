"""Table-supported two-arm relay. No direct airborne handoff is claimed."""
from copy import deepcopy
import numpy as np
from .dinner_autonomy import DinnerSequence,DinnerTask
from .command_task import ground_destination
from .language import CommandError


class TableRelay(DinnerSequence):
    def start(self,recipient='right',object_id='bottle',destination=None,**kwargs):
        if self.active:raise CommandError('Cancel the current task first.')
        if recipient not in ('left','right') or object_id!='bottle':raise CommandError('This relay prototype supports the bottle between left and right arms.')
        if self.layout.get('dinner_preset')!='task':raise CommandError('Load a dinner Task start scene.')
        self.__init__(self.model,self.data,self.layout)
        self.recipient=recipient;self.donor='left' if recipient=='right' else 'right'
        self.final_destination=np.asarray(destination if destination is not None else ([.20,.05,self.layout['table_z']] if recipient=='right' else [-.20,.05,self.layout['table_z']]),dtype=float)
        self.relay_point=np.array([0.,-.10,self.layout['table_z']])
        self.kind='table_relay';self.steps=['bottle','bottle'];self.status='running';self._next()

    def _next(self):
        index=len(self.results);arm=self.donor if index==0 else self.recipient
        point=self.relay_point if index==0 else self.final_destination
        layout=deepcopy(self.layout)
        positions={o['id']:self.data.body(o['body']).xpos.copy() for o in layout['objects']}
        ground_destination({'kind':'point','position':point},'bottle',layout,positions)
        layout['targets']=[t for t in layout['targets'] if t['object_id']!='bottle']
        layout['targets'].append({'id':'shared_relay_area' if index==0 else 'receiver_serving_area','object_id':'bottle','position_m':point.tolist(),'radius_m':.012})
        self.child=DinnerTask(self.model,self.data,layout);self.child.start(side=arm,object_id='bottle');self.stage=self.child.stage

    def update(self,targets):
        try:super().update(targets)
        except CommandError as exc:
            targets[:]=self.data.qpos[:12];self._finish('failed',str(exc),True)

    def snapshot(self):
        result=super().snapshot();result.update(policy_mode='programmed_table_relay',
            cooperation='First arm releases on the table and parks; second arm regrips and places. No airborne handoff.',
            donor=getattr(self,'donor',None),recipient=getattr(self,'recipient',None))
        if self.child:result['stage_label']=f'{self.child.side or "Planning"} arm · {self.child.snapshot()["stage_label"]}'
        return result
