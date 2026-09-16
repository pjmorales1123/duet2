"""Portable actual-hardware, OpenVINO and physical verification of a dinner suite.

No model is overwritten. Reports distinguish legacy Intel, Core Ultra and
non-Intel machines; successful CPU inference alone is not hardware eligibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space
from simulation_lab.hardware_evidence import all_inference_devices_are_intel
from scripts.inspect_intel_target import inspect
from scripts.export_primitive_openvino import run as export
from scripts.evaluate_learned_dinner import run as evaluate


def save(path,value):
    require_space(path,1024**2)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def run(args):
    if args.output.exists():raise FileExistsError('Choose a new report folder; old evidence is retained.')
    require_space(args.output,512*1024**2)
    paths=json.loads(args.suite.read_text(encoding='utf-8-sig'))
    skills=args.skills.split(',')
    sources={s:(args.suite.parent/paths[s]).resolve() for s in skills}
    if any(not (p/'primitive.safetensors').is_file() for p in sources.values()):
        raise ValueError('A requested skill is missing its checkpoint.')
    args.output.mkdir(parents=True)
    hardware=inspect();save(args.output/'hardware.json',hardware)
    available=[row['id'] for row in hardware['openvino_devices']]
    if args.device not in available:
        raise ValueError('Choose an exact enumerated OpenVINO device: '+', '.join(available)+'. Hardware evidence is retained; no export or physical trial started.')
    suite={};benchmarks={}
    for skill,source in sources.items():
        destination=args.output/skill
        export(SimpleNamespace(checkpoint=str(source),output=str(destination)))
        benchmark=json.loads((destination/'benchmark.json').read_text())
        result=next((r for r in benchmark['devices'] if r['device']==args.device and r.get('parity_passed')),None)
        if result is None:raise ValueError('Requested OpenVINO device failed parity for '+skill+'. Results are retained.')
        save(destination/'openvino.json',{'device':args.device,'precision':'f32'})
        suite[skill]=skill;benchmarks[skill]=result
    save(args.output/'suite.json',suite)
    returncode=evaluate(SimpleNamespace(suite=args.output/'suite.json',checkpoint=None,episode=None,
        seed=args.seed,skills=args.skills,output=args.output/'physical.json',capture=args.output/'physical-recording',disturbance_x_n=0.))
    intel_inference=all_inference_devices_are_intel(benchmarks)
    strict=hardware['strict_written_hardware_check'] and hardware['all_intel_cpu_and_graphics'] and intel_inference
    result={'hardware':hardware,'benchmarks':benchmarks,'physical_passed':returncode==0,
            'inference_devices_intel':intel_inference,
            'source_checkpoint_sha256':{s:hashlib.sha256((p/'primitive.safetensors').read_bytes()).hexdigest() for s,p in sources.items()},
            'strict_hardware_and_physics_passed':bool(strict and returncode==0),
            'note':'Legacy Intel eligibility depends on organizer clarification. Diagnostic mode never turns this flag true.'}
    save(args.output/'verification.json',result)
    print(json.dumps({'physical_passed':returncode==0,'strict_hardware_and_physics_passed':result['strict_hardware_and_physics_passed']}))
    return returncode if args.diagnostic else int(not result['strict_hardware_and_physics_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='CPU',help='Exact enumerated OpenVINO device, e.g. CPU or GPU.0. Must pass measured parity.')
    p.add_argument('--skills',default='bottle,plate,mug,drawer,fork,spoon')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--diagnostic',action='store_true',help='Return physical status on nonqualifying hardware, without changing reported eligibility.')
    raise SystemExit(run(p.parse_args()))
