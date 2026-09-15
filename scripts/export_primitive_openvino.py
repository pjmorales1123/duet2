"""Export the primitive network and measure parity/latency on available devices.

Hardware names are reported verbatim; an AMD CPU result is not Intel evidence.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,openvino as ov,torch
from safetensors.torch import load_file
from simulation_lab.primitive_policy import PrimitiveNet
from simulation_lab.storage import require_space


def run(a):
    source=Path(a.checkpoint);out=Path(a.output);require_space(out,256*1024**2)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(2);net=PrimitiveNet().eval();net.load_state_dict(load_file(str(source/'primitive.safetensors')))
    sample=(torch.zeros(20,32),torch.arange(20,dtype=torch.float32)/20)
    with torch.inference_mode():traced=torch.jit.trace(net,sample)
    converted=ov.convert_model(traced,example_input=sample)
    for port,name in zip(converted.inputs,['visual','seconds']):port.get_tensor().set_names({name})
    converted.reshape({'visual':[20,32],'seconds':[20]})
    out.mkdir(parents=True);ov.save_model(converted,out/'primitive.xml',compress_to_fp16=False)
    for name in ['primitive.safetensors','primitive.json','visual.npz','talos_normalization.json']:shutil.copy2(source/name,out/name)
    core=ov.Core();reports=[];rng=np.random.default_rng(42171)
    scale=np.array(json.loads((source/'primitive.json').read_text())['action_std'])
    for device in core.available_devices:
        options={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32'}
        if device=='CPU':options['INFERENCE_NUM_THREADS']=2
        report={'device':device,'name':core.get_property(device,'FULL_DEVICE_NAME')}
        try:
            compiled=core.compile_model(str(out/'primitive.xml'),device,options);errors=[];latencies=[]
            samples=[{'visual':rng.normal(size=(20,32)).astype('float32'),'seconds':rng.uniform(0,50,20).astype('float32')} for _ in range(32)]
            for inputs in samples:
                with torch.inference_mode():expected=net(torch.from_numpy(inputs['visual']),torch.from_numpy(inputs['seconds'])).numpy()
                actual=compiled(inputs)[0];errors.append(float(np.max(np.abs((actual-expected)*scale))))
            for _ in range(20):compiled(samples[0])
            benchmark_start=time.perf_counter()
            for i in range(300):
                start=time.perf_counter();compiled(samples[i%len(samples)]);latencies.append((time.perf_counter()-start)*1000)
            benchmark_seconds=time.perf_counter()-benchmark_start
            report.update(execution_devices=list(compiled.get_property('EXECUTION_DEVICES')),max_joint_error_rad=max(errors),
                          parity_passed=max(errors)<1e-4,median_ms=float(np.median(latencies)),p95_ms=float(np.percentile(latencies,95)),
                          synchronous_chunks_per_second=300/benchmark_seconds,output_endpoints_per_chunk=20,
                          precision_hint=str(compiled.get_property('INFERENCE_PRECISION_HINT')),warmup_calls=20,measured_calls=300)
        except Exception as exc:report.update(error_type=type(exc).__name__,supported=False)
        reports.append(report);print(json.dumps(report),flush=True)
    result={'openvino':ov.__version__,'source_sha256':hashlib.sha256((source/'primitive.safetensors').read_bytes()).hexdigest(),
            'scope':'Neural network only. CPU visual encoding, joint-feedback guard and MuJoCo/rendering excluded from latency.',
            'parity_inputs':'32 synthetic feature/time batches, not a physical success evaluation','devices':reports}
    (out/'benchmark.json').write_text(json.dumps(result,indent=2))
    if not any(r['device']=='CPU' and r.get('parity_passed') for r in reports):raise RuntimeError('CPU parity failed; runtime not enabled.')
    (out/'openvino.json').write_text(json.dumps({'device':'CPU','precision':'f32'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
