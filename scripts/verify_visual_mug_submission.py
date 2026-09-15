"""Serial target-hardware verification of the promoted late-mug controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import openvino as ov
from simulation_lab.mug_visual_profile import DEFAULT_PROFILE, load_profile
from simulation_lab.storage import require_space
from scripts.inspect_intel_target import inspect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    payload = (json.dumps(value, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024)
    with path.open('xb') as stream:
        stream.write(payload)


def benchmark(path, inputs):
    core = ov.Core()
    started = time.perf_counter()
    compiled = core.compile_model(str(path), 'CPU',
        {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
    compilation_seconds = time.perf_counter()-started
    for _ in range(10):
        compiled(inputs)
    timings = []
    for _ in range(100):
        started = time.perf_counter()
        result = compiled(inputs)
        timings.append((time.perf_counter()-started)*1000)
        if any(not np.isfinite(result[port]).all() for port in range(len(compiled.outputs))):
            raise ValueError('A target-hardware benchmark produced nonfinite outputs.')
    return {'model_xml_sha256': sha(path), 'model_bin_sha256': sha(path.with_suffix('.bin')),
        'device': 'CPU', 'device_name': core.get_property('CPU', 'FULL_DEVICE_NAME'),
        'execution_devices': list(compiled.get_property('EXECUTION_DEVICES')),
        'precision': 'f32', 'inference_threads': 2, 'warmup_calls': 10, 'measured_calls': 100,
        'median_ms': float(np.median(timings)), 'p95_ms': float(np.percentile(timings, 95)),
        'synchronous_calls_per_second': 1000/float(np.mean(timings)), 'compilation_seconds': compilation_seconds,
        'input_shapes': [list(a.shape) for a in inputs],
        'scope': 'Network-only latency on fixed synthetic arrays. Excludes camera rendering, decoding, geometric guards and physics; no new accuracy or export-parity claim.'}


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a new verification folder; previous results are retained.')
    preflight = require_space(args.output, 256*1024**2)
    profile, protocol = load_profile()
    args.output.mkdir(parents=True)
    sources = [ROOT/'scripts/evaluate_visual_mug_deployment.py', *(ROOT/'simulation_lab').glob('*.py'),
               Path(__file__)]
    write(args.output/'protocol.json', {'schema': 'talos.visual-mug-target-verification.v1',
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        'profile_sha256': sha(DEFAULT_PROFILE), 'cases': [
            {'surface': 'engine', 'preset': preset, 'seed': 42} for preset in ('upright', 'wide_left')],
        'maximum_workers': 1, 'maximum_wall_seconds_per_trial': 420, 'preflight': preflight,
        'scope': 'Two exposed production-entry checks, run serially on this target. No training or model selection.'})
    hardware = inspect()
    write(args.output/'hardware.json', hardware)
    rng = np.random.default_rng(42)
    benchmarks = {
        'stereo_observer': benchmark(ROOT/protocol['observer']/'openvino/observer.xml',
            [rng.random((2, 3, 240, 320), dtype=np.float32)]),
        'local_motor': benchmark(ROOT/protocol['motor']/'openvino/motor.xml',
            [np.zeros((1, 5), np.float32), np.array([[.001, 0, 0]], np.float32)])}
    write(args.output/'benchmarks.json', benchmarks)
    cases = []
    for preset in ('upright', 'wide_left'):
        log, output = args.output/(preset+'.console.log'), args.output/('engine-'+preset)
        if log.exists() or output.exists():
            raise FileExistsError('Preserve per-trial output and console.')
        require_space(output, 64*1024**2)
        with log.open('xb') as stream:
            code = subprocess.run([sys.executable, '-I', str(ROOT/'scripts/evaluate_visual_mug_deployment.py'),
                '--output', str(args.output.resolve()), '--surface', 'engine', '--preset', preset],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        report = json.loads((output/'report.json').read_text(encoding='utf-8-sig'))
        row = {'preset': preset, 'exit_code': code, 'passed': report['passed'],
            'wall_seconds': report['wall_seconds'], 'trace_frames': report['trace_frames'],
            'correction_queries': report.get('correction_queries'), 'report_sha256': sha(output/'report.json')}
        cases.append(row)
        print(json.dumps(row), flush=True)
    physics = all(row['passed'] and row['exit_code'] == 0 for row in cases)
    intel_inference = hardware['intel_cpu'] and all('Intel' in b['device_name'] and b['execution_devices'] == ['CPU'] for b in benchmarks.values())
    strict = physics and intel_inference and hardware['strict_written_hardware_check'] and hardware['all_intel_cpu_and_graphics']
    result = {'schema': 'talos.visual-mug-target-verification.v1', 'profile': profile['name'],
        'profile_sha256': sha(DEFAULT_PROFILE), 'hardware': hardware, 'cases': cases,
        'physical_passed': physics, 'inference_devices_intel': intel_inference,
        'strict_hardware_and_physics_passed': bool(strict), 'benchmarks': benchmarks,
        'scope': 'This result applies to late-mug live vision with the selected dinner/relay models on two exposed starts. Legacy Intel cannot pass the strict Core Ultra flag; diagnostic mode changes the exit status only.'}
    write(args.output/'verification.json', result)
    print(json.dumps({k: result[k] for k in ('physical_passed', 'inference_devices_intel', 'strict_hardware_and_physics_passed')}), flush=True)
    return int(not (physics if args.diagnostic else strict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true', help='Return physical status while retaining the measured strict hardware result.')
    raise SystemExit(run(parser.parse_args()))
