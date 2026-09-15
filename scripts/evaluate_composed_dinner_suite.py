"""Run every reserved composed-command scene against a frozen selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space
from scripts.evaluate_composed_dinner import frozen_inputs


def run(args):
    protocol_path = Path('docs/robotics/experiments/composed-dinner-relay-v1.json')
    protocol = json.loads(protocol_path.read_text())
    selection = json.loads(args.freeze.read_text())
    current = frozen_inputs(protocol_path.resolve(), Path('models/dinner_suite/suite.json').resolve())
    if any(selection[key] != current[key] for key in current):
        raise ValueError('Inputs changed after the final freeze.')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if selection['batch_sha256'] != digest:
        raise ValueError('Batch driver changed after the final freeze.')
    seeds = protocol['evaluation_seeds']
    if selection['evaluation_seeds'] != seeds or not 1 <= args.workers <= 3:
        raise ValueError('Unexpected seeds or worker count.')
    if args.output.exists():
        raise FileExistsError(args.output)
    require_space(args.output, len(seeds) * 12 * 1024**2)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    rows = []

    def trial(seed):
        folder = args.output / str(seed)
        command = [sys.executable, 'scripts/evaluate_composed_dinner.py', '--seed', str(seed),
                   '--split', 'evaluation', '--freeze', str(args.freeze), '--output', str(folder)]
        with (args.output / (str(seed) + '.log')).open('w') as output:
            process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        report_path = folder / 'report.json'
        if not report_path.exists():
            return {'seed': seed, 'passed': False, 'status': 'execution_error', 'exit_code': process.returncode}
        report = json.loads(report_path.read_text())
        return {'seed': seed, 'passed': report['status'] == 'succeeded', 'status': report['status'],
                'completed_steps': report.get('task', {}).get('completed_steps'),
                'simulation_seconds': report['simulation_seconds'], 'wall_seconds': report['wall_seconds'],
                'report': str(seed) + '/report.json',
                'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job in as_completed([pool.submit(trial, seed) for seed in seeds]):
            row = job.result()
            rows.append(row)
            result = {'attempted': len(rows), 'planned': len(seeds), 'passed': sum(r['passed'] for r in rows),
                      'rows': sorted(rows, key=lambda r: r['seed']), 'frozen_inputs': current,
                      'wall_seconds': time.perf_counter()-began}
            require_space(args.output, 1024**2)
            (args.output / 'summary.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    run(parser.parse_args())
