"""Preserve the completed stereo observer without changing the selected robot."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.mug_keypoint_experiment import protocol, read, sha, space, summarize, write
from scripts.evaluate_mug_keypoints import frozen_export


def check_rows(result, p):
    rows = result['rows']
    if [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Missing or reordered observations.')
    for row in rows:
        if row['scoring_only_present'] and row['observation']['status'] == 'observed':
            delta = np.asarray(row['observation']['midpoint_m'])-row['scoring_only_midpoint_m']
            if abs(np.linalg.norm(delta)*1000-row['error_3d_mm']) > 1e-10 or abs(np.linalg.norm(delta[:2])*1000-row['error_xy_mm']) > 1e-10:
                raise ValueError('An independently recomputed score disagrees.')
    if summarize(rows, p['perception_gate']) != result['summary']:
        raise ValueError('Aggregate scores disagree with complete individual outcomes.')


def package(args):
    p = protocol(args.protocol)
    raw = ROOT/p['raw_root']
    roots = {key: ROOT/p[key+'_package'] for key in ('model', 'training', 'evidence')}
    if any(root.exists() for root in roots.values()):
        raise FileExistsError('Preserve earlier packages.')
    selection, parity = frozen_export(p, args.protocol)
    fit = read(raw/'fit/training.json')
    if fit['completed_updates'] != p['training']['updates'] or [r['step'] for r in fit['candidates']] != p['training']['checkpoints']:
        raise ValueError('Both declared checkpoints and the complete fit are required.')
    sources = dict(fit['source_sha256'])
    payloads = {key: [] for key in roots}
    for split in ('training', 'development', 'evaluation'):
        folder = raw/split
        manifest, audit = read(folder/'manifest.json'), read(folder/'audit.json')
        if manifest['protocol_sha256'] != sha(args.protocol) or not audit['passed'] or audit['recipes_sha256'] != sha(folder/'recipes.npz'):
            raise ValueError('A complete independently audited split is required.')
        if manifest['states'] != p['data'][split+'_states'] or manifest['completed_states'] != manifest['states']:
            raise ValueError('Incomplete split.')
        for shard in manifest['shards']:
            if sha(folder/shard['file']) != shard['sha256']:
                raise ValueError('A raw shard changed before packaging.')
        sources.update(manifest['source_sha256'])
        for path in sorted(folder.iterdir()):
            if path.name in ('recipes.npz', 'manifest.json', 'audit.json') or path.suffix == '.png':
                payloads['training'].append((path, Path(split)/path.name))
    for candidate in fit['candidates']:
        folder = raw/f'fit/step-{candidate["step"]:06d}'
        if sha(folder/'model.safetensors') != candidate['checkpoint_sha256']:
            raise ValueError('A fitted checkpoint changed.')
        result = read(folder/'development.json')
        check_rows(result, p)
        if result['summary'] != candidate['development'] or len(result['rows']) != p['data']['development_states']:
            raise ValueError('A checkpoint score changed.')
        payloads['model'].append((folder/'model.safetensors', Path(folder.name)/'model.safetensors'))
        payloads['evidence'].append((folder/'development.json', Path(folder.name)/'development.json'))
    check_rows(parity['development'], p)
    fresh, physical = read(raw/'evaluation/perception.json'), read(raw/'physical-regression.json')
    for result, count in ((fresh, 512), (physical, 144)):
        check_rows(result, p)
        if len(result['rows']) != count or not result['summary']['gate_passed'] or result['checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise ValueError('This completed observer package requires its unchanged successful gates.')
        for name, digest in result['input_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('A scored input changed: '+name)
    for name in ('observer.xml', 'observer.bin'):
        payloads['model'].append((raw/'openvino'/name, Path('openvino')/name))
    payloads['model'].append((raw/'fit/selection.json', Path('selection.json')))
    for source, name in (('fit/training.json', 'training.json'), ('fit/selection.json', 'selection.json'),
                         ('openvino/parity.json', 'parity.json'), ('evaluation/perception.json', 'fresh-perception.json'),
                         ('physical-regression.json', 'physical-regression.json')):
        payloads['evidence'].append((raw/source, Path(name)))
    for path in sorted((raw/'audit-attempts').rglob('*')):
        if path.is_file():
            payloads['evidence'].append((path, path.relative_to(raw)))
    for path in sorted((ROOT/'.run/final-goal').glob('mug-keypoint-observer-v1*.log')):
        # Original console evidence is retained locally. Exclude a local path if
        # a library ever prints one; record exact provenance for either case.
        payloads['evidence'].append((path, Path('console')/path.name))
    helpers = [Path(__file__), ROOT/'scripts/reproduce_mug_keypoints.py', ROOT/'scripts/audit_mug_keypoint_data.py',
               ROOT/'scripts/export_mug_keypoints.py', ROOT/'scripts/evaluate_mug_keypoints.py']
    for helper in helpers:
        sources[helper.relative_to(ROOT).as_posix()] = sha(helper)
    for name, digest in sources.items():
        source = ROOT/name
        if sha(source) != digest:
            raise ValueError('A frozen dependency changed: '+name)
        if source.suffix == '.py':
            payloads['training'].append((source, Path('source')/name))
    for key in roots:
        payloads[key].append((args.protocol, Path('protocol.json')))
    expected = sum(source.stat().st_size for values in payloads.values() for source, _ in values)+8*1024**2
    preflight = space(p, roots['evidence'], expected)
    for root in roots.values():
        root.mkdir(parents=True)
    originals, console = {}, {}
    for key, values in payloads.items():
        for source, name in values:
            destination = roots[key]/name
            space(p, destination, source.stat().st_size+1024)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            if name.parts[0] == 'console':
                for prefix, replacement in ((str(ROOT), '<repository>'), (ROOT.as_posix(), '<repository>')):
                    data = data.replace(prefix.encode(), replacement.encode())
                console[name.as_posix()] = {'original_sha256': sha(source), 'original_bytes': source.stat().st_size,
                                           'repository_path_redacted': data != source.read_bytes()}
            with destination.open('xb') as stream:
                stream.write(data)
            originals[source.relative_to(ROOT).as_posix()] = {'package': destination.relative_to(ROOT).as_posix(), 'original_sha256': sha(source)}
    write(roots['training']/'reproduction-inputs.json', {'schema': p['schema'], 'source_sha256': sources,
        'artifact_mapping': originals, 'raw_rgb_shards_packaged': False,
        'instructions': 'All 5120 exact pose/exposure recipes, image points, visibility labels/counts and RGB hashes are packaged. Re-render with the frozen source and repository scene/assets. Raw RGB/segmentation shards remain local; they are regenerable. The reproduction script tests every fresh image and selected CPU prediction without reading .run.',
        'new_physical_trials': 0})
    write(roots['evidence']/'package-audit.json', {'schema': p['schema'], 'protocol_sha256': sha(args.protocol),
        'source_sha256': sha(Path(__file__)), 'passed': True, 'preflight': preflight,
        'counts': {'training': 4096, 'development': 512, 'fresh': 512, 'exposed_physical_states': 144},
        'both_checkpoints_retained': True, 'scores_independently_recomputed': True,
        'fresh': fresh['summary'], 'physical_regression': physical['summary'], 'console_provenance': console,
        'new_physical_trials': 0, 'motor_integration': False, 'selected_dinner_models_unchanged': True,
        'scope': 'Passing bounded perception only. A separate live-versus-frozen physical controller experiment is still required.'})
    counts = {}
    for key, root in roots.items():
        files = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob('*')) if path.is_file()}
        size = sum((root/name).stat().st_size for name in files)
        write(root/'manifest.json', {'schema': p['schema'], 'files': files, 'bytes': size})
        counts[key] = {'files': len(files), 'bytes': size}
    space(p, roots['evidence'], 0)
    print({'packages': counts, 'fresh': fresh['summary'], 'physical_regression': physical['summary']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-keypoint-observer-v1.json')
    package(parser.parse_args())
