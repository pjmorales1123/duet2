"""Build an explicit, credential-free Gradio Space upload folder; never upload."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def modules_for(entry):
    pending, found = [entry, 'learned_skills', 'hosted_worker', 'hosted_gpu'], {'__init__'}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / 'simulation_lab' / (name + '.py')
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                pending.append(node.module)
    return sorted(found)


def build(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Use a new Space build folder.')
    mapping = {f'simulation_lab/{name}.py': ROOT / f'simulation_lab/{name}.py' for name in modules_for('public_trial')}
    # Asset provenance and meshes come only from tracked SO-101 files.
    assets = subprocess.check_output(['git', 'ls-files', '--', 'simulation_lab/assets/so101'], cwd=ROOT, text=True).splitlines()
    for name in assets:
        mapping[name] = ROOT / name
    for name in ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'simulation_lab/NOTICE.md', 'hosting/gradio_app.py','hosting/gpu_inference.py']:
        mapping[name] = ROOT / name
    mapping['requirements.txt'] = ROOT / 'hosting/requirements.txt'
    mapping['packages.txt'] = ROOT / 'hosting/packages.txt'
    model_files = {'primitive.json','primitive.safetensors','primitive.xml','primitive.bin','visual.npz','talos_normalization.json','openvino.json','benchmark.json','README.md','suite.json'}
    for folder in ['bottle_visual', 'bottle_relays', 'dinner_suite']:
        base = ROOT / 'models' / folder
        for path in base.rglob('*'):
            if path.is_file() and path.name in model_files:
                mapping[path.relative_to(ROOT).as_posix()] = path
    visual_profile=ROOT/'models/dinner_visual_mug_v1/profile.json'
    if visual_profile.is_file():
        from simulation_lab.mug_visual_profile import load_profile
        profile,_=load_profile(visual_profile)
        for name in profile['model_files']:
            if not name.startswith(('models/','docs/robotics/experiments/')) or '..' in Path(name).parts:
                raise ValueError('Unexpected visual mug artifact path.')
            mapping[name]=ROOT/name
        for name in profile['runtime_sources']:
            if not name.startswith('simulation_lab/') or Path(name).suffix!='.py' or '..' in Path(name).parts:
                raise ValueError('Unexpected visual mug runtime dependency.')
            mapping[name]=ROOT/name
        for path in visual_profile.parent.iterdir():
            if path.is_file() and path.suffix in ('.json','.md'):
                mapping[path.relative_to(ROOT).as_posix()]=path
    expected = sum(p.stat().st_size for p in mapping.values()) + 1024**2
    space = require_space(output, expected)
    output.mkdir(parents=True)
    for name, source in sorted(mapping.items()):
        require_space(output, expected)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    readme = '''---
title: Talos Dinner Robotics
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: 3.12
app_file: hosting/gradio_app.py
pinned: false
license: mit
---

# Talos · Dinner robotics

Fresh, isolated MuJoCo trials with dual SO-101 arms. This compact interface offers
learned or programmed control, a scene seed, a bottle region and one of six camera
views. Its image is rendered from actual running physics. Outcomes include
failures and explicit refusals; each trial starts a new scene.

Learned models use initial RGB observations and motor-feedback progress guards.
They do not provide continuous visual correction or arbitrary workspace coverage.
The programmed mode uses exact simulator state. Bottle relays release onto the
table between arms; they are not airborne handoffs.

This deployment contains no Speechmatics credentials or microphone service. Use
the complete local application for voice and six simultaneous camera streams.

Source, models, measured results and [setup instructions](https://github.com/jannissio/talos-ai-infra/blob/main/docs/SETUP.md) are available in the [Talos repository](https://github.com/jannissio/talos-ai-infra).

Measured scope: the original dinner suite passed 8/10 frozen workflows. The late-mug comparison passed 20/20 live, 20/20 original baseline and 0/20 frozen-image workflows. This is bounded feedback, not general workspace coverage.

Use the OpenVINO CPU option to avoid shared-GPU allocation. A full dinner can take about five minutes. No login is needed for the CPU trial.

## Licenses and attribution

**MIT covers original Talos code, model weights and procedural dinner assets.** Third-party components retain their own licenses. The [MIT license](https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics/blob/main/LICENSE) does not replace the SO-101 assets' [Apache-2.0 license](https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics/blob/main/simulation_lab/assets/so101/LICENSE) or their [attribution and modification notices](https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics/blob/main/simulation_lab/NOTICE.md).

See [third-party notices and the dependency-license table](https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics/blob/main/THIRD_PARTY_NOTICES.md) for upstream terms, including Apache-2.0, BSD and MIT-CMU. The full notices file is included in this bundle.
'''
    if visual_profile.is_file():
        readme=readme.replace('Learned models use initial RGB observations and motor-feedback progress guards.\nThey do not provide continuous visual correction or arbitrary workspace coverage.',
            'The live mug vision option adds stereo visual correction during late mug placement,\nverified within complete table-setting workflows. Other skills retain initial RGB\nobservations and motor-feedback guards. Arbitrary workspace coverage remains unfinished.')
    (output / 'README.md').write_text(readme, encoding='utf-8', newline='\n')
    (output / '.gitignore').write_text('__pycache__/\n*.py[cod]\n.env\n.env.*\n*.log\n', encoding='utf-8')
    (output / '.gitattributes').write_text('* -text\n'+''.join(f'*.{extension} filter=lfs diff=lfs merge=lfs -text\n'
                                               for extension in ['bin','safetensors','stl','npz']), encoding='utf-8')
    hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file()}
    report = {'files':len(hashes),'bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
              'storage_preflight':space,'sha256':hashes,
              'privacy':'Explicit allowlist: no credentials, local recordings or development scratch files. Upload only after review, with nonpersonal Git metadata.'}
    (output / 'build-manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files':report['files'],'bytes':report['bytes'],'storage_preflight':space}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
