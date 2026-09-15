"""Load a physically promoted mug profile; preserve the original dinner mode."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PROFILE=ROOT/'models/dinner_visual_mug_v1/profile.json'


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact(name):
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('A visual mug artifact leaves the repository.')
    return path


def load_profile(path=None):
    path=Path(path or DEFAULT_PROFILE)
    try:
        profile=_read(path)
        if profile['schema']!='talos.promoted-mug-vision.v1':raise ValueError('Unknown visual mug profile.')
        for name,digest in profile['evidence_files'].items():
            if _sha(path.parent/name)!=digest:raise ValueError('Visual mug promotion evidence changed.')
        for split in ('development','evaluation'):
            report=_read(path.parent/(split+'-audit.json'))
            if not report['gate_passed'] or not all(report['requirements'].values()):
                raise ValueError('Visual mug correction requires both physical promotion gates.')
        protocol=_read(path.parent/'protocol.json')
        if _sha(path.parent/'protocol.json')!=profile['protocol_sha256']:
            raise ValueError('The promoted correction settings changed.')
        for name,digest in {**profile['model_files'],**profile['runtime_sources']}.items():
            if _sha(_artifact(name))!=digest:raise ValueError('A verified visual mug dependency changed: '+name)
        if protocol['baseline_suite_sha256']!=_sha(_artifact(protocol['baseline_suite'])):
            raise ValueError('The verified dinner suite changed.')
    except (OSError,KeyError,json.JSONDecodeError) as exc:
        raise ValueError('The verified visual mug profile is unavailable or incomplete.') from None
    return profile,protocol


def make_sequence(model,data,layout,checkpoints,steps,profile_path=None):
    profile,protocol=load_profile(profile_path)
    verified_plans=[['bottle','plate','mug','drawer','fork','spoon'],
        ['reverse_bottle_right','reverse_bottle_left','plate','mug','drawer','fork','spoon']]
    if 'mug' in steps and list(steps) not in verified_plans:
        raise ValueError('Live mug vision is verified for full table setting. Use “set the table” or choose the original dinner controller.')
    suite=_artifact(protocol['baseline_suite']);expected=_read(suite)
    if any(skill not in expected or Path(checkpoints[skill]).resolve()!=(suite.parent/expected[skill]).resolve() for skill in steps):
        raise ValueError('Visual mug correction requires the verified dinner models.')
    from .mug_visual_control import VisualMugSequence

    class PromotedMugSequence(VisualMugSequence):
        def snapshot(self):
            result=super().snapshot()
            result.update(visual_feedback_profile=profile['name'],
                visual_feedback_scope='late_mug_placement' if 'mug' in self.steps else 'no_mug_in_plan')
            return result

    return PromotedMugSequence(model,data,layout,checkpoints,steps,protocol,'live')
