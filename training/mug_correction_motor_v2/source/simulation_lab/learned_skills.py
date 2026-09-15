"""Explicit trained skill identities; no arbitrary destinations are inferred."""
from .dinner_autonomy import SKILLS

RELAY_SKILLS=('relay_bottle_left','relay_bottle_right','reverse_bottle_right','reverse_bottle_left')
ALL_LEARNED_SKILLS=(*SKILLS,*RELAY_SKILLS)
SKILL_LABELS={s:s for s in SKILLS}
SKILL_LABELS.update(relay_bottle_left='left arm to shared table',relay_bottle_right='right arm to serving spot',
                    reverse_bottle_right='right arm to shared table',reverse_bottle_left='left arm to bottle setting')


def object_for_skill(skill):
    if skill not in ALL_LEARNED_SKILLS:raise ValueError('Unknown learned skill.')
    return 'bottle' if skill in RELAY_SKILLS else skill
