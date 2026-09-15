"""Bounded neural-skill plans grounded in explicit camera observations."""
from .language import CommandError
from .dinner_autonomy import SKILLS


def plan_learned_steps(plan, scene, available_skills=None):
    steps, reasons = [], []
    available=set(SKILLS if available_skills is None else available_skills)
    clear_bottle = scene.get('bottle_path_clear')
    clear_plate = scene.get('drawer_path_clear')
    drawer_open = scene.get('drawer_open')

    def add(skill, reason=None):
        nonlocal clear_bottle, clear_plate, drawer_open
        if skill=='bottle' and scene.get('bottle_region')=='left_reach':
            required=['reverse_bottle_right','reverse_bottle_left']
            if not set(required)<=available:raise CommandError('This observed bottle position needs the trained right-to-left relay, which is not loaded.')
            for leg in required:
                if leg not in steps:steps.append(leg)
            if not clear_bottle:reasons.append('Use the right arm to reach the bottle, release on the shared table, then let the left arm finish placement.')
            clear_bottle=True
            return
        if skill not in steps:
            steps.append(skill)
            if reason: reasons.append(reason)
        if skill == 'bottle': clear_bottle = True
        if skill == 'plate': clear_plate = True
        if skill == 'drawer': drawer_open = True

    def clear_plate_path():
        if clear_bottle is None: raise CommandError('Camera cannot verify that the bottle is clear of the plate path.')
        if not clear_bottle: add('bottle', 'Move the bottle first to clear the plate path.')

    def open_drawer():
        if drawer_open is None: raise CommandError('Camera cannot determine whether the drawer is open or closed.')
        if drawer_open: return
        if clear_plate is None: raise CommandError('Camera cannot verify that the plate is clear of the drawer path.')
        if not clear_plate:
            clear_plate_path(); add('plate', 'Move the plate first to clear the drawer path.')
        add('drawer', 'Open the drawer before retrieving cutlery.')

    for intent in plan['steps']:
        if intent['kind'] == 'set_table':
            for skill in SKILLS:
                if skill != 'drawer' or not drawer_open: add(skill)
            continue
        skill = 'drawer' if intent['kind'] == 'drawer_open' else intent.get('object_id')
        destination=intent.get('destination') or {}
        if destination.get('kind')=='handoff':
            if skill!='bottle' or len(plan['steps'])!=1:raise CommandError('The trained bottle relay must be requested as one complete task.')
            recipient=destination['recipient'];starter='left' if recipient=='right' else 'right'
            required=['relay_bottle_left','relay_bottle_right'] if recipient=='right' else ['reverse_bottle_right','reverse_bottle_left']
            if intent.get('arm','auto') not in ('auto',starter):raise CommandError('This relay starts with the '+starter+' arm.')
            if not set(required)<=available:raise CommandError('The requested neural relay checkpoints are not loaded.')
            steps.extend(required);reasons.append('Release on the shared table and park before the other arm regrasp. This is a table-supported relay.')
            continue
        if intent['kind'] not in ('drawer_open', 'dinner_place') or skill not in SKILLS:
            raise CommandError('This learned suite supports bottle, plate, mug, drawer, fork and spoon.')
        expected_arm = 'right' if skill == 'mug' else 'left'
        if skill=='bottle' and scene.get('bottle_region')=='left_reach' and intent.get('arm','auto')!='auto':
            raise CommandError('This bottle position needs both arms; use an automatic bottle-placement instruction.')
        if intent.get('arm', 'auto') not in ('auto', expected_arm):
            raise CommandError('The trained '+skill+' skill uses the '+expected_arm+' arm.')
        if (intent.get('destination') or {}).get('kind', 'default') != 'default':
            raise CommandError('This learned suite uses the trained table settings; that destination or handoff is not trained yet.')
        if skill == 'plate': clear_plate_path()
        if skill in ('fork', 'spoon', 'drawer'): open_drawer()
        if skill != 'drawer': add(skill)
    if not steps: raise CommandError('The requested drawer is already open.')
    if not set(steps)<=available:raise CommandError('A requested neural skill is not installed.')
    return {'steps': steps, 'reasons': reasons, 'observation': dict(scene),
            'instruction': plan['instruction'], 'interpreter': 'bounded grammar and RGB-grounded dependencies'}
