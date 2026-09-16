"""Bounded table-setting language grammar. This module is not an LLM.

It produces typed intents only; grounding/reachability and physical execution are
separate. Unsupported language never silently falls back to another movement.
"""
from dataclasses import dataclass,asdict
import re

LANGUAGE_SCHEMA = "duet-2.language.v1"

class CommandError(ValueError):pass

@dataclass
class Intent:
    kind:str
    object_id:str|None=None
    arm:str='auto'
    destination:dict|None=None

ALIASES={'water bottle':'bottle','pottle':'bottle','flasche':'bottle','teller':'plate','cup':'mug',
         'tasse':'mug','gabel':'fork','löffel':'spoon','loeffel':'spoon','schublade':'drawer',
         'side plate':'side_plate','gold plate':'side_plate','blue plate':'plate'}
ITEMS=('side_plate','bottle','plate','mug','glass','fork','spoon')

def parse_command(text,last_object=None):
    if not isinstance(text,str) or not 1<=len(text.strip())<=500:
        raise CommandError('Use a short table-setting instruction (1–500 characters).')
    original=text.strip();s=re.sub(r'\s+',' ',original.lower()).strip(' .!?')
    s=re.sub(r'\s+([,.!?;])',r'\1',s)
    for phrase,value in sorted(ALIASES.items(),key=lambda r:-len(r[0])):
        s=re.sub(r'\b'+re.escape(phrase)+r'\b',value,s)
    # ASR can place a sentence boundary between a verb and its object. Only
    # join this unambiguous fragment; never replace a misrecognized object word.
    s=re.sub(r'\b(put|place|move|take|bring|open)[.!?]\s+(?=(?:the )?(?:'+ '|'.join((*ITEMS,'drawer'))+r')\b)',r'\1 ',s)
    if re.search(r"\b(don't|do not|never|nicht)\b",s):
        raise CommandError('Negated movement instructions are not executed. Say stop to cancel.')
    if s in ('stop','cancel','stop moving','cancel the task','halt','stopp'):
        return {'instruction':original,'interpreter':'constrained_grammar','control':'cancel','steps':[]}
    if s in ('set the table','set up the table','set the dinner table','arrange the table','deck den tisch'):
        return {'instruction':original,'interpreter':'constrained_grammar','steps':[asdict(Intent('set_table'))]}
    if s in ('put water','pour water','pour the water','pour some water','give me water'):
        return {'instruction':original,'interpreter':'constrained_grammar','steps':[asdict(Intent('pour_water'))]}
    clauses=re.split(r'\s*(?:;|,?\s+(?:and then|then)|\.\s+)\s*',s)
    if len(clauses)>8:raise CommandError('Use at most eight task steps.')
    intents=[];context=last_object
    for clause in clauses:
        clause=re.sub(r'^(?:and then|then)[,\s]+','',clause).strip(' ,.!?')
        if not clause:continue
        if re.fullmatch(r'(?:please )?open (?:the )?(?:top )?drawer',clause):
            intents.append(Intent('drawer_open'));continue
        if not re.match(r'^(?:please )?(?:put|place|move|take|pick up|transfer|hand|pass|bring)\b',clause):
            raise CommandError('Try “place the plate”, “open the drawer”, or “put the bottle in front of the right plate”.')
        found=re.search(r'\b('+'|'.join(ITEMS)+r')\b',clause)
        item=found.group(1) if found else context if re.search(r'\bit\b',clause) else None
        if item is None:raise CommandError('Name the item to move.')
        tail=clause[found.end():] if found else clause
        if re.search(r'\b(under|underneath|over|above|inside)\b',tail):
            raise CommandError('That spatial relation is not supported.')
        arm_match=re.search(r'\b(?:with|using) (?:the )?(left|right) arm\b',tail)
        arm=arm_match.group(1) if arm_match else 'auto'
        if arm_match:tail=tail[:arm_match.start()]+tail[arm_match.end():]
        destination={'kind':'default'}
        relative=re.search(r'\b(in front of|infront of|behind|to the left of|to the right of|next to) (?:the )?(?:(left|right) )?('+'|'.join(ITEMS)+r')\b',tail)
        if relative:
            relation={'in front of':'front','infront of':'front','behind':'behind','to the left of':'left','to the right of':'right','next to':'beside'}[relative.group(1)]
            destination={'kind':'relative','relation':relation,'reference':relative.group(3),'reference_side':relative.group(2)}
            if destination['reference']==item and not destination['reference_side']:
                raise CommandError('The destination must refer to another item.')
        elif spot:=re.search(r'\b(?:(far) )?(left|right|center|middle) (?:spot|place|setting|side)\b',tail):
            destination={'kind':'spot','side':'center' if spot.group(2)=='middle' else spot.group(2),'far':bool(spot.group(1))}
        elif re.search(r'\b(?:to|in|on|into|under|over)\b',tail) and not re.search(r'\b(?:its|the) (?:spot|place|setting|table)\b|\bout of (?:the )?drawer\b',tail) and not (re.search(r'\b(hand|pass|transfer)\b',clause) and re.search(r'\b(left|right) arm\b',tail)):
            raise CommandError('I cannot resolve that destination. Use a left/right spot or a relation to a named item.')
        if re.search(r'\b(hand|pass|transfer)\b',clause) and re.search(r'\b(left|right) arm\b',tail):
            recipient=re.search(r'\b(left|right) arm\b',tail).group(1)
            destination={'kind':'handoff','recipient':recipient}
        # Drawer retrieval is one intent; the executor inserts opening only if needed.
        if item in ('fork','spoon') and re.search(r'\bdrawer\b',clause):destination['from_drawer']=True
        # Do not execute a second object hidden in a compound clause.
        if re.search(r'\band\s+(?:put|place|move|take|pick|transfer)\b',tail):
            # “take X out of the drawer and put it ...” describes one move.
            if not re.search(r'\band\s+(?:put|place)\s+it\b',tail):
                raise CommandError('Separate different movements with “then”.')
        intents.append(Intent('dinner_place',item,arm,destination));context=item
    if not intents:raise CommandError('No supported action found.')
    return {'instruction':original,'interpreter':'constrained_grammar','steps':[asdict(i) for i in intents],'last_object':context}


def instruction_metadata(parse_result: dict) -> dict:
    """Return a stable, schema-versioned language block for inclusion in every manifest.

    This grammar is bounded and deterministic.  It is not an LLM or VLA.
    Every field is safe to persist alongside physics trajectories.
    """
    return {
        'schema': LANGUAGE_SCHEMA,
        'instruction': parse_result.get('instruction'),
        'interpreter': parse_result.get('interpreter', 'constrained_grammar'),
        'steps': parse_result.get('steps', []),
        'is_llm': False,
        'is_vla': False,
    }
