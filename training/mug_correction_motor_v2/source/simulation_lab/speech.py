"""Speechmatics server-only credential loading and short-lived browser tokens."""
import json,os
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

ROOT=Path(__file__).resolve().parents[1]

def api_key():
    value=os.environ.get('SPEECHMATICS_API_KEY','').strip()
    if value:return value
    path=ROOT/'.env'
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.lstrip().startswith('#') or '=' not in line:continue
            name,value=line.split('=',1)
            if name.strip()=='SPEECHMATICS_API_KEY':
                value=value.strip()
                if len(value)>1 and value[0]==value[-1] and value[0] in ('\"',"'"):value=value[1:-1]
                return value.strip()
    return ''

def status():
    return {'configured':bool(api_key()),'provider':'Speechmatics','mode':'realtime',
            'maximum_recording_seconds':20,'credentials':'server-only; browser receives a 60-second temporary token'}

def temporary_token():
    key=api_key()
    if not key:raise ValueError('Add SPEECHMATICS_API_KEY to the local .env file to enable voice.')
    request=Request('https://mp.speechmatics.com/v1/api_keys?type=rt',data=json.dumps({'ttl':60}).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urlopen(request,timeout=20) as response:payload=json.loads(response.read(65536))
    except HTTPError as exc:
        # Never echo response bodies or authenticated requests to logs/the browser.
        raise RuntimeError('Speechmatics authentication failed (HTTP '+str(exc.code)+'). Check key permissions and credits.') from None
    except (URLError,TimeoutError):raise RuntimeError('Could not reach Speechmatics. Try again or use typed commands.') from None
    token=payload.get('key_value')
    if not isinstance(token,str) or not token:raise RuntimeError('Speechmatics did not return a temporary token.')
    return {'token':token,'url':'wss://eu.rt.speechmatics.com/v2','expires_in_seconds':60,'maximum_recording_seconds':20}
