"""One explicitly selected short audio test; never logs keys or token URLs."""
import argparse,asyncio,json,sys,time,wave
from pathlib import Path
from urllib.parse import quote
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.speech import temporary_token
from simulation_lab.language import parse_command

async def run(a):
    import websockets
    with wave.open(a.audio,'rb') as wav:
        if wav.getnchannels()!=1 or wav.getsampwidth()!=2:raise ValueError('Use mono 16-bit PCM WAV.')
        rate=wav.getframerate();audio=wav.readframes(wav.getnframes());duration=wav.getnframes()/rate
    if duration>20:raise ValueError('Test capped at twenty seconds of audio.')
    auth=await asyncio.to_thread(temporary_token);transcripts=[];start=time.perf_counter();sequence=0
    async with websockets.connect(auth['url']+'?jwt='+quote(auth['token']),open_timeout=20) as ws:
        await ws.send(json.dumps({'message':'StartRecognition','audio_format':{'type':'raw','encoding':'pcm_s16le','sample_rate':rate},
                                 'transcription_config':{'language':'en','model':'enhanced','max_delay':.7}}))
        while True:
            msg=json.loads(await asyncio.wait_for(ws.recv(),20))
            if msg.get('message')=='Error':raise RuntimeError('Speechmatics rejected the session configuration.')
            if msg.get('message')=='RecognitionStarted':break
        async def sender():
            nonlocal sequence
            for offset in range(0,len(audio),3200):
                await ws.send(audio[offset:offset+3200]);sequence+=1;await asyncio.sleep(3200/(2*rate))
            await ws.send(json.dumps({'message':'EndOfStream','last_seq_no':sequence}))
        sending=asyncio.create_task(sender())
        try:
            while True:
                msg=json.loads(await asyncio.wait_for(ws.recv(),25))
                if msg.get('message')=='AddTranscript':transcripts.append(msg.get('metadata',{}).get('transcript',''))
                if msg.get('message')=='Error':raise RuntimeError('Speechmatics returned a transcription error.')
                if msg.get('message')=='EndOfTranscript':break
            await sending
        finally:
            if not sending.done():sending.cancel()
    text=' '.join(transcripts).strip()
    report={'provider':'Speechmatics Realtime API','audio_source':a.label,'audio_seconds':duration,
            'wall_seconds':time.perf_counter()-start,'transcript':text,
            'robot_execution':False,'credential_or_token_logged':False}
    try:report['parsed_plan']=parse_command(text)
    except ValueError:report['parsed_plan']=None;report['parse_error']='Unsupported recognized instruction; no robot action executed.'
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audio',required=True);p.add_argument('--output',required=True)
    p.add_argument('--label',default='User-selected test audio');a=p.parse_args()
    try:asyncio.run(run(a))
    except Exception as exc:
        print('Speech test failed:',type(exc).__name__, '(details withheld to avoid exposing authenticated URLs)');raise SystemExit(1)
