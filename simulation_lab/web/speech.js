// Speechmatics real-time microphone input. Long-lived credentials never leave Python.
export async function initSpeech({button,feedback,onTranscript}){
  let active=null;
  const finish=()=>{if(active)active.stop();};
  async function updateAvailability(){
    try{const s=await(await fetch('/api/speech/status')).json();button.disabled=!s.configured;button.textContent=s.configured?'Speak instruction':'Voice key not configured';}
    catch{button.disabled=true;button.textContent='Voice unavailable';}
  }
  await updateAvailability();
  button.onclick=async()=>{
    if(active){finish();return;}
    let mic,context,socket,worklet,timer;let stopping=false,seq=0,finals=[],settled=false;
    const cleanup=()=>{
      clearTimeout(timer);worklet?.disconnect();mic?.getTracks().forEach(t=>t.stop());context?.close();
      if(socket&&socket.readyState<2)socket.close();active=null;button.textContent='Speak instruction';button.disabled=false;
    };
    const stop=()=>{
      if(stopping)return;stopping=true;worklet?.disconnect();mic?.getTracks().forEach(t=>t.stop());
      feedback.textContent='Finishing transcription…';
      if(socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify({message:'EndOfStream',last_seq_no:seq}));
      clearTimeout(timer);timer=setTimeout(()=>{feedback.textContent='Speech recognition timed out. Please use text or try again.';cleanup();},15000);
    };
    try{
      button.disabled=true;feedback.textContent='Microphone audio is sent to Speechmatics while recording (maximum 20 seconds).';
      mic=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true},video:false});
      const response=await fetch('/api/speech/token',{method:'POST'});const auth=await response.json();
      if(!response.ok)throw Error(auth.detail||'Voice authentication failed.');
      context=new AudioContext();await context.audioWorklet.addModule('/static/speech-worklet.js');
      await context.resume();
      socket=new WebSocket(auth.url+'?jwt='+encodeURIComponent(auth.token));
      socket.onopen=()=>socket.send(JSON.stringify({message:'StartRecognition',audio_format:{type:'raw',encoding:'pcm_f32le',sample_rate:context.sampleRate},
        transcription_config:{language:'en',model:'enhanced',enable_partials:true,max_delay:.7}}));
      socket.onmessage=async event=>{
        const data=JSON.parse(event.data);
        if(data.message==='RecognitionStarted'){
          clearTimeout(timer);
          const source=context.createMediaStreamSource(mic);worklet=new AudioWorkletNode(context,'duet2-microphone');
          worklet.port.onmessage=event=>{if(!stopping&&socket.readyState===WebSocket.OPEN){socket.send(event.data);seq++;}};
          source.connect(worklet);const silent=context.createGain();silent.gain.value=0;worklet.connect(silent).connect(context.destination);
          active={stop,cancel:cleanup};button.disabled=false;button.textContent='Finish recording';feedback.textContent='Listening…';
          timer=setTimeout(stop,auth.maximum_recording_seconds*1000);
        }else if(data.message==='AddPartialTranscript')feedback.textContent=finals.join(' ')+' '+(data.metadata?.transcript||'');
        else if(data.message==='AddTranscript'){finals.push(data.metadata?.transcript||'');feedback.textContent=finals.join(' ');}
        else if(data.message==='EndOfTranscript'){
          settled=true;const text=finals.join(' ').trim();cleanup();
          if(text){feedback.textContent='Speechmatics: '+text;await onTranscript(text);}
          else feedback.textContent='No speech was recognized. Please try again.';
        }else if(data.message==='Error'){settled=true;feedback.textContent='Speechmatics could not transcribe this recording. Use typed input or try again.';cleanup();}
      };
      socket.onerror=()=>{feedback.textContent='Speech service connection failed.';cleanup();};
      socket.onclose=()=>{if(!settled&&active){feedback.textContent='Speech connection closed before completion.';cleanup();}};
      timer=setTimeout(()=>{if(!active){feedback.textContent='Speech service did not start. Try again.';cleanup();}},15000);
    }catch(error){feedback.textContent=error.message;cleanup();}
  };
  window.addEventListener('pagehide',()=>{if(active)active.cancel();});
}
