class TalosMicrophone extends AudioWorkletProcessor {
  process(inputs){
    const samples=inputs[0]?.[0];
    if(samples){const copy=new Float32Array(samples);this.port.postMessage(copy.buffer,[copy.buffer]);}
    return true;
  }
}
registerProcessor('talos-microphone',TalosMicrophone);
