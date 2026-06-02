import {
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';

async function supportsWebGPU() {
  if (!('gpu' in navigator)) return false;
  try { const a = await navigator.gpu.requestAdapter(); return a !== null; }
  catch { return false; }
}

const device = (await supportsWebGPU()) ? 'webgpu' : 'wasm';
self.postMessage({ type: 'info', message: `Device: ${device}` });

const DTYPE_CONFIGS = {
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  wasm:   { encoder_model: 'fp32', decoder_model_merged: 'q8' },
};

let transcriber = null;
let localModelLoading = false;
let localModelReady = false;
let engine = 'local';
let modalUrl = '';

async function ensureLocalTranscriber() {
  if (transcriber) return;
  if (localModelLoading) {
    while (localModelLoading) {
      await new Promise(r => setTimeout(r, 100));
    }
    return;
  }
  localModelLoading = true;
  self.postMessage({ type: 'status', status: 'loading', message: 'Loading local model…' });
  
  try {
    transcriber = await pipeline(
      'automatic-speech-recognition',
      'onnx-community/moonshine-tiny-ONNX',
      {
        device, dtype: DTYPE_CONFIGS[device],
        progress_callback: (progress) => {
          if (progress.status === 'progress' && progress.total) {
            self.postMessage({ type: 'progress', percent: Math.round((progress.loaded / progress.total) * 100), file: progress.file });
          } else if (progress.status === 'done') {
            self.postMessage({ type: 'progress', percent: 100, file: progress.file });
          }
        },
      },
    );
    await transcriber(new Float32Array(16000)); // warm up
    localModelReady = true;
    self.postMessage({ type: 'status', status: 'ready', message: 'Ready' });
  } catch (err) {
    self.postMessage({ type: 'error', message: `Failed to load Moonshine: ${err.message}` });
    localModelLoading = false;
    throw err;
  } finally {
    localModelLoading = false;
  }
}

// Helper to convert 16kHz float32 audio to a 16-bit PCM WAV Blob
function encodeWAV(samples) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true);
  view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

let txChain = Promise.resolve();
let _latestPartial = null;

async function transcribeAndEmit(buffer, isFinal, vadEmitTs, audioMs) {
  const txStartTs = performance.now();
  self.postMessage({ type: 'status', status: 'transcribing', message: 'Transcribing…' });
  
  let text = '';
  
  if (engine === 'local') {
    try {
      await ensureLocalTranscriber();
      const result = await transcriber(buffer);
      text = result.text;
    } catch (err) {
      console.error('[TX worker] Local ONNX error:', err);
      self.postMessage({ type: 'error', message: `Local transcription error: ${err.message}` });
    }
  } else if (engine === 'modal') {
    if (!modalUrl) {
      self.postMessage({ type: 'error', message: 'Modal Endpoint URL is empty. Please enter a valid URL in configurations.' });
      self.postMessage({ type: 'status', status: 'ready', message: 'Ready' });
      return;
    }
    
    try {
      const wavBlob = encodeWAV(buffer);
      const formData = new FormData();
      formData.append('file', wavBlob, 'audio.wav');
      
      const response = await fetch(modalUrl, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }
      
      const data = await response.json();
      text = data.text || '';
    } catch (err) {
      console.error('[TX worker] Modal API error:', err);
      self.postMessage({ type: 'error', message: `Modal API error: ${err.message}` });
    }
  }

  const txEndTs = performance.now();
  const cleaned = text.trim();
  if (cleaned) {
    self.postMessage({ type: 'transcript', text: cleaned, isFinal, vadEmitTs, audioMs, txStartTs, txEndTs, txDurMs: Math.round(txEndTs - txStartTs) });
  }
  self.postMessage({ type: 'status', status: 'recording', message: 'Listening…' });
}

self.onmessage = ({ data }) => {
  const { type, buffer, isFinal, vadEmitTs, audioMs } = data;
  
  if (type === 'configure') {
    engine = data.engine;
    modalUrl = data.modalUrl;
    
    if (engine === 'local') {
      ensureLocalTranscriber().catch(err => console.error("Error loading local model:", err));
    } else {
      // Modal is ready immediately as it runs on server
      self.postMessage({ type: 'status', status: 'ready', message: 'Ready' });
    }
    return;
  }
  
  if (type !== 'segment' || !buffer) return;

  if (isFinal) {
    txChain = txChain.then(() => transcribeAndEmit(buffer, true, vadEmitTs, audioMs).catch(
      err => console.error('[TX worker] Transcription error:', err)));
  } else {
    _latestPartial = { buffer, vadEmitTs, audioMs };
    txChain = txChain.then(() => {
      const p = _latestPartial;
      if (!p) return;
      _latestPartial = null;
      return transcribeAndEmit(p.buffer, false, p.vadEmitTs, p.audioMs).catch(() => {});
    });
  }
};
