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
self.postMessage({ type: 'status', status: 'loading', message: 'Loading speech model…' });

const DTYPE_CONFIGS = {
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  wasm:   { encoder_model: 'fp32', decoder_model_merged: 'q8' },
};

let transcriber;
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
} catch (err) {
  self.postMessage({ type: 'error', message: `Failed to load Moonshine: ${err.message}` });
  throw err;
}

await transcriber(new Float32Array(16000));
self.postMessage({ type: 'status', status: 'ready', message: 'Ready' });

let txChain = Promise.resolve();
let _latestPartial = null;

async function transcribeAndEmit(buffer, isFinal, vadEmitTs, audioMs) {
  const txStartTs = performance.now();
  self.postMessage({ type: 'status', status: 'transcribing', message: 'Transcribing…' });
  const { text } = await transcriber(buffer);
  const txEndTs = performance.now();
  const cleaned = text.trim();
  if (cleaned) self.postMessage({ type: 'transcript', text: cleaned, isFinal, vadEmitTs, audioMs, txStartTs, txEndTs, txDurMs: Math.round(txEndTs - txStartTs) });
  self.postMessage({ type: 'status', status: 'recording', message: 'Listening…' });
}

self.onmessage = ({ data }) => {
  const { type, buffer, isFinal, vadEmitTs, audioMs } = data;
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
