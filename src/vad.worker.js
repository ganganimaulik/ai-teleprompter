import {
  AutoModel,
  Tensor,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';

const SAMPLE_RATE          = 16000;
const SPEECH_THRESHOLD     = 0.30;
const EXIT_THRESHOLD       = 0.08;
const MIN_SILENCE_MS       = 500;
const MIN_SILENCE_SAMPLES  = MIN_SILENCE_MS * (SAMPLE_RATE / 1000);
const MIN_SPEECH_SAMPLES   = 200 * (SAMPLE_RATE / 1000);
const SPEECH_PAD_SAMPLES   = 64  * (SAMPLE_RATE / 1000);
const MAX_BUFFER_DURATION  = 30;
const NEW_BUFFER_SIZE      = 512;
const MAX_NUM_PREV_BUFFERS = Math.ceil(SPEECH_PAD_SAMPLES / NEW_BUFFER_SIZE);

const EAGER_PARTIAL_MS      = 1000;
const EAGER_PARTIAL_SAMPLES = EAGER_PARTIAL_MS * (SAMPLE_RATE / 1000);
const MAX_PARTIAL_EMITS     = 4;

let silero_vad;
try {
  silero_vad = await AutoModel.from_pretrained('onnx-community/silero-vad', {
    config: { model_type: 'custom' }, dtype: 'fp32',
  });
} catch (err) {
  self.postMessage({ type: 'error', message: `Failed to load VAD: ${err.message}` });
  throw err;
}

self.postMessage({ type: 'status', status: 'vad_ready', message: 'VAD ready' });

const sr      = new Tensor('int64', [SAMPLE_RATE], []);
let vadState  = new Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);

const BUFFER  = new Float32Array(MAX_BUFFER_DURATION * SAMPLE_RATE);
let bufferPointer     = 0;
let isRecording       = false;
let postSpeechSamples = 0;
let prevBuffers       = [];
let _prevBufLen       = 0;
let partialEmitCount  = 0;
let partialEmitLast   = 0;

let vadChain = Promise.resolve();

const _INV_CHUNK_LEN          = 1 / NEW_BUFFER_SIZE;
const ENERGY_SILENCE_THRESHOLD = 1e-6;

function frameEnergy(buffer) {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return sum * _INV_CHUNK_LEN;
}

async function vad(buffer) {
  if (!isRecording && frameEnergy(buffer) < ENERGY_SILENCE_THRESHOLD) {
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) _prevBufLen -= prevBuffers.shift().length;
    prevBuffers.push(buffer);
    _prevBufLen += buffer.length;
    return false;
  }
  const input = new Tensor('float32', buffer, [1, buffer.length]);
  const { stateN, output } = await (vadChain = vadChain.then(() =>
    silero_vad({ input, sr, state: vadState }),
  ));
  vadState = stateN;
  const isSpeech = output.data[0];
  return isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD);
}

function maybeEmitPartial() {
  if (partialEmitCount >= MAX_PARTIAL_EMITS) return;
  const newSamples = bufferPointer - partialEmitLast;
  if (newSamples < EAGER_PARTIAL_SAMPLES) return;

  partialEmitLast = bufferPointer;
  partialEmitCount++;

  const segment = new Float32Array(_prevBufLen + bufferPointer);
  let off = 0;
  for (const b of prevBuffers) { segment.set(b, off); off += b.length; }
  segment.set(BUFFER.subarray(0, bufferPointer), off);

  self.postMessage({ type: 'segment', buffer: segment, isFinal: false, vadEmitTs: performance.now(), audioMs: Math.round(segment.length / SAMPLE_RATE * 1000) });
}

function reset(offset = 0) {
  BUFFER.fill(0, offset, Math.min(offset + SPEECH_PAD_SAMPLES + NEW_BUFFER_SIZE, BUFFER.length));
  bufferPointer     = offset;
  isRecording       = false;
  postSpeechSamples = 0;
  partialEmitCount  = 0;
  partialEmitLast   = 0;
}

function flushAndTranscribe(overflow) {
  const segment = new Float32Array(_prevBufLen + bufferPointer + SPEECH_PAD_SAMPLES);
  let off = 0;
  for (const b of prevBuffers) { segment.set(b, off); off += b.length; }
  segment.set(BUFFER.slice(0, bufferPointer + SPEECH_PAD_SAMPLES), off);

  self.postMessage({ type: 'segment', buffer: segment, isFinal: true, vadEmitTs: performance.now(), audioMs: Math.round(segment.length / SAMPLE_RATE * 1000) }, [segment.buffer]);

  prevBuffers = [];
  _prevBufLen = 0;
  if (overflow) BUFFER.set(overflow, 0);
  reset(overflow?.length ?? 0);
}

self.onmessage = async (event) => {
  const { buffer } = event.data;
  if (!buffer) return;

  const wasRecording = isRecording;
  const isSpeech     = await vad(buffer);

  if (!wasRecording && !isSpeech) {
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) _prevBufLen -= prevBuffers.shift().length;
    prevBuffers.push(buffer);
    _prevBufLen += buffer.length;
    return;
  }

  const remaining = BUFFER.length - bufferPointer;
  if (buffer.length >= remaining) {
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
    bufferPointer += remaining;
    flushAndTranscribe(buffer.subarray(remaining));
    return;
  } else {
    BUFFER.set(buffer, bufferPointer);
    bufferPointer += buffer.length;
  }

  if (isSpeech) {
    if (!isRecording) self.postMessage({ type: 'status', status: 'recording', message: 'Listening…' });
    isRecording = true;
    postSpeechSamples = 0;
    maybeEmitPartial();
    return;
  }

  postSpeechSamples += buffer.length;
  if (postSpeechSamples < MIN_SILENCE_SAMPLES) return;
  if (bufferPointer < MIN_SPEECH_SAMPLES) { reset(); return; }

  flushAndTranscribe();
};
