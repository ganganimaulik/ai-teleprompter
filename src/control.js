import { doubleMetaphone } from 'https://cdn.jsdelivr.net/npm/double-metaphone/+esm';

'use strict';

const SAMPLE_SCRIPT = `Welcome to this AI-powered desktop teleprompter.

As you speak, the script will automatically scroll to keep up with your words. Each word you say is highlighted in real time, so you always know exactly where you are in your presentation.

This teleprompter uses an embedded AI voice model that runs entirely on your device. No data leaves your machine, and no internet connection is required after the first load.

To get started, click Start Tracking. The microphone will listen to your voice and track your progress through the script automatically.

You can adjust the font size and speaking rate using the controls. Mirror mode flips the display horizontally for physical teleprompter glass rigs.

Thank you for using this app. Good luck with your speech!`;

const ASR_OVERRIDES = {
  'a':   'hw_art', 'the': 'hw_art',
  'an':  'hw_and', 'and': 'hw_and',
};

// ═══════════════════════════════════════════════════════
// STATE OBJECT
// ═══════════════════════════════════════════════════════
const state = {
  paragraphs: [],
  words: [],
  wordCount: 0,
  scriptNormTokens: [],  
  tokenIndex: null,      

  currentWordIndex: 0,
  currentParaIndex: 0,
  paragraphCompleteTimer: null,
  sessionStartTime: null,

  accumulatedText: '',
  lastChunk: '',
  startingWord: '',
  recognitionBuffer: [],

  // Multi-hypothesis beam
  hypotheses: [],

  // Anchor / WPM
  lastAnchorIndex: 0,
  lastAnchorTime: 0,
  anchorWpm: 0,
  wpmSamples: [],   

  // Stall / nudge
  lastAdvanceTime: 0,
  stallNudgeTimer: null,

  // Optimistic creep (UX tracking state)
  creepTargetIndex: 0,       
  creepActive: false,

  // Shadow cursor
  shadowIndex: -1,

  // Speech activity tracking
  lastSpeechTime: 0,

  settings: { 
    fontSize: 2.8, 
    mirror: false,
    wpm: 130,
    theme: 'dark'
  },
  
  vadWorker: null,    
  txWorker:  null,    
  audioContext: null,
  workletNode: null,
  micStream: null,
  isRecording: false,
  modelReady: false,
  modelLoading: false,
  _startPending: false,
  overlayActive: false
};

// ═══════════════════════════════════════════════════════
// ALGORITHM CONFIGURATION
// ═══════════════════════════════════════════════════════
const CFG = {
  ACCEPT_THRESHOLD:        0.30,  
  ANCHOR_THRESHOLD:        0.50,  
  NEW_HYP_MIN_SCORE:       0.40,  
  WINDOW_BASE:             25,    
  WINDOW_MULT:             2.0,
  BEAM_SIZE:               3,
  STALL_MS:                11000, 
  NUDGE_INTERVAL_MS:       7000,
  WPM_DEFAULT:             130,   
  PARA_COMPLETE_RATIO:     0.55,  
  PARA_COMPLETE_DELAY:     500,
  CREEP_SILENCE_PAUSE_MS:  1000,  
  CREEP_MAX_LOOKAHEAD:     2,     
  LOCALITY_HALVING_DIST:   20,    
  SEARCH_LOOKBACK:         11,    
  FAR_JUMP_MIN_TOKENS:     2,     
  FAR_JUMP_MAX_DIST:       20,    
};

let _txCount = 0;

// ═══════════════════════════════════════════════════════
// ALGORITHM LAYER (Double Metaphone + Banded Levenshtein)
// ═══════════════════════════════════════════════════════
function normTok(w) {
  const s  = w.toLowerCase().replace(/[^a-z0-9äöüæøåéàèêëîïôùûüç'-]/gi, '').trim();
  const s2 = s.replace(/'/g, '');
  if (ASR_OVERRIDES[s])  return ASR_OVERRIDES[s];
  if (ASR_OVERRIDES[s2]) return ASR_OVERRIDES[s2];
  const dm = doubleMetaphone(s2 || s);
  return dm[0] || s2 || s;
}

function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^a-z0-9äöüæøåéàèêëîïôùûüç' -]/g, ' ')
    .split(/\s+/)
    .map(w => {
      const c  = w.replace(/[^a-z0-9äöüæøåéàèêëîïôùûüç'-]/g, '').trim();
      const c2 = c.replace(/'/g, '');
      if (ASR_OVERRIDES[c])  return ASR_OVERRIDES[c];
      if (ASR_OVERRIDES[c2]) return ASR_OVERRIDES[c2];
      const dm = doubleMetaphone(c2 || c);
      return dm[0] || c2 || c;
    })
    .filter(w => w.length > 0);
}

// Pre-allocated DP rows to avoid GC collection pressure
const _DP_SIZE = 128;
const _dpPrev = new Int16Array(_DP_SIZE);
const _dpCurr = new Int16Array(_DP_SIZE);

function _bandedSimRange(s, sLen, t, tOff, tWinLen, maxEdits) {
  const tLen = Math.min(tWinLen, t.length - tOff);
  if (sLen === 0 || tLen === 0) return 0;
  const maxLen = Math.max(sLen, tLen);
  if (Math.abs(sLen - tLen) > maxEdits) {
    return Math.max(0, 1 - Math.abs(sLen - tLen) / maxLen);
  }
  const tSize = tLen + 1;
  let prev = tSize <= _DP_SIZE ? _dpPrev : new Int16Array(tSize);
  let curr = tSize <= _DP_SIZE ? _dpCurr : new Int16Array(tSize);
  for (let j = 0; j <= tLen; j++) prev[j] = j;
  for (let i = 1; i <= sLen; i++) {
    curr[0] = i;
    const jStart = Math.max(1, i - maxEdits);
    const jEnd   = Math.min(tLen, i + maxEdits);
    if (jStart > 1)  curr[jStart - 1] = maxEdits + 1;
    if (jEnd < tLen) curr[jEnd + 1]   = maxEdits + 1;
    let rowMin = maxEdits + 1;
    const si = s[i - 1];
    for (let j = jStart; j <= jEnd; j++) {
      curr[j] = si === t[tOff + j - 1]
        ? prev[j - 1]
        : 1 + Math.min(prev[j - 1], prev[j], curr[j - 1]);
      if (curr[j] < rowMin) rowMin = curr[j];
    }
    if (rowMin > maxEdits) return Math.max(0, 1 - (maxEdits + 1) / maxLen);
    const _r = prev; prev = curr; curr = _r;
  }
  return Math.max(0, 1 - prev[tLen] / maxLen);
}

function scoreWindow(spoken, scriptToks, start, slack) {
  const sLen = spoken.length;
  if (sLen === 0 || start >= scriptToks.length) return { score: 0, matchLen: sLen };
  slack = slack || Math.min(2, Math.ceil(sLen * 0.25));
  const maxEdits = Math.ceil(Math.max(sLen, sLen + slack) * 0.55);
  let best = 0;
  let bestMatchLen = sLen;
  for (let d = -slack; d <= slack; d++) {
    const wLen = Math.max(1, sLen + d);
    const sc   = _bandedSimRange(spoken, sLen, scriptToks, start, wLen, maxEdits);
    if (sc > best) { best = sc; bestMatchLen = wLen; if (sc >= 0.98) break; }
  }
  return { score: best, matchLen: bestMatchLen };
}

function buildTokenIndex(scriptToks) {
  const idx = new Map();
  for (let i = 0; i < scriptToks.length; i++) {
    const tok = scriptToks[i];
    if (!idx.has(tok)) idx.set(tok, []);
    idx.get(tok).push(i);
  }
  return idx;
}

const _candidateBitset = new Uint8Array(2048);
const _candidateBuf    = new Int32Array(512);

function getCandidates(spoken, tokenIdx, currentPos, windowSize) {
  const searchFrom = Math.max(0, currentPos - CFG.SEARCH_LOOKBACK);
  const lookahead  = currentPos + windowSize + spoken.length;
  const range      = Math.min(lookahead - searchFrom + 1, _candidateBitset.length);
  _candidateBitset.fill(0, 0, range);
  let len = 0;

  const mark = (c) => {
    const off = c - searchFrom;
    if (off >= 0 && off < range && !_candidateBitset[off] && len < _candidateBuf.length) {
      _candidateBitset[off] = 1;
      _candidateBuf[len++]  = c;
    }
  };

  for (const tok of spoken) {
    const positions = tokenIdx.get(tok);
    if (!positions) continue;
    let lo = 0, hi = positions.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (positions[mid] < searchFrom) lo = mid + 1; else hi = mid;
    }
    for (let k = lo; k < positions.length && positions[k] <= lookahead; k++) {
      const base = positions[k];
      mark(base - 2); mark(base - 1); mark(base); mark(base + 1);
    }
  }
  mark(currentPos);

  const result = _candidateBuf.subarray(0, len);
  result.sort();
  return result;
}

function ariaMatch(spokenText) {
  if (!spokenText?.trim() || !state.words.length) return null;

  let spoken = tokenize(spokenText);
  if (spoken.length === 0) return null;

  // Collapse consecutive duplicates
  const deduped = [spoken[0]];
  for (let i = 1; i < spoken.length; i++) {
    if (spoken[i] !== spoken[i - 1]) deduped.push(spoken[i]);
  }

  // Filter phantom/ASR noise
  const cleaned = deduped.filter(t => state.tokenIndex.has(t));
  spoken = cleaned.length >= 2 ? cleaned : deduped;
  if (spoken.length === 0) return null;

  const scriptToks = state.scriptNormTokens;
  const curPos     = state.creepTargetIndex;
  const scriptLen  = state.wordCount;
  const winSize    = Math.max(CFG.WINDOW_BASE, Math.ceil(spoken.length * CFG.WINDOW_MULT));

  const candidates = getCandidates(spoken, state.tokenIndex, curPos, winSize);
  if (candidates.length === 0) return null;

  let best = null;
  let bestScore = CFG.ACCEPT_THRESHOLD - 0.01;
  const slack = Math.min(2, Math.ceil(spoken.length * 0.25));

  for (const start of candidates) {
    if (start >= scriptLen) continue;
    const distAhead = Math.max(0, start - curPos);
    if (spoken.length < CFG.FAR_JUMP_MIN_TOKENS && distAhead > CFG.FAR_JUMP_MAX_DIST) continue;

    const { score: sc, matchLen } = scoreWindow(spoken, scriptToks, start, slack);
    if (sc <= 0) continue;

    const localityFactor = 1 / (1 + distAhead / CFG.LOCALITY_HALVING_DIST);
    const adjustedScore  = sc * localityFactor;

    if (adjustedScore > bestScore) {
      bestScore = adjustedScore;
      const dest = Math.min(start + matchLen, scriptLen - 1);
      best = { 
        globalIdx: dest, 
        score: adjustedScore, 
        rawScore: sc, 
        startPos: start,
        isAnchor: adjustedScore >= CFG.ANCHOR_THRESHOLD 
      };
      if (sc >= 0.98 && distAhead < 8) break;
    }
  }

  return best;
}

function updateBeam(matchResult) {
  if (!matchResult) {
    state.hypotheses = state.hypotheses
      .map(h => ({ ...h, age: h.age+1, score: h.score*0.82 }))
      .filter(h => h.score > 0.05);
    return pickBestHypothesis();
  }

  const { globalIdx, score } = matchResult;
  const existing = state.hypotheses.find(h => Math.abs(h.pos - globalIdx) <= 4);
  if (existing) {
    existing.score = Math.min(1, existing.score + score * 0.45);
    existing.pos   = Math.round((existing.pos * 0.3 + globalIdx * 0.7)); 
    existing.age   = 0;
  } else {
    if (score >= CFG.NEW_HYP_MIN_SCORE) {
      state.hypotheses.push({ pos: globalIdx, score, age: 0 });
    }
  }

  state.hypotheses = state.hypotheses
    .map(h => h === existing ? h : { ...h, age: h.age+1, score: h.score*0.88 })
    .sort((a,b) => b.score - a.score)
    .slice(0, CFG.BEAM_SIZE);

  return pickBestHypothesis();
}

function pickBestHypothesis() {
  const forward = state.hypotheses.filter(h => h.pos > state.currentWordIndex);
  if (!forward.length) return null;
  return forward.reduce((best, h) => h.score > best.score ? h : best, forward[0]);
}

// ═══════════════════════════════════════════════════════
// WPM ESTIMATION
// ═══════════════════════════════════════════════════════
function recordAnchor(newIdx) {
  const now = Date.now();
  if (state.lastAnchorTime > 0 && newIdx > state.lastAnchorIndex) {
    const words = newIdx - state.lastAnchorIndex;
    const ms    = now - state.lastAnchorTime;
    if (ms > 400) {
      const wpm = words / (ms / 60000);
      const capHi = state.settings.wpm * 1.4;
      if (wpm > 40 && wpm <= capHi) {
        state.wpmSamples.push(wpm);
        if (state.wpmSamples.length > 8) state.wpmSamples.shift();
        const sorted = [...state.wpmSamples].sort((a,b)=>a-b);
        const trim   = sorted.slice(1, -1);
        state.anchorWpm = trim.length ? trim.reduce((s,v)=>s+v,0)/trim.length : sorted[0];
      }
    }
  }
  state.lastAnchorIndex = newIdx;
  state.lastAnchorTime  = now;
}

function effectiveWpm() {
  if (state.anchorWpm > 0) return state.anchorWpm;
  return state.settings.wpm || CFG.WPM_DEFAULT;
}

// ═══════════════════════════════════════════════════════
// POSITION MOVES
// ═══════════════════════════════════════════════════════
function confirmMove(globalIdx, smooth) {
  if (globalIdx < 0 || globalIdx >= state.wordCount) return;
  const creepAhead  = state.currentWordIndex - globalIdx;
  const silentMs    = Date.now() - state.lastSpeechTime;
  const speakerPaused = state.lastSpeechTime > 0 && silentMs > CFG.CREEP_SILENCE_PAUSE_MS;

  if (globalIdx > state.currentWordIndex) {
    snapTo(globalIdx, smooth);
    state.creepTargetIndex = globalIdx;
    return;
  }

  if (creepAhead > 0 && creepAhead <= CFG.CREEP_MAX_LOOKAHEAD && !speakerPaused) {
    state.creepTargetIndex = globalIdx;
    return;
  }

  state.creepTargetIndex = globalIdx;
}

function snapTo(globalIdx, smooth) {
  if (globalIdx <= state.currentWordIndex) return;
  if (globalIdx >= state.wordCount) globalIdx = state.wordCount - 1;

  state.currentWordIndex = globalIdx;
  state.lastAdvanceTime  = Date.now();
  
  updateShadowCursor();
  updateProgress();
  updateWPM();
  syncStateToOverlay(smooth ? 'snap' : 'instant');
}

function moveTo(globalIdx, smooth = true) {
  snapTo(globalIdx, smooth);
  state.creepTargetIndex = Math.max(state.creepTargetIndex, globalIdx);
}

function updateShadowCursor() {
  const shadowIdx = state.creepTargetIndex + 2;
  if (shadowIdx < state.wordCount) {
    state.shadowIndex = shadowIdx;
  } else {
    state.shadowIndex = -1;
  }
}

function seekToWord(wordIdx) {
  if (wordIdx < 0 || wordIdx >= state.wordCount) return;

  state.currentWordIndex = wordIdx;
  state.creepTargetIndex = wordIdx;
  state.shadowIndex = -1;
  state.lastAdvanceTime = Date.now();

  state.hypotheses        = [];
  state.accumulatedText   = '';
  state.lastChunk         = '';
  state.startingWord      = '';
  state.recognitionBuffer = [];

  updateParaForPos(wordIdx);

  state.lastAnchorIndex = wordIdx;
  state.lastAnchorTime  = Date.now();

  updateProgress();
  syncStateToOverlay('snap');

  if (state.isRecording) {
    scheduleStallNudge();
  }
}

// ═══════════════════════════════════════════════════════
// TEXT PARSING & RENDER SYNC
// ═══════════════════════════════════════════════════════
function normalizeWord(w) {
  return w.toLowerCase().replace(/[^a-z0-9äöüæøåéàèêëîïôùûüç'-]/gi,'').trim();
}

function parseScript(text) {
  const rawParas = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const paragraphs = [];
  const allWords = [];

  rawParas.forEach((paraText, pIdx) => {
    const startIndex = allWords.length;
    const words = [];
    paraText.split(/\s+/).filter(Boolean).forEach((token, localIdx) => {
      const norm = normalizeWord(token);
      if (!norm) return;
      const word = {
        id: allWords.length, localId: localIdx,
        text: token, normalized: norm,
        normToken: normTok(norm),
      };
      allWords.push(word);
      words.push(word);
    });
    if (words.length > 0) paragraphs.push({ text: paraText, words, startIndex, id: pIdx });
  });

  return { paragraphs, allWords };
}

function renderScript() {
  const text = dom.scriptInput.value.trim();
  if (!text) {
    state.paragraphs = []; state.words = []; state.wordCount = 0;
    state.scriptNormTokens = []; state.tokenIndex = null;
    updateProgress();
    if (state.overlayActive) {
      window.electronAPI.controlAction('update-script', { paragraphs: [], words: [] });
    }
    return;
  }

  const { paragraphs, allWords } = parseScript(text);
  state.paragraphs = paragraphs;
  state.words      = allWords;
  state.wordCount  = allWords.length;
  state.scriptNormTokens = allWords.map(w => w.normToken);
  state.tokenIndex = buildTokenIndex(state.scriptNormTokens);

  resetPosition(false);

  // Sync script to overlay
  if (state.overlayActive) {
    window.electronAPI.controlAction('update-script', { paragraphs, allWords });
  }
}

function resetPosition(clearTx = true) {
  if (state.paragraphCompleteTimer) { clearTimeout(state.paragraphCompleteTimer); state.paragraphCompleteTimer = null; }
  if (state.stallNudgeTimer)        { clearTimeout(state.stallNudgeTimer);        state.stallNudgeTimer        = null; }

  state.currentWordIndex = 0;
  state.currentParaIndex = 0;
  state.sessionStartTime = null;
  state.lastAdvanceTime  = 0;
  state.hypotheses       = [];
  state.lastAnchorIndex  = 0;
  state.lastAnchorTime   = 0;
  state.anchorWpm        = 0;
  state.wpmSamples       = [];
  state.creepTargetIndex = 0;
  state.shadowIndex      = -1;
  state.lastSpeechTime   = 0;
  _txCount               = 0;

  if (clearTx) { 
    state.accumulatedText = ''; 
    state.lastChunk = ''; 
    state.startingWord = ''; 
    state.recognitionBuffer = []; 
    dom.transcriptBox.innerHTML = '<span class="transcript-placeholder">Start speaking to see your text transcribed in real-time...</span>';
  }

  updateProgress();
  
  if (dom.wpmValue)  dom.wpmValue.textContent = '—';
  if (dom.etaValue)  dom.etaValue.textContent = '--';

  syncStateToOverlay('instant');
}

// ═══════════════════════════════════════════════════════
// TRANSCRIPT ACCUMULATION & ALIGNMENT
// ═══════════════════════════════════════════════════════
function processTranscript(chunk, isFinal) {
  if (!chunk?.trim() || !state.paragraphs.length) return;
  _txCount++;

  // Track speech activity — used by stall nudge silence gate
  state.lastSpeechTime = Date.now();

  const acc = isFinal ? accumulateTranscript(chunk) : (state.accumulatedText ? state.accumulatedText + ' ' + chunk : chunk);
  if (isFinal) state.recognitionBuffer = [...state.recognitionBuffer, chunk].slice(-8);

  // ── Run 4 ARIA matches (all synchronous but fast) ──────────────────────
  const m1 = ariaMatch(chunk);                                    // new chunk only
  const m2 = state.recognitionBuffer.length >= 2
    ? ariaMatch(state.recognitionBuffer.slice(-3).join(' '))      // recent buffer
    : null;
  const m3 = isFinal ? ariaMatch(acc) : null;                    // full accumulated (final only)
  
  const accTail = acc ? acc.split(/\s+/).slice(-12).join(' ') : null;
  const m4 = (accTail && accTail !== chunk && accTail !== acc) ? ariaMatch(accTail) : null;

  // Pick best
  let best = null;
  const matchers = [m1, m2, m3, m4];
  for (const m of matchers) {
    if (m && (!best || m.score > best.score)) { best = m; }
  }

  // ── Update beam ──
  const hyp = updateBeam(best);

  // ── Advance if beam converged ──
  if (hyp && hyp.pos > state.currentWordIndex) {
    const target = Math.min(hyp.pos, state.wordCount - 1);

    confirmMove(target, isFinal);
    updateParaForPos(target);

    if (best?.isAnchor) recordAnchor(target);

    scheduleStallNudge();
    state.creepTargetIndex = target;
  }

  // ── Paragraph completion ──
  if (isFinal) checkParaCompletion(acc);
}

function accumulateTranscript(chunk) {
  if (!chunk.trim()) return state.accumulatedText;
  const newWords     = chunk.trim().split(/\s+/).filter(Boolean);
  const newFirstWord = normalizeWord(newWords[0]);

  let acc;
  if (state.startingWord && newFirstWord === state.startingWord && state.lastChunk) {
    const base = state.accumulatedText.endsWith(state.lastChunk)
      ? state.accumulatedText.slice(0, -state.lastChunk.length).trimEnd()
      : state.accumulatedText;
    acc = base ? base + ' ' + chunk.trim() : chunk.trim();
  } else {
    acc = state.accumulatedText ? state.accumulatedText + ' ' + chunk.trim() : chunk.trim();
  }

  const toks = acc.split(/\s+/);
  if (toks.length > 150) acc = toks.slice(-150).join(' ');

  state.lastChunk      = chunk.trim();
  state.startingWord   = newFirstWord;
  state.accumulatedText = acc;
  return acc;
}

function checkParaCompletion(acc) {
  const para = state.paragraphs[state.currentParaIndex];
  if (!para) return;

  const localIdx = state.currentWordIndex - para.startIndex;
  if (localIdx >= para.words.length - 3) { triggerParaComplete(); return; }

  if (localIdx < para.words.length * 0.7) return;

  const lastWords    = para.words.slice(-3).filter(w => w.normalized.length >= 4);
  if (!lastWords.length) { triggerParaComplete(); return; }

  const textLower = acc.toLowerCase().replace(/[^a-z0-9äöüæøåéàèêëîïôùûüç' -]/g,' ');
  const matched   = lastWords.filter(w => textLower.includes(w.normalized) || textLower.includes(w.normToken)).length;
  const needed    = Math.max(1, Math.ceil(lastWords.length * CFG.PARA_COMPLETE_RATIO));
  if (matched >= needed) triggerParaComplete();
}

function triggerParaComplete() {
  if (state.paragraphCompleteTimer) return;
  state.paragraphCompleteTimer = setTimeout(() => {
    state.paragraphCompleteTimer = null;
    advancePara();
  }, CFG.PARA_COMPLETE_DELAY);
}

function advancePara() {
  const next = state.currentParaIndex + 1;
  if (next >= state.paragraphs.length) { 
    setStatus('idle','Script complete!'); 
    stopAudio();
    return; 
  }

  state.currentParaIndex   = next;
  state.accumulatedText    = '';
  state.lastChunk          = '';
  state.startingWord       = '';
  state.recognitionBuffer  = [];
  state.hypotheses         = [];

  const para = state.paragraphs[next];
  if (para) moveTo(para.startIndex, true);
}

function updateParaForPos(idx) {
  for (let pi = state.paragraphs.length - 1; pi >= 0; pi--) {
    if (idx >= state.paragraphs[pi].startIndex) {
      if (pi !== state.currentParaIndex) {
        state.currentParaIndex   = pi;
        state.accumulatedText    = '';
        state.lastChunk          = '';
        state.startingWord       = '';
        state.recognitionBuffer  = [];
      }
      break;
    }
  }
}

function scheduleStallNudge() {
  if (state.stallNudgeTimer) clearTimeout(state.stallNudgeTimer);
  state.stallNudgeTimer = setTimeout(function nudge() {
    if (!state.isRecording) return;
    const stalledMs = Date.now() - state.lastAdvanceTime;
    if (stalledMs >= CFG.STALL_MS) {
      const speechRecency  = Date.now() - state.lastSpeechTime;
      const userIsSpeaking = state.lastSpeechTime > 0 && speechRecency < 5000;
      if (userIsSpeaking) {
        const wpm    = effectiveWpm();
        const words  = Math.max(1, Math.round(wpm * (stalledMs / 60000) * 0.35));
        const target = Math.min(state.currentWordIndex + words, state.wordCount - 1);
        if (target > state.currentWordIndex) {
          moveTo(target, true);
          updateParaForPos(target);
        }
      }
    }
    state.stallNudgeTimer = setTimeout(nudge, CFG.NUDGE_INTERVAL_MS);
  }, CFG.STALL_MS);
}

// ═══════════════════════════════════════════════════════
// SYNC STATE TO OVERLAY
// ═══════════════════════════════════════════════════════
function syncStateToOverlay(scrollMode = 'snap') {
  if (!state.overlayActive) return;
  
  window.electronAPI.syncState({
    currentWordIndex: state.currentWordIndex,
    creepTargetIndex: state.creepTargetIndex,
    shadowIndex: state.shadowIndex,
    lastSpeechTime: state.lastSpeechTime,
    isRecording: state.isRecording,
    
    // Configurations
    wpm: state.settings.wpm,
    fontSize: state.settings.fontSize,
    mirror: state.settings.mirror,
    
    // Commands
    scrollMode: scrollMode
  });
}

// ═══════════════════════════════════════════════════════
// WORKER MANAGEMENT
// ═══════════════════════════════════════════════════════
function initWorkers() {
  if (state.vadWorker) return;

  const handleMessage = ({ data }) => {
    const { type, status, message, text, isFinal } = data;

    if (type === 'status') {
      if (status === 'loading') {
        setStatus('loading', message || 'Loading Speech AI...');
        state.modelLoading = true; 
        state.modelReady = false;
      } else if (status === 'ready') {
        setStatus('idle', 'Speech Model Ready');
        state.modelLoading = false; 
        state.modelReady = true;
        if (dom.modelProgress) dom.modelProgress.classList.remove('active');
        if (dom.engineIndicator) dom.engineIndicator.textContent = 'Engine: Moonshine AI (Active)';
        
        dom.startBtn.removeAttribute('disabled');
        if (state._startPending) { 
          state._startPending = false; 
          startAudio(); 
        }
      } else if (status === 'recording') {
        setStatus('recording', 'Listening...');
        state.lastSpeechTime = Date.now();
        syncStateToOverlay('snap');
      } else if (status === 'transcribing') {
        setStatus('loading', 'Transcribing...');
      }
    }

    if (type === 'transcript') {
      if (text?.trim()) {
        state.lastSpeechTime = Date.now();
        appendTranscript(text, isFinal === false);
        processTranscript(text, isFinal !== false);
        
        if (!state.sessionStartTime) {
          state.sessionStartTime = Date.now();
          state.lastAdvanceTime  = Date.now();
        }
        syncStateToOverlay('snap');
      }
    }

    if (type === 'progress' && dom.modelProgressBar && dom.modelProgress) {
      dom.modelProgress.classList.add('active');
      dom.modelProgressBar.style.width = `${data.percent}%`;
      setStatus('loading', `Downloading AI: ${data.percent}%`);
    }

    if (type === 'info') {
      console.log('[ASR Workers]', message);
      if (dom.engineIndicator && message.includes('Device:')) {
        dom.engineIndicator.textContent = `Engine: Moonshine AI (${message.includes('webgpu') ? 'WebGPU' : 'WASM'})`;
      }
    }

    if (type === 'error') {
      setStatus('idle', `Speech Engine Error: ${message}`);
      state.modelLoading = false;
    }
  };

  const handleError = err => {
    console.error('ASR Worker failed:', err);
    setStatus('idle', 'ASR model failed to initialize');
    state.modelLoading = false;
  };

  // VAD Worker setup
  state.vadWorker = new Worker('./vad.worker.js', { type: 'module' });
  state.vadWorker.onmessage = ({ data }) => {
    if (data.type === 'segment') {
      state.txWorker.postMessage(data, data.buffer ? [data.buffer.buffer] : []);
      return;
    }
    handleMessage({ data });
  };
  state.vadWorker.onerror = handleError;

  // Transcribe Worker setup
  state.txWorker = new Worker('./transcribe.worker.js', { type: 'module' });
  state.txWorker.onmessage = handleMessage;
  state.txWorker.onerror   = handleError;
}

// ═══════════════════════════════════════════════════════
// MICROPHONE CAPTURE
// ═══════════════════════════════════════════════════════
const WORKLET_SRC = `
const MIN_CHUNK = 512;
let ptr = 0;
let buf = new Float32Array(MIN_CHUNK);
class VADProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    if (ch.length >= MIN_CHUNK) {
      this.port.postMessage({ buffer: ch });
    } else {
      const rem = MIN_CHUNK - ptr;
      if (ch.length >= rem) {
        buf.set(ch.subarray(0, rem), ptr);
        const _send = buf;
        buf = new Float32Array(MIN_CHUNK);
        buf.set(ch.subarray(rem), 0);
        ptr = ch.length - rem;
        this.port.postMessage({ buffer: _send }, [_send.buffer]);
      } else {
        buf.set(ch, ptr);
        ptr += ch.length;
      }
    }
    return true;
  }
}
registerProcessor('vad-processor', VADProcessor);
`;

async function startAudio() {
  if (state.isRecording) return;
  
  // Request system level permission (Electron level check)
  const isMicGranted = await window.electronAPI.requestMicrophone();
  if (!isMicGranted) {
    setStatus('idle', 'Microphone permission denied on macOS.');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    state.micStream = stream;

    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    await state.audioContext.resume();

    const blob    = new Blob([WORKLET_SRC], { type: 'application/javascript' });
    const blobURL = URL.createObjectURL(blob);
    await state.audioContext.audioWorklet.addModule(blobURL);
    URL.revokeObjectURL(blobURL);

    const source      = state.audioContext.createMediaStreamSource(stream);
    state.workletNode = new AudioWorkletNode(state.audioContext, 'vad-processor');
    state.workletNode.port.onmessage = e => {
      if (state.vadWorker && e.data.buffer) {
        state.vadWorker.postMessage({ buffer: e.data.buffer });
      }
    };
    source.connect(state.workletNode);

    state.isRecording  = true;
    state.lastAdvanceTime = Date.now();
    if (!state.sessionStartTime) state.sessionStartTime = Date.now();

    setStatus('recording', 'Listening...');
    updateButtons(true);
    scheduleStallNudge();

    syncStateToOverlay('snap');
  } catch (err) {
    console.error('Audio capture error:', err);
    setStatus('idle', `Mic capture error: ${err.message}`);
    updateButtons(false);
  }
}

function stopAudio() {
  if (!state.isRecording) return;

  if (state.workletNode) {
    state.workletNode.disconnect();
    state.workletNode = null;
  }
  if (state.micStream) {
    state.micStream.getTracks().forEach(t => t.stop());
    state.micStream = null;
  }
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }

  state.isRecording = false;
  
  if (state.stallNudgeTimer) {
    clearTimeout(state.stallNudgeTimer);
    state.stallNudgeTimer = null;
  }

  setStatus('idle', 'Stopped');
  updateButtons(false);

  syncStateToOverlay('instant');
}

// ═══════════════════════════════════════════════════════
// UI & DISPLAY INTERACTIVE OPERATIONS
// ═══════════════════════════════════════════════════════
const $ = id => document.getElementById(id);
const dom = {
  statusBadge:      $('statusBadge'),
  statusText:       $('statusText'),
  startBtn:         $('startBtn'),
  stopBtn:          $('stopBtn'),
  engineIndicator:  $('engineIndicator'),
  scriptInput:      $('scriptInput'),
  clearScriptBtn:   $('clearScriptBtn'),
  sampleScriptBtn:  $('sampleScriptBtn'),
  wpmValue:         $('wpmValue'),
  progressValue:    $('progressValue'),
  etaValue:         $('etaValue'),
  fontSizeRange:    $('fontSizeRange'),
  fontSizeVal:      $('fontSizeVal'),
  wpmRange:         $('wpmRange'),
  wpmRangeVal:      $('wpmRangeVal'),
  mirrorToggle:     $('mirrorToggle'),
  resetBtn:         $('resetBtn'),
  modelProgress:    $('modelProgress'),
  modelProgressBar: $('modelProgressBar'),
  launchOverlayBtn: $('launchOverlayBtn'),
  transcriptBox:    $('transcriptBox'),
  interimBox:       $('interimBox'),
  themeToggle:      $('themeToggle')
};

function setStatus(type, text) {
  if (!dom.statusBadge || !dom.statusText) return;
  dom.statusBadge.className = `status-badge status-${type}`;
  dom.statusText.textContent = text;
}

function appendTranscript(text, isPartial) {
  const ph = dom.transcriptBox.querySelector('.transcript-placeholder');
  if (ph) ph.remove();
  
  if (dom.interimBox) dom.interimBox.textContent = '';
  
  if (isPartial) {
    if (dom.interimBox) dom.interimBox.textContent = `Hearing: "${text}"`;
  } else {
    const seg = document.createElement('span');
    seg.className = 'transcript-segment';
    seg.textContent = text + ' ';
    dom.transcriptBox.appendChild(seg);
    dom.transcriptBox.scrollTop = dom.transcriptBox.scrollHeight;
  }
}

function updateButtons(recording) {
  if (recording) {
    dom.startBtn.setAttribute('disabled', 'true');
    dom.stopBtn.removeAttribute('disabled');
  } else {
    dom.startBtn.removeAttribute('disabled');
    dom.stopBtn.setAttribute('disabled', 'true');
  }
}

function updateWPM() {
  if (!state.sessionStartTime || state.currentWordIndex === 0) return;
  const wpm = state.anchorWpm > 0
    ? Math.round(state.anchorWpm)
    : Math.round(state.currentWordIndex / ((Date.now() - state.sessionStartTime) / 60000));
  if (dom.wpmValue) dom.wpmValue.textContent = wpm || '—';
}

function updateProgress() {
  if (!state.wordCount) {
    if (dom.progressValue) dom.progressValue.textContent = '0%';
    if (dom.etaValue)     dom.etaValue.textContent = '--';
    return;
  }
  const pct = Math.min(100, Math.round((state.currentWordIndex / state.wordCount) * 100));
  if (dom.progressValue) dom.progressValue.textContent = `${pct}%`;

  if (dom.etaValue) {
    const wordsLeft = state.wordCount - state.currentWordIndex;
    const wpm = effectiveWpm();
    if (!state.isRecording || wordsLeft <= 0 || wpm <= 0) {
      dom.etaValue.textContent = '--';
    } else {
      const totalSec = Math.round((wordsLeft / wpm) * 60);
      const mins = Math.floor(totalSec / 60);
      const secs = totalSec % 60;
      dom.etaValue.textContent = mins > 0
        ? `${mins}m ${secs}s left`
        : `${secs}s left`;
    }
  }
}

// Bind interactive event listeners
function bindEvents() {
  dom.startBtn.onclick = () => {
    if (!state.modelReady) {
      state._startPending = true;
      initWorkers();
    } else {
      startAudio();
    }
  };
  
  dom.stopBtn.onclick = () => stopAudio();
  
  dom.resetBtn.onclick = () => resetPosition(true);
  
  dom.clearScriptBtn.onclick = () => {
    dom.scriptInput.value = '';
    renderScript();
  };
  
  dom.sampleScriptBtn.onclick = () => {
    dom.scriptInput.value = SAMPLE_SCRIPT;
    renderScript();
  };

  dom.scriptInput.oninput = () => renderScript();

  dom.fontSizeRange.oninput = (e) => {
    const val = parseFloat(e.target.value);
    dom.fontSizeVal.textContent = `${val.toFixed(1)}rem`;
    state.settings.fontSize = val;
    syncStateToOverlay('instant');
  };

  dom.wpmRange.oninput = (e) => {
    const val = parseInt(e.target.value);
    dom.wpmRangeVal.textContent = `${val} WPM`;
    state.settings.wpm = val;
    syncStateToOverlay('instant');
  };

  dom.mirrorToggle.onchange = (e) => {
    state.settings.mirror = e.target.checked;
    syncStateToOverlay('instant');
  };

  dom.themeToggle.onclick = () => {
    if (document.body.classList.contains('light-theme')) {
      document.body.classList.remove('light-theme');
      state.settings.theme = 'dark';
    } else {
      document.body.classList.add('light-theme');
      state.settings.theme = 'light';
    }
    syncStateToOverlay('instant');
  };

  // Launch transparent overlay window via IPC
  dom.launchOverlayBtn.onclick = () => {
    if (state.overlayActive) {
      window.electronAPI.controlAction('close-overlay');
      dom.launchOverlayBtn.textContent = '✨ Launch Teleprompter Overlay';
      dom.launchOverlayBtn.classList.remove('btn-danger');
      dom.launchOverlayBtn.classList.add('btn-primary');
      state.overlayActive = false;
    } else {
      const { paragraphs, allWords } = parseScript(dom.scriptInput.value || SAMPLE_SCRIPT);
      state.overlayActive = true;
      dom.launchOverlayBtn.textContent = '❌ Close Teleprompter Overlay';
      dom.launchOverlayBtn.classList.remove('btn-primary');
      dom.launchOverlayBtn.classList.add('btn-danger');
      
      window.electronAPI.controlAction('open-overlay', {
        paragraphs,
        allWords,
        settings: state.settings
      });
    }
  };

  // Pre-load model immediately on startup to avoid delays
  setTimeout(() => {
    initWorkers();
  }, 1000);
}

// ═══════════════════════════════════════════════════════
// IPC RELAY MESSAGES (Bidirectional)
// ═══════════════════════════════════════════════════════
function setupIpcListeners() {
  // Listen for global shortcut triggers from main process
  window.electronAPI.onHotkeyTriggered((hotkey) => {
    if (hotkey === 'toggle-pause') {
      if (state.isRecording) {
        stopAudio();
      } else {
        if (!state.modelReady) {
          state._startPending = true;
          initWorkers();
        } else {
          startAudio();
        }
      }
    }
  });

  // Listen for window state commands
  window.electronAPI.onControlEvent((action, data) => {
    if (action === 'overlay-closed') {
      dom.launchOverlayBtn.textContent = '✨ Launch Teleprompter Overlay';
      dom.launchOverlayBtn.classList.remove('btn-danger');
      dom.launchOverlayBtn.classList.add('btn-primary');
      state.overlayActive = false;
      stopAudio();
    } else if (action === 'seek-to-word') {
      seekToWord(data);
    } else if (action === 'reset-position') {
      resetPosition(true);
    }
  });
}

// ═══════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  dom.scriptInput.value = SAMPLE_SCRIPT;
  renderScript();
  bindEvents();
  setupIpcListeners();
});
