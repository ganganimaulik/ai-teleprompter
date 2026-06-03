'use strict';

// ═══════════════════════════════════════════════════════
// STATE OBJECT
// ═══════════════════════════════════════════════════════
const state = {
  paragraphs: [],
  words: [],
  wordCount: 0,
  
  currentWordIndex: 0,
  creepTargetIndex: 0,
  shadowIndex: -1,
  lastConfirmedWordIndex: -1,
  
  isRecording: false,
  lastSpeechTime: 0,
  
  settings: {
    fontSize: 2.8,
    mirror: false,
    wpm: 120,
    overlayTheme: 'glass',
    overlayHighlight: 'glow',
    overlayFont: 'sans',
    overlayAlign: 'center',
    overlayOpacity: 20,
    overlayBlur: 8
  },
  
  // Creep animation variables
  creepFractional: 0,
  creepLastTime: 0
};

// Algorithmic Constants
const CREEP_SILENCE_PAUSE_MS = 1000;
const CREEP_MAX_LOOKAHEAD = 2;
const SCROLL_LERP = 0.09;

// Scroll Animation variables
let _scrollTarget = 0;
let _scrollCurrent = 0;
let _scrollRafId = null;

// Span lookup cache for fast O(1) retrieval
let _spanCache = [];

const $ = id => document.getElementById(id);
const dom = {
  overlayContainer:  $('overlayContainer'),
  controlBar:        $('controlBar'),
  overlayDragHandle: $('overlayDragHandle'),
  teleprompterScroll:$('teleprompterScroll'),
  teleprompterContent:$('teleprompterContent'),
  highlightPill:     $('highlightPill'),
  overlayPlayBtn:    $('overlayPlayBtn'),
  overlayCloseBtn:   $('overlayCloseBtn')
};

// ═══════════════════════════════════════════════════════
// SMART HOVER PASSTHROUGH (Click-through toggler)
// ═══════════════════════════════════════════════════════
// Because Electron allows setIgnoreMouseEvents(true, { forward: true }),
// we can capture mouse movements in the webpage even when clicks pass through.
// When the cursor is over interactive elements (like the control bar),
// we disable ignoreMouseEvents so the buttons can be clicked.
window.addEventListener('mousemove', (e) => {
  const isInteractive = e.target.closest('.interactive');
  if (isInteractive) {
    window.electronAPI.setClickThrough(false);
  } else {
    // Forward mouse events allows mousemove to trigger even when clicks pass through
    window.electronAPI.setClickThrough(true, { forward: true });
  }
});

// ═══════════════════════════════════════════════════════
// SCRIPT RENDERING
// ═══════════════════════════════════════════════════════
function renderScript(paragraphs, allWords) {
  state.paragraphs = paragraphs;
  state.words = allWords;
  state.wordCount = allWords.length;
  _spanCache = new Array(allWords.length);

  const container = document.createElement('div');
  paragraphs.forEach((para, pi) => {
    const paraDiv = document.createElement('div');
    paraDiv.className = 'teleprompter-paragraph';
    
    // Find words belonging to this paragraph
    const paraWords = allWords.filter(w => w.id >= para.startIndex && w.id < (para.startIndex + (para.words?.length || 0)));
    
    paraWords.forEach((word, wi) => {
      const span = document.createElement('span');
      span.className = 'word';
      span.dataset.id = word.id;
      span.textContent = word.text;
      
      // Allow clicking on words to seek when overlay is interactive
      span.onclick = () => {
        if (state.overlayActive || !window.electronAPI) return;
        window.electronAPI.controlAction('seek-to-word', word.id);
      };
      span.classList.add('interactive'); // Allow interaction/clicking
      
      _spanCache[word.id] = span;
      paraDiv.appendChild(span);
      if (wi < paraWords.length - 1) {
        paraDiv.appendChild(document.createTextNode(' '));
      }
    });
    
    container.appendChild(paraDiv);
  });

  dom.teleprompterContent.innerHTML = '';
  dom.teleprompterContent.appendChild(container);
  
  // Re-append pill inside content
  if (dom.highlightPill) {
    dom.teleprompterContent.appendChild(dom.highlightPill);
    dom.highlightPill.style.opacity = '0';
  }

  // Set font size and mirror configuration
  updateVisualSettings();
  resetPosition();
  
  // Ensure the script scrolls to the first word (at the focus line) on start
  requestAnimationFrame(() => {
    scrollToCurrent(false);
  });
}

function getWordSpan(id) {
  return (id >= 0 && id < _spanCache.length) ? _spanCache[id] : null;
}

function updateVisualSettings() {
  // Update Font size
  dom.teleprompterContent.style.fontSize = `${state.settings.fontSize}rem`;
  
  // Update mirroring
  if (state.settings.mirror) {
    dom.teleprompterContent.style.transform = `scaleX(-1) translateY(${_scrollCurrent}px)`;
  } else {
    dom.teleprompterContent.style.transform = `translateY(${_scrollCurrent}px)`;
  }
  
  // Apply Font Style
  dom.teleprompterContent.classList.remove('font-sans', 'font-mono');
  dom.teleprompterContent.classList.add(`font-${state.settings.overlayFont || 'sans'}`);
  
  // Apply Alignment
  dom.teleprompterContent.classList.remove('align-center', 'align-left');
  dom.teleprompterContent.classList.add(`align-${state.settings.overlayAlign || 'center'}`);
  
  // Apply Theme/Backdrop Preset
  dom.overlayContainer.className = 'teleprompter-overlay-container';
  if (state.isRecording) {
    dom.overlayContainer.classList.add('listening');
  }
  dom.overlayContainer.classList.add(`theme-${state.settings.overlayTheme || 'glass'}`);
  
  // Apply Highlight Style class
  dom.overlayContainer.classList.remove('highlight-style-glow', 'highlight-style-pill', 'highlight-style-underline', 'highlight-style-minimal');
  dom.overlayContainer.classList.add(`highlight-style-${state.settings.overlayHighlight || 'glow'}`);
  
  // Apply Backdrop Opacity & Blur variables
  const opacity = state.settings.overlayOpacity !== undefined ? state.settings.overlayOpacity : 20;
  const blur = state.settings.overlayBlur !== undefined ? state.settings.overlayBlur : 8;
  dom.overlayContainer.style.setProperty('--overlay-bg-opacity', (opacity / 100).toFixed(2));
  dom.overlayContainer.style.setProperty('--overlay-blur', `${blur}px`);
  
  // Apply Stealth / Hide features
  if (state.settings.overlayHidden) {
    document.body.style.opacity = '0';
  } else if (state.settings.overlayFaded) {
    document.body.style.opacity = '0.15';
  } else {
    document.body.style.opacity = '1';
  }
  
  movePillToWord(state.currentWordIndex);
}

function resetPosition() {
  state.currentWordIndex = 0;
  state.creepTargetIndex = 0;
  state.shadowIndex = -1;
  state.lastConfirmedWordIndex = -1;
  state.creepFractional = 0;

  if (_scrollRafId) {
    cancelAnimationFrame(_scrollRafId);
    _scrollRafId = null;
  }
  _scrollTarget = 0;
  _scrollCurrent = 0;

  for (let i = 0; i < _spanCache.length; i++) {
    const span = _spanCache[i];
    if (span) {
      span.className = 'word interactive';
    }
  }

  _applyTransformRaw(0);
  highlightCurrent();
  if (dom.highlightPill) dom.highlightPill.style.opacity = '0';
}

// ═══════════════════════════════════════════════════════
// ANIMATION & SCROLL LERP LOOP
// ═══════════════════════════════════════════════════════
function startScrollAnim() {
  if (!_scrollRafId) {
    _scrollRafId = requestAnimationFrame(scrollFrame);
  }
}

function scrollFrame(now) {
  // 1. Speculative Creep Advancements (Runs locally for maximum smoothness)
  if (state.isRecording && state.wordCount > 0) {
    if (state.creepLastTime === 0) state.creepLastTime = now;
    const dt = now - state.creepLastTime;
    state.creepLastTime = now;

    const silentMs = Date.now() - state.lastSpeechTime;
    
    // Only creep if speaking, not paused, and not ad-libbing
    if (state.lastSpeechTime > 0 && silentMs <= CREEP_SILENCE_PAUSE_MS && !state.isAdLibbing) {
      const wordsPerMs = (state.settings.wpm * 0.85) / 60000;
      
      const prevWord = state.words[state.currentWordIndex - 1];
      const prevText = prevWord ? prevWord.text : '';
      const endsWithSentence = /[.!?:;—]["'»)]*$/.test(prevText);
      const puncScale = endsWithSentence ? 0.45 : 1;

      state.creepFractional += wordsPerMs * dt * puncScale;

      if (state.creepFractional >= 1) {
        const steps = Math.floor(state.creepFractional);
        state.creepFractional -= steps;

        const maxCreep = state.creepTargetIndex + CREEP_MAX_LOOKAHEAD;
        const target = Math.min(state.currentWordIndex + steps, maxCreep, state.wordCount - 1);

        if (target > state.currentWordIndex) {
          moveCreep(target);
        }
      }
    }
  } else {
    state.creepLastTime = 0;
    state.creepFractional = 0;
  }

  // 2. Linear Interpolation of Scroll coordinates
  const diff = _scrollTarget - _scrollCurrent;
  if (Math.abs(diff) < 0.05) {
    if (_scrollCurrent !== _scrollTarget) {
      _scrollCurrent = _scrollTarget;
      _applyTransformRaw(_scrollCurrent);
    }
    
    // If not recording, we can stop the loop once aligned
    if (!state.isRecording) {
      _scrollRafId = null;
      return;
    }
  } else {
    _scrollCurrent += diff * SCROLL_LERP;
    _applyTransformRaw(_scrollCurrent);
  }

  _scrollRafId = requestAnimationFrame(scrollFrame);
}

function _applyTransformRaw(ty) {
  dom.teleprompterContent.style.transform = state.settings.mirror
    ? `scaleX(-1) translateY(${ty}px)`
    : `translateY(${ty}px)`;
}

function moveCreep(idx) {
  if (idx <= state.currentWordIndex) return;
  if (idx >= state.wordCount) idx = state.wordCount - 1;

  // Mark words between current index and new index as 'creep' (lighter colored)
  for (let i = state.currentWordIndex; i < idx; i++) {
    const span = getWordSpan(i);
    if (span && !span.classList.contains('spoken')) {
      span.classList.add('creep');
    }
  }

  state.currentWordIndex = idx;
  highlightCurrent();
  scrollToCurrent(true);
}

function snapTo(globalIdx, smooth) {
  if (globalIdx >= state.wordCount) globalIdx = state.wordCount - 1;

  // Reclassify all preceding words as spoken (fully highlighted)
  for (let i = 0; i < globalIdx; i++) {
    const span = getWordSpan(i);
    if (span) {
      span.classList.remove('current', 'creep', 'shadow');
      span.classList.add('spoken');
    }
  }

  // Clear any trailing words
  for (let i = globalIdx + 1; i < state.wordCount; i++) {
    const span = getWordSpan(i);
    if (span) {
      span.classList.remove('spoken', 'current', 'creep', 'shadow');
    }
  }

  state.currentWordIndex = globalIdx;
  highlightCurrent();
  scrollToCurrent(smooth);
  updateShadowCursor();
}

function updateShadowCursor() {
  // Remove old shadow highlight
  const oldShadow = dom.teleprompterContent.querySelector('.word.shadow-curr');
  if (oldShadow) {
    oldShadow.classList.remove('shadow-curr');
  }

  if (state.shadowIndex >= 0 && state.shadowIndex < state.wordCount) {
    const span = getWordSpan(state.shadowIndex);
    if (span && !span.classList.contains('current') && !span.classList.contains('spoken')) {
      span.classList.add('shadow-curr');
    }
  }
}

function highlightCurrent() {
  const prev = dom.teleprompterContent.querySelector('.word.current');
  if (prev) prev.classList.remove('current');
  
  const span = getWordSpan(state.currentWordIndex);
  if (span) {
    span.classList.remove('creep', 'shadow-curr');
    span.classList.add('current');
  }
  
  movePillToWord(state.currentWordIndex);
}

function movePillToWord(idx) {
  const pill = dom.highlightPill;
  if (!pill) return;
  
  const span = getWordSpan(idx);
  if (!span) {
    pill.style.opacity = '0';
    return;
  }
  
  const contentRect = dom.teleprompterContent.getBoundingClientRect();
  const spanRect    = span.getBoundingClientRect();
  const PH = 6; // Horizontal padding
  const PV = 2; // Vertical padding
  
  // Calculate relative coordinates in viewport space
  let left = spanRect.left - contentRect.left;
  if (state.settings.mirror) {
    // Reverse bounds for horizontal mirroring
    left = contentRect.width - left - spanRect.width;
  }
  
  pill.style.top    = (spanRect.top  - contentRect.top  - PV) + 'px';
  pill.style.left   = (left - PH) + 'px';
  pill.style.width  = (spanRect.width  + PH * 2) + 'px';
  pill.style.height = (spanRect.height + PV * 2) + 'px';
  
  if (state.wordCount > 0) {
    pill.style.opacity = '1';
  }
}

function scrollToCurrent(smooth) {
  const span = getWordSpan(state.currentWordIndex);
  if (!span) return;

  const scrollEl      = dom.teleprompterScroll;
  const containerRect = scrollEl.getBoundingClientRect();
  const spanRect      = span.getBoundingClientRect();

  // Scroll to position target word at 38% height from top of viewport
  const targetY  = containerRect.height * 0.38;
  const currentY = spanRect.top - containerRect.top + spanRect.height / 2;
  const delta    = currentY - targetY;

  if (Math.abs(delta) < 1) return;

  _scrollTarget = _scrollCurrent - delta;

  if (!smooth) {
    _scrollCurrent = _scrollTarget;
    _applyTransformRaw(_scrollCurrent);
  } else {
    startScrollAnim();
  }
}

// ═══════════════════════════════════════════════════════
// IPC & STATE RECEIVERS
// ═══════════════════════════════════════════════════════
function setupIpcListeners() {
  // Listen for initial script loading
  window.electronAPI.onControlEvent((action, data) => {
    if (action === 'load-script') {
      state.settings = data.settings;
      renderScript(data.paragraphs, data.allWords);
      
      if (data.currentState) {
        state.isRecording = data.currentState.isRecording;
        state.isAdLibbing = data.currentState.isAdLibbing;
        state.lastSpeechTime = data.currentState.lastSpeechTime;
        state.shadowIndex = data.currentState.shadowIndex;
        
        // sync scroll position without animation
        snapTo(data.currentState.currentWordIndex, false);
        
        // update UI buttons to match recording state
        if (state.isRecording) {
          dom.overlayPlayBtn.textContent = '⏸️ Pause';
          dom.overlayPlayBtn.classList.remove('btn-primary');
          dom.overlayPlayBtn.classList.add('btn-secondary');
          dom.overlayContainer.classList.add('listening');
          startScrollAnim();
        } else {
          dom.overlayPlayBtn.textContent = '🎙️ Start';
          dom.overlayPlayBtn.classList.remove('btn-secondary');
          dom.overlayPlayBtn.classList.add('btn-primary');
          dom.overlayContainer.classList.remove('listening');
        }
        
        if (state.isAdLibbing) {
          dom.overlayContainer.classList.add('ad-libbing');
        }
      }
    }
  });

  // Listen for state sync updates from Control Panel
  window.electronAPI.onStateUpdated((update) => {
    state.settings.wpm = update.wpm;
    state.settings.fontSize = update.fontSize;
    state.settings.mirror = update.mirror;
    state.lastSpeechTime = update.lastSpeechTime;
    
    state.settings.overlayTheme = update.overlayTheme;
    state.settings.overlayHighlight = update.overlayHighlight;
    state.settings.overlayFont = update.overlayFont;
    state.settings.overlayAlign = update.overlayAlign;
    state.settings.overlayOpacity = update.overlayOpacity;
    state.settings.overlayBlur = update.overlayBlur;
    
    state.settings.overlayHidden = update.overlayHidden;
    state.settings.overlayFaded = update.overlayFaded;
    
    // Toggle recording states
    const wasRecording = state.isRecording;
    state.isRecording = update.isRecording;
    
    if (state.isRecording) {
      dom.overlayPlayBtn.textContent = '⏸️ Pause';
      dom.overlayPlayBtn.classList.remove('btn-primary');
      dom.overlayPlayBtn.classList.add('btn-secondary');
      dom.overlayContainer.classList.add('listening');
    } else {
      dom.overlayPlayBtn.textContent = '🎙️ Start';
      dom.overlayPlayBtn.classList.remove('btn-secondary');
      dom.overlayPlayBtn.classList.add('btn-primary');
      dom.overlayContainer.classList.remove('listening');
    }
    
    state.isAdLibbing = !!update.isAdLibbing;
    if (state.isAdLibbing) {
      dom.overlayContainer.classList.add('ad-libbing');
    } else {
      dom.overlayContainer.classList.remove('ad-libbing');
    }

    state.creepTargetIndex = update.creepTargetIndex;
    state.shadowIndex = update.shadowIndex;

    const smooth = update.scrollMode === 'snap';
    
    // Smart Sync Snapping:
    // 1. If it's an instant scroll mode (e.g. initial load, reset), we always snap.
    // 2. If the confirmed index is past our current index, we snap (progress catchup or forward jump).
    // 3. If the confirmed index went backward relative to the last confirmed index we saw,
    //    we snap (manual seek backward).
    // Otherwise, we do not snap (let speculative creep continue smoothly).
    const isManualSeekBackward = update.currentWordIndex < state.lastConfirmedWordIndex;
    const isForwardCatchup = update.currentWordIndex > state.currentWordIndex;
    
    state.lastConfirmedWordIndex = update.currentWordIndex;

    if (update.scrollMode === 'instant') {
      resetPosition();
      snapTo(update.currentWordIndex, false);
    } else if (isForwardCatchup || isManualSeekBackward) {
      snapTo(update.currentWordIndex, smooth);
    } else {
      updateShadowCursor();
    }

    updateVisualSettings();

    // Start/Ensure animation loop is running if recording
    if (state.isRecording && !wasRecording) {
      startScrollAnim();
    }
  });
}

// ═══════════════════════════════════════════════════════
// INTERACTIVE BUTTON BINDINGS
// ═══════════════════════════════════════════════════════
function bindButtons() {
  dom.overlayPlayBtn.onclick = () => {
    window.electronAPI.controlAction('toggle-pause');
  };


  dom.overlayCloseBtn.onclick = () => {
    window.electronAPI.controlAction('close-overlay');
  };
}

// Renderer dragging variables for custom drag
let isDragging = false;
let startX = 0;
let startY = 0;

function setupDragEvents() {
  const dragHandle = dom.overlayDragHandle;
  if (!dragHandle) return;

  dragHandle.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.screenX;
    startY = e.screenY;
    
    // Explicitly disable click-through while dragging
    window.electronAPI.setClickThrough(false);
  });

  window.addEventListener('mousemove', (e) => {
    if (isDragging) {
      const deltaX = e.screenX - startX;
      const deltaY = e.screenY - startY;
      startX = e.screenX;
      startY = e.screenY;
      window.electronAPI.controlAction('drag-window', { deltaX, deltaY });
    }
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
    }
  });
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  setupIpcListeners();
  bindButtons();
  setupDragEvents();
  
  // Kick off initial transform draw
  _applyTransformRaw(0);
});
