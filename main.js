const { app, BrowserWindow, ipcMain, globalShortcut, systemPreferences } = require('electron');
app.commandLine.appendSwitch('enable-unsafe-webgpu');
const path = require('path');
const http = require('http');
const fs = require('fs');
const crypto = require('crypto');


try {
  require('electron-reloader')(module);
} catch (_) {}

let server = null;
let serverPort = 0;
let controlWindow = null;
let overlayWindow = null;

// Enforce single instance lock
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (controlWindow) {
      if (controlWindow.isMinimized()) controlWindow.restore();
      controlWindow.focus();
    }
  });
}

// ═══════════════════════════════════════════════════════
// LOCAL COOP/COEP HTTP SERVER
// ═══════════════════════════════════════════════════════
function startLocalServer() {
  const PREFERRED_PORT = 58273;

  return new Promise((resolve) => {
    const tryListen = (port) => {
      // Create a fresh server instance each time to ensure no state pollution
      const currentServer = http.createServer((req, res) => {
        // Basic route mappings
        let relativePath = req.url.split('?')[0];
        if (relativePath === '/' || relativePath === '/control') {
          relativePath = '/control.html';
        } else if (relativePath === '/overlay') {
          relativePath = '/overlay.html';
        }

        const filePath = path.join(__dirname, 'src', relativePath);

        fs.stat(filePath, (err, stats) => {
          if (err || !stats.isFile()) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('404 Not Found');
            return;
          }

          // MIME Types mapping
          const ext = path.extname(filePath).toLowerCase();
          let contentType = 'text/html';
          if (ext === '.js') contentType = 'application/javascript';
          else if (ext === '.css') contentType = 'text/css';
          else if (ext === '.json') contentType = 'application/json';
          else if (ext === '.png') contentType = 'image/png';
          else if (ext === '.svg') contentType = 'image/svg+xml';
          else if (ext === '.wasm') contentType = 'application/wasm';

          // Set COOP and COEP headers for SharedArrayBuffer
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
          res.setHeader('Access-Control-Allow-Origin', '*');

          res.writeHead(200, { 'Content-Type': contentType });
          fs.createReadStream(filePath).pipe(res);
        });
      });

      currentServer.once('error', (err) => {
        currentServer.close();
        if (err.code === 'EADDRINUSE') {
          console.warn(`Preferred port ${port} in use. Trying port 0 (OS auto-assign)...`);
          tryListen(0);
        } else {
          console.error('Server error:', err);
        }
      });

      currentServer.listen(port, '127.0.0.1', () => {
        server = currentServer;
        serverPort = currentServer.address().port;
        console.log(`Local web server listening on http://127.0.0.1:${serverPort}`);
        resolve(serverPort);
      });
    };

    tryListen(PREFERRED_PORT);
  });
}

// ═══════════════════════════════════════════════════════
// WINDOW CREATION
// ═══════════════════════════════════════════════════════
function createControlWindow() {
  controlWindow = new BrowserWindow({
    width: 950,
    height: 850,
    minWidth: 800,
    minHeight: 700,
    title: 'AI Teleprompter - Control Panel',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Enable Screen Recording protection for control panel
  controlWindow.setContentProtection(true);

  controlWindow.loadURL(`http://127.0.0.1:${serverPort}/control.html`);

  // Log renderer console messages (including worker logs) to the terminal
  controlWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    const levelStr = level === 0 ? 'INFO' : level === 1 ? 'WARN' : level === 2 ? 'ERROR' : 'LOG';
    console.log(`[Renderer ${levelStr}] (${path.basename(sourceId)}:${line}) ${message}`);
  });

  controlWindow.on('closed', () => {
    controlWindow = null;
    if (overlayWindow) {
      overlayWindow.close();
    }
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });
}

function createOverlayWindow() {
  if (overlayWindow) {
    overlayWindow.showInactive();
    return;
  }

  // Define transparent frameless window near the top of the screen
  const primaryDisplay = require('electron').screen.getPrimaryDisplay();
  const { width } = primaryDisplay.workAreaSize;

  const overlayWidth = Math.round(width * 0.7);
  const overlayHeight = 320;
  const overlayX = Math.round((width - overlayWidth) / 2);
  const overlayY = 40; // near the webcam at the top

  const overlayOptions = {
    width: overlayWidth,
    height: overlayHeight,
    x: overlayX,
    y: overlayY,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000', // fully transparent to prevent white flash
    show: false, // hidden initially until ready-to-show
    alwaysOnTop: true,
    hasShadow: false,
    resizable: true,
    focusable: false, // does not steal focus
    skipTaskbar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      backgroundThrottling: false,
    },
  };

  if (process.platform === 'darwin') {
    overlayOptions.type = 'panel';
  }

  overlayWindow = new BrowserWindow(overlayOptions);

  // Enable Screen Sharing & recording prevention for teleprompter overlay
  overlayWindow.setContentProtection(true);

  // For macOS, ensure it shows above fullscreen apps (like Google Slides in Chrome)
  if (process.platform === 'darwin') {
    overlayWindow.setVisibleOnAllWorkspaces(true, { 
      visibleOnFullScreen: true,
      skipTransformProcessType: true
    });
    overlayWindow.setAlwaysOnTop(true, 'screen-saver', 1);
  }

  // Initial mouse passthrough
  overlayWindow.setIgnoreMouseEvents(true, { forward: true });

  overlayWindow.loadURL(`http://127.0.0.1:${serverPort}/overlay.html`);

  overlayWindow.once('ready-to-show', () => {
    overlayWindow.showInactive();
  });

  overlayWindow.on('closed', () => {
    overlayWindow = null;
    // Tell control panel window overlay closed
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('control-event', 'overlay-closed');
    }
  });
}

// ═══════════════════════════════════════════════════════
// IPC CHANNELS & COMMUNICATION
// ═══════════════════════════════════════════════════════
function setupIpcHandlers() {
  // Sync state from Control Panel to Overlay
  ipcMain.on('sync-state', (event, data) => {
    if (overlayWindow && !overlayWindow.isDestroyed()) {
      overlayWindow.webContents.send('state-updated', data);
    }
  });

  // Mouse passthrough click-through toggle
  ipcMain.on('set-click-through', (event, ignore, options) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win) {
      win.setIgnoreMouseEvents(ignore, options);
    }
  });

  // Handle Control Panel Actions
  ipcMain.on('control-action', (event, action, data) => {
    if (action === 'open-overlay') {
      createOverlayWindow();
      // Wait for window to load before passing script text
      overlayWindow.webContents.once('did-finish-load', () => {
        overlayWindow.webContents.send('control-event', 'load-script', data);
      });
    } else if (action === 'close-overlay') {
      if (overlayWindow && !overlayWindow.isDestroyed()) {
        overlayWindow.close();
      }
    } else if (action === 'update-script') {
      if (overlayWindow && !overlayWindow.isDestroyed()) {
        overlayWindow.webContents.send('control-event', 'load-script', data);
      }
    } else if (action === 'seek-to-word') {
      if (controlWindow && !controlWindow.isDestroyed()) {
        controlWindow.webContents.send('control-event', 'seek-to-word', data);
      }
    } else if (action === 'toggle-pause') {
      if (controlWindow && !controlWindow.isDestroyed()) {
        controlWindow.webContents.send('hotkey-triggered', 'toggle-pause');
      }
    } else if (action === 'reset-position') {
      if (controlWindow && !controlWindow.isDestroyed()) {
        controlWindow.webContents.send('control-event', 'reset-position');
      }
    } else if (action === 'drag-window') {
      const win = BrowserWindow.fromWebContents(event.sender);
      if (win) {
        const [x, y] = win.getPosition();
        win.setPosition(x + data.deltaX, y + data.deltaY);
      }
    }
  });

  // macOS system level microphone permission check and request
  ipcMain.handle('request-microphone-permission', async () => {
    if (process.platform !== 'darwin') return true;
    
    const status = systemPreferences.getMediaAccessStatus('microphone');
    if (status === 'granted') return true;
    if (status === 'denied' || status === 'restricted') return false;

    // Prompt user for permission
    try {
      const granted = await systemPreferences.askForMediaAccess('microphone');
      return granted;
    } catch (err) {
      console.error('Failed to request microphone permission:', err);
      return false;
    }
  });

  // Vertex AI API Call Handler
  ipcMain.handle('call-vertex-api', async (event, { projectId, location, model, authMethod, apiKey, serviceAccountJson, payload }) => {
    try {
      const loc = location || 'us-central1';
      const host = loc === 'global' ? 'aiplatform.googleapis.com' : `${loc}-aiplatform.googleapis.com`;
      const endpoint = `https://${host}/v1/projects/${projectId}/locations/${loc}/publishers/google/models/${model}:generateContent`;

      let headers = {
        'Content-Type': 'application/json'
      };

      let url = endpoint;

      if (authMethod === 'api-key') {
        if (!apiKey) {
          throw new Error('API Key is missing for Vertex AI authentication.');
        }
        url = `${endpoint}?key=${apiKey}`;
      } else if (authMethod === 'service-account') {
        if (!serviceAccountJson) {
          throw new Error('Service Account JSON is missing for Vertex AI authentication.');
        }
        const accessToken = await getServiceAccountAccessToken(serviceAccountJson);
        headers['Authorization'] = `Bearer ${accessToken}`;
      } else {
        throw new Error(`Unsupported authentication method: ${authMethod}`);
      }

      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errText = await response.text();
        let errMsg = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errJson = JSON.parse(errText);
          if (errJson.error && errJson.error.message) {
            errMsg = errJson.error.message;
          }
        } catch (_) {}
        throw new Error(errMsg);
      }

      return await response.json();
    } catch (err) {
      console.error('Vertex AI Call error:', err);
      throw err; // Reject in renderer
    }
  });
}

async function getServiceAccountAccessToken(serviceAccountJson) {
  const sa = JSON.parse(serviceAccountJson);
  const header = {
    alg: 'RS256',
    typ: 'JWT'
  };
  const now = Math.floor(Date.now() / 1000);
  const claim = {
    iss: sa.client_email,
    scope: 'https://www.googleapis.com/auth/cloud-platform',
    aud: 'https://oauth2.googleapis.com/token',
    exp: now + 3600,
    iat: now
  };
  
  const base64UrlEncode = (obj) => {
    return Buffer.from(JSON.stringify(obj))
      .toString('base64')
      .replace(/=/g, '')
      .replace(/\+/g, '-')
      .replace(/\//g, '_');
  };
  
  const encodedHeader = base64UrlEncode(header);
  const encodedClaim = base64UrlEncode(claim);
  const signatureInput = `${encodedHeader}.${encodedClaim}`;
  
  const sign = crypto.createSign('RSA-SHA256');
  sign.update(signatureInput);
  const signature = sign.sign(sa.private_key, 'base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');
    
  const jwt = `${signatureInput}.${signature}`;
  
  const response = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`
  });
  
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OAuth token exchange failed: ${errText}`);
  }
  
  const data = await response.json();
  return data.access_token;
}

// ═══════════════════════════════════════════════════════
// APP LIFECYCLE
// ═══════════════════════════════════════════════════════
app.whenReady().then(async () => {
  await startLocalServer();
  setupIpcHandlers();
  createControlWindow();

  // Option+Space: Toggle play/pause
  globalShortcut.register('Option+Space', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'toggle-pause');
    }
  });

  // Option+Up: Nudge to previous sentence
  globalShortcut.register('Option+Up', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'nudge-up');
    }
  });

  // Option+Down: Nudge to next sentence
  globalShortcut.register('Option+Down', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'nudge-down');
    }
  });

  // Option+Escape: Close overlay
  globalShortcut.register('Option+Escape', () => {
    if (overlayWindow && !overlayWindow.isDestroyed()) {
      overlayWindow.close();
    }
  });

  // Option+[: Decrease WPM
  globalShortcut.register('Option+[', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'wpm-down');
    }
  });

  // Option+]: Increase WPM
  globalShortcut.register('Option+]', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'wpm-up');
    }
  });

  // Option+H: Toggle overlay visibility
  globalShortcut.register('Option+H', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'toggle-visibility');
    }
  });

  // Option+O: Toggle overlay opacity
  globalShortcut.register('Option+O', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'toggle-opacity');
    }
  });

  // Option+-: Decrease overlay opacity
  globalShortcut.register('Option+-', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'opacity-down');
    }
  });

  // Option+=: Increase overlay opacity
  globalShortcut.register('Option+=', () => {
    if (controlWindow && !controlWindow.isDestroyed()) {
      controlWindow.webContents.send('hotkey-triggered', 'opacity-up');
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createControlWindow();
    } else {
      if (controlWindow) {
        if (controlWindow.isMinimized()) {
          controlWindow.restore();
        } else {
          // If the overlay window is visible, don't steal focus/switch spaces
          const isOverlayActive = overlayWindow && !overlayWindow.isDestroyed() && overlayWindow.isVisible();
          if (!isOverlayActive) {
            controlWindow.show();
            controlWindow.focus();
          }
        }
      }
    }
  });
});

app.on('will-quit', () => {
  // Unregister all hotkeys
  globalShortcut.unregisterAll();
  
  if (server) {
    server.close();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
