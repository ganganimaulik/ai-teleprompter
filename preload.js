const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // State Sync
  syncState: (state) => ipcRenderer.send('sync-state', state),
  onStateUpdated: (callback) => {
    const subscription = (event, state) => callback(state);
    ipcRenderer.on('state-updated', subscription);
    return () => ipcRenderer.removeListener('state-updated', subscription);
  },

  // Mouse Passthrough (for overlay click-through)
  setClickThrough: (ignore, options) => ipcRenderer.send('set-click-through', ignore, options),

  // UI Actions
  controlAction: (action, data) => ipcRenderer.send('control-action', action, data),
  onControlEvent: (callback) => {
    const subscription = (event, action, data) => callback(action, data);
    ipcRenderer.on('control-event', subscription);
    return () => ipcRenderer.removeListener('control-event', subscription);
  },

  // Permissions & Shortcuts
  requestMicrophone: () => ipcRenderer.invoke('request-microphone-permission'),
  callVertexAPI: (options) => ipcRenderer.invoke('call-vertex-api', options),
  onHotkeyTriggered: (callback) => {
    const subscription = (event, hotkey) => callback(hotkey);
    ipcRenderer.on('hotkey-triggered', subscription);
    return () => ipcRenderer.removeListener('hotkey-triggered', subscription);
  }
});
