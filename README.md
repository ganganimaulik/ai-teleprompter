# AI Teleprompter

A desktop AI teleprompter app built with Electron. The main window stays hidden from screen sharing and recording software. It uses microphone access to listen to your voice and align the script scrolling in real-time, making presentations and recordings seamless.

## Features
- **Invisible to Screen Sharing**: Built to be hidden from screen recorders and sharing applications.
- **AI-driven Voice Alignment**: Automatically scrolls the script as you speak.
- **AI Feature Processing**: Select multiple AI processing options for your script content.

## Getting Started

### Prerequisites
- Node.js & npm

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

### Running the App
Run the app in development mode:
```bash
npm start
```

### Building for Production
To build the macOS DMG artifact:
```bash
npm run dist
```
