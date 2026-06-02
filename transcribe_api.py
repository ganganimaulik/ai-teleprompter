import io
import modal

# Define the container image with faster-whisper, FastAPI, and NVIDIA CUDA libraries
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "faster-whisper",
        "fastapi",
        "python-multipart",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12"
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib"})
)

app = modal.App("teleprompter-asr")

@app.cls(
    gpu="T4",               # Cheap and fast GPU
    image=image,
    max_containers=5,       # Auto-scale upper limit
    min_containers=0,       # Spin down to 0 when idle to minimize costs
)
class Transcriber:
    @modal.enter()
    def load_model(self):
        from faster_whisper import WhisperModel
        # Load Distil-Whisper into GPU memory once during container boot
        self.model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware

        web_app = FastAPI()

        # Add CORS middleware to allow cross-origin requests from Electron/browser client
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/")
        async def transcribe(file: UploadFile = File(...)):
            # Read the raw WAV file bytes
            audio_bytes = await file.read()
            audio_file = io.BytesIO(audio_bytes)
            
            # Transcribe audio segment (beam_size=1 is fast and sufficient for clear presentation speech)
            segments, info = self.model.transcribe(audio_file, beam_size=1, language="en")
            
            text = " ".join([segment.text for segment in segments]).strip()
            return {"text": text}

        return web_app
