import os
import io
import asyncio
from typing import List, Literal, AsyncGenerator

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# --- INICIALISASI GROQ & FASTAPI ---
try:
    # Menggunakan Environment Variable GROQ_API_KEY yang harus diset di Vercel
    GROQ_CLIENT = Groq()
except Exception as e:
    print(f"ERROR: Groq client failed to initialize. Please check GROQ_API_KEY. Detail: {e}")
    GROQ_CLIENT = None

app = FastAPI(
    title="Masbro/Systa Groq API Backend",
    description="FastAPI backend untuk layanan Chatbot, Transkripsi, dan Vision.",
    version="1.0.0"
)

# --- KONFIGURASI CORS (UNTUK FIX KONEKSI) ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SKEMA DATA Pydantic ---
Role = Literal['user', 'assistant', 'system']

class ApiMessage(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[ApiMessage]
    model: str

class VisionRequest(BaseModel):
    prompt: str
    image_url: str
    model: str
# ----------------------------------------

# --- ENDPOINT UTAMA: CHAT STREAMING (/api/chat) ---

async def chat_generator(messages: List[ApiMessage], model_id: str) -> AsyncGenerator[str, None]:
    if not GROQ_CLIENT:
        yield "[ERROR]: Groq client not initialized. Check GROQ_API_KEY in Vercel environment variables."
        return

    groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

    try:
        stream = GROQ_CLIENT.chat.completions.create(
            messages=groq_messages,
            model=model_id,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                
    except Exception as e:
        yield f"\n[ERROR GROQ STREAMING]: Gagal memanggil API Groq. Detail: {e}"
        print(f"Groq Chat Streaming Error: {e}")

@app.post("/api/chat", response_class=StreamingResponse)
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(chat_generator(request.messages, request.model), media_type="text/plain")

# --- ENDPOINT: SPEECH TO TEXT (/api/transcribe) ---

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...), 
    model: str = "whisper-large-v3-turbo"
):
    if not GROQ_CLIENT:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Groq client not ready. Check API Key.")
    
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File audio harus diunggah.")

    # Groq SDK membutuhkan object file-like. Kita gunakan io.BytesIO dari isi file
    try:
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = file.filename # Nama file diperlukan oleh Groq

        transcription = GROQ_CLIENT.audio.transcriptions.create(
            file=audio_stream,
            model=model,
            response_format="text",
        )
        
        return {"text": transcription}

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal memproses file transkripsi: {e}")

# --- ENDPOINT: IMAGE & VISION (VISION TIDAK DIUJI DI FRONTEND SAAT INI) ---

@app.post("/api/chat-vision")
async def chat_vision(request: VisionRequest):
    if not GROQ_CLIENT:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Groq client not ready. Check API Key.")
        
    try:
        completion = GROQ_CLIENT.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {"type": "image_url", "image_url": {"url": request.image_url}}
                    ]
                }
            ],
            stream=False,
        )
        
        return {"text": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Groq Vision API error: {e}")


# --- HEALTH CHECK ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI Groq Backend is fully integrated and running."}
