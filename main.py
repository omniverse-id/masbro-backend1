import asyncio
import time
from typing import List, Literal, AsyncGenerator

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Masbro/Systa AI Backend",
    description="FastAPI backend untuk layanan Chatbot dan Transkripsi Audio.",
    version="1.0.0"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Role = Literal['user', 'assistant', 'system']

class ApiMessage(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[ApiMessage]
    model: str

async def chat_generator(user_prompt: str, model_id: str) -> AsyncGenerator[str, None]:
    
    response_text = (
        f"Halo! Saya adalah {model_id.split('/')[-1]} yang Anda pilih. "
        f"Backend Anda sekarang mengizinkan CORS. Anda bertanya: '{user_prompt}'.\n\n"
        "Integrasi streaming berhasil! "
        "Anda dapat mengganti logika ini dengan panggilan API LLM Anda yang sebenarnya "
        "menggunakan SDK Groq atau lainnya."
    )

    for chunk in response_text.split(" "):
        yield chunk + " "
        await asyncio.sleep(0.03)
        
    yield "\n\n--- Streaming Selesai ---"


@app.post("/api/chat", response_class=StreamingResponse)
async def chat_endpoint(request: ChatRequest):
    
    user_prompt = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), "Tidak ada prompt ditemukan.")
    
    if not request.model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model ID tidak boleh kosong.")

    return StreamingResponse(chat_generator(user_prompt, request.model), media_type="text/plain")


@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...), 
    model: str = "whisper-large-v3-turbo"
):
    
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File audio harus diunggah.")
    
    allowed_mime_types = ["audio/mpeg", "audio/wav", "audio/flac", "audio/m4a"]
    if file.content_type not in allowed_mime_types:
        
        if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipe file audio tidak didukung ({file.content_type}). Harap unggah MP3, WAV, FLAC, atau M4A."
            )

    try:
        
        file_contents = await file.read()
        file_size_kb = len(file_contents) / 1024
        
        simulated_text = (
            f"Transkripsi audio berhasil menggunakan {model}. "
            f"File: {file.filename} ({file_size_kb:.2f} KB). "
            "Teks simulasi: 'Backend Anda sekarang siap menerima file audio untuk transkripsi.'"
        )

        return {"text": simulated_text}

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gagal memproses file transkripsi.")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI Vercel Backend (with CORS enabled) is running."}
