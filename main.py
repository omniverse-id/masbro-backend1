import os
import io
import json
from typing import List, Literal, AsyncGenerator, Union, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# --- INICIALISASI GROQ & FASTAPI ---
try:
    GROQ_CLIENT = Groq()
except Exception:
    GROQ_CLIENT = None

app = FastAPI(
    title="Masbro/Systa Groq API Backend",
    description="FastAPI backend untuk layanan Chatbot, Transkripsi, dan Vision.",
    version="1.0.0"
)

# --- KONFIGURASI CORS ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SKEMA DATA Pydantic (FIX 422 ERROR) ---
Role = Literal['user', 'assistant', 'system']
ReasoningEffort = Literal['none', 'default', 'low', 'medium', 'high']

class ApiMessage(BaseModel):
    role: Role
    # FIX: Mengizinkan string (teks) ATAU List of Dictionaries (multimodal)
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    messages: List[ApiMessage]
    model: str
    # BARU: Menambahkan parameter reasoning_effort (optional)
    reasoning_effort: Optional[ReasoningEffort] = None
# ----------------------------------------

# --- FUNGSI UTILITY ---
def format_messages_for_groq(messages: List[ApiMessage]) -> List[Dict[str, Any]]:
    groq_messages = []
    for msg in messages:
        # Pydantic sudah memastikan msg.content adalah str atau List[Dict]
        groq_messages.append({"role": msg.role, "content": msg.content})
    return groq_messages

# --- ENDPOINT UTAMA: CHAT STREAMING (/api/chat) ---

async def chat_generator(messages: List[ApiMessage], model_id: str, reasoning_effort: Optional[ReasoningEffort]) -> AsyncGenerator[str, None]:
    if not GROQ_CLIENT:
        yield "[ERROR]: Groq client not initialized. Check GROQ_API_KEY in Vercel environment variables."
        return

    groq_messages = format_messages_for_groq(messages)
    
    # Menyiapkan parameter API Groq
    groq_params = {
        "messages": groq_messages,
        "model": model_id,
        "stream": True,
    }

    # BARU: Menambahkan reasoning_effort jika disediakan
    if reasoning_effort:
        groq_params["reasoning_effort"] = reasoning_effort

    try:
        # Memanggil Groq API dengan parameter yang dimodifikasi
        stream = GROQ_CLIENT.chat.completions.create(**groq_params)

        for chunk in stream:
            content = chunk.choices[0].delta.content
            # Pada streaming Groq, konten reasoning untuk GPT-OSS biasanya 
            # digabungkan ke 'content'. Kita hanya mengeluarkan konten.
            if content:
                yield content
                
    except Exception as e:
        yield f"\n[ERROR GROQ STREAMING]: Gagal memanggil API Groq. Detail: {e}"
        print(f"Groq Chat Streaming Error: {e}")

@app.post("/api/chat", response_class=StreamingResponse)
async def chat_endpoint(request: ChatRequest):
    # Mengirimkan parameter baru ke chat_generator
    return StreamingResponse(chat_generator(request.messages, request.model, request.reasoning_effort), media_type="text/plain")

# --- ENDPOINT: IMAGE & VISION (/api/chat-vision) ---

@app.post("/api/chat-vision")
async def chat_vision(request: ChatRequest):
    if not GROQ_CLIENT:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Groq client not ready. Check API Key.")
        
    groq_messages = format_messages_for_groq(request.messages)
    
    is_gpt_oss = "gpt-oss" in request.model.lower()
        
    # Menyiapkan parameter API Groq untuk Vision (Non-Streaming)
    groq_params = {
        "messages": groq_messages,
        "model": request.model,
        "stream": False,
        # Menambahkan include_reasoning=True jika model adalah GPT-OSS
        "include_reasoning": is_gpt_oss
    }

    # Menambahkan reasoning_effort jika disediakan
    if request.reasoning_effort:
        groq_params["reasoning_effort"] = request.reasoning_effort
        
    try:
        completion = GROQ_CLIENT.chat.completions.create(**groq_params)
        
        main_content = completion.choices[0].message.content
        reasoning_content = None
        
        # LOGIKA BARU: Ekstraksi reasoning khusus untuk GPT-OSS (non-streaming)
        if is_gpt_oss and completion.choices[0].message:
            # Menggunakan getattr untuk akses aman ke atribut 'reasoning'
            reasoning_content = getattr(completion.choices[0].message, 'reasoning', None)
        
        # Menggabungkan reasoning (jika ada) dan konten utama
        if reasoning_content:
            # Format Markdown untuk Reasoning Card di frontend
            full_response = f"""**Thinking Process:**
