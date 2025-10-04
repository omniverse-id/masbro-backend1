import os
import io
import json
from typing import List, Literal, AsyncGenerator, Union, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

try:
    GROQ_CLIENT = Groq()
except Exception:
    GROQ_CLIENT = None

app = FastAPI(
    title="Masbro/Systa Groq API Backend",
    description="FastAPI backend untuk layanan Chatbot, Transkripsi, dan Vision.",
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
ReasoningEffort = Literal['none', 'default', 'low', 'medium', 'high']

class ApiMessage(BaseModel):
    role: Role
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    messages: List[ApiMessage]
    model: str
    reasoning_effort: Optional[ReasoningEffort] = None

def format_messages_for_groq(messages: List[ApiMessage]) -> List[Dict[str, Any]]:
    groq_messages = []
    for msg in messages:
        groq_messages.append({"role": msg.role, "content": msg.content})
    return groq_messages

async def chat_generator(messages: List[ApiMessage], model_id: str, reasoning_effort: Optional[ReasoningEffort]) -> AsyncGenerator[str, None]:
    if not GROQ_CLIENT:
        yield "[ERROR]: Groq client not initialized. Check GROQ_API_KEY in Vercel environment variables."
        return

    groq_messages = format_messages_for_groq(messages)
    
    groq_params = {
        "messages": groq_messages,
        "model": model_id,
        "stream": True,
    }

    if reasoning_effort:
        groq_params["reasoning_effort"] = reasoning_effort

    try:
        stream = GROQ_CLIENT.chat.completions.create(**groq_params)

        for chunk in stream:
            content = chunk.choices[0].delta.content
<<<<<<< HEAD
=======
            # Pada streaming Groq, konten reasoning untuk GPT-OSS biasanya 
            # digabungkan ke 'content'. Kita hanya mengeluarkan konten.
>>>>>>> 4502fff5e78c8b0c5abbb4ce395c542692f12c30
            if content:
                yield content
                
    except Exception as e:
        yield f"\n[ERROR GROQ STREAMING]: Gagal memanggil API Groq. Detail: {e}"
        print(f"Groq Chat Streaming Error: {e}")

@app.post("/api/chat", response_class=StreamingResponse)
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(chat_generator(request.messages, request.model, request.reasoning_effort), media_type="text/plain")

@app.post("/api/chat-vision")
async def chat_vision(request: ChatRequest):
    if not GROQ_CLIENT:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Groq client not ready. Check API Key.")
        
    groq_messages = format_messages_for_groq(request.messages)
    
    is_gpt_oss = "gpt-oss" in request.model.lower()
        
    groq_params = {
        "messages": groq_messages,
        "model": request.model,
        "stream": False,
<<<<<<< HEAD
        "include_reasoning": is_gpt_oss
    }

=======
        # Menambahkan include_reasoning=True jika model adalah GPT-OSS
        "include_reasoning": is_gpt_oss
    }

    # Menambahkan reasoning_effort jika disediakan
>>>>>>> 4502fff5e78c8b0c5abbb4ce395c542692f12c30
    if request.reasoning_effort:
        groq_params["reasoning_effort"] = request.reasoning_effort
        
    try:
        completion = GROQ_CLIENT.chat.completions.create(**groq_params)
        
        main_content = completion.choices[0].message.content
        reasoning_content = None
        
<<<<<<< HEAD
        if is_gpt_oss and completion.choices[0].message:
            raw_reasoning = getattr(completion.choices[0].message, 'reasoning', None)
            
            if raw_reasoning and isinstance(raw_reasoning, str):
                reasoning_content = raw_reasoning
        
        if reasoning_content:
            full_response = f"""**Thinking Process:**
=======
        # LOGIKA BARU: Ekstraksi reasoning khusus untuk GPT-OSS (non-streaming)
        if is_gpt_oss and completion.choices[0].message:
            # Menggunakan getattr untuk akses aman ke atribut 'reasoning'
            reasoning_content = getattr(completion.choices[0].message, 'reasoning', None)
        
        # Menggabungkan reasoning (jika ada) dan konten utama
        if reasoning_content:
            # Format Markdown untuk Reasoning Card di frontend
            full_response = f"""**Thinking Process:**
>>>>>>> 4502fff5e78c8b0c5abbb4ce395c542692f12c30
