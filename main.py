# main.py
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from groq.types.shared_types import Content
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# 1. Muat variabel lingkungan
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Inisialisasi FastAPI
app = FastAPI(
    title="Groq Full API Capabilities Backend",
    description="Backend FastAPI mencakup Chat, Vision, Speech-to-Text, Structured Output, Files, dan Fine-Tuning."
)

# 3. Konfigurasi CORS
origins = ["http://localhost:3000"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Inisialisasi Groq Client
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY tidak ditemukan. Harap atur di file .env")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error saat inisialisasi Groq client: {e}")
        groq_client = None

# Dependency Injection untuk Client
def get_groq_client():
    """Helper untuk mendapatkan client dan menangani error jika tidak terinisialisasi."""
    if not groq_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key not configured or failed to initialize."
        )
    return groq_client


# ----------------------------------------------------
# I. CHAT, VISION, & STRUCTURED OUTPUT (Sudah ada di jawaban sebelumnya)
# ----------------------------------------------------

# (Kode untuk /api/chat, /api/chat-vision, /api/structured-chat, /api/transcribe, /api/models, /api/usage ditiadakan di sini untuk menghemat ruang, tetapi *asumsikan* itu ada di file lengkap Anda.)

# Model Pydantic untuk Structured Output (Contoh)
class ExtractedData(BaseModel):
    product_name: str
    rating: float
    sentiment: Literal["positive", "negative", "neutral"]
    key_features: List[str]

# ----------------------------------------------------
# II. FILE MANAGEMENT (Prasyarat Fine-Tuning)
# ----------------------------------------------------

@app.post("/api/files")
async def upload_file_for_fine_tuning(
    file: UploadFile = File(...),
    purpose: str = Form("fine-tune"), # Groq API mungkin memerlukan tujuan
    client: Groq = Depends(get_groq_client)
):
    """
    Meng-*upload* file (misalnya JSONL) untuk digunakan dalam fine-tuning.
    Ref: /files (Tidak ada di referensi Anda, tetapi merupakan prasyarat)
    """
    try:
        groq_file = client.files.create(
            file=file.file,
            purpose=purpose
        )
        return JSONResponse(content=groq_file.model_dump(), status_code=status.HTTP_201_CREATED)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal upload file: {str(e)}")

@app.get("/api/files")
async def list_files(client: Groq = Depends(get_groq_client)):
    """Melihat semua file yang telah di-*upload*."""
    try:
        files = client.files.list()
        return {"files": [f.model_dump() for f in files.data]}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal mengambil daftar file: {str(e)}")

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str, client: Groq = Depends(get_groq_client)):
    """Menghapus file berdasarkan ID."""
    try:
        result = client.files.delete(file_id)
        return {"id": file_id, "deleted": result.deleted}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal menghapus file: {str(e)}")

# ----------------------------------------------------
# III. FINE TUNING MANAGEMENT
# ----------------------------------------------------

class FineTuningCreate(BaseModel):
    """Skema untuk request POST /fine_tunings"""
    input_file_id: str = Field(description="ID dari file yang telah di-*upload*.")
    name: str = Field(description="Nama untuk model fine-tuning ini.")
    base_model: str = Field(description="Model dasar (base model) yang akan di-fine tune.")
    type: Literal["lora"] = Field("lora", description="Jenis fine-tuning, saat ini hanya 'lora'.")

@app.post("/api/fine_tunings")
async def create_fine_tuning_job(
    job_data: FineTuningCreate,
    client: Groq = Depends(get_groq_client)
):
    """
    Membuat pekerjaan fine-tuning baru.
    Ref: POST /fine_tunings
    """
    try:
        job = client.fine_tunings.create(
            input_file_id=job_data.input_file_id,
            name=job_data.name,
            base_model=job_data.base_model,
            type=job_data.type
        )
        return JSONResponse(content=job.model_dump(), status_code=status.HTTP_201_CREATED)
    except Exception as e:
        # Menangkap error jika file_id tidak valid, base_model salah, dll.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal membuat pekerjaan fine-tuning: {str(e)}")

@app.get("/api/fine_tunings")
async def list_fine_tuning_jobs(client: Groq = Depends(get_groq_client)):
    """
    Melihat semua pekerjaan fine-tuning yang pernah dibuat.
    Ref: GET /fine_tunings
    """
    try:
        jobs = client.fine_tunings.list()
        return {"jobs": [job.model_dump() for job in jobs.data]}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal mengambil daftar fine-tuning: {str(e)}")

@app.get("/api/fine_tunings/{job_id}")
async def get_fine_tuning_job(job_id: str, client: Groq = Depends(get_groq_client)):
    """
    Mengambil status detail pekerjaan fine-tuning tertentu.
    Ref: GET /fine_tunings/{id}
    """
    try:
        job = client.fine_tunings.get(job_id)
        return job.model_dump()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal mengambil detail fine-tuning {job_id}: {str(e)}")

@app.delete("/api/fine_tunings/{job_id}")
async def delete_fine_tuning_job(job_id: str, client: Groq = Depends(get_groq_client)):
    """
    Menghapus pekerjaan fine-tuning tertentu.
    Ref: DELETE /fine_tunings/{id}
    """
    try:
        result = client.fine_tunings.delete(job_id)
        return {"id": job_id, "deleted": result.deleted}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal menghapus fine-tuning {job_id}: {str(e)}")

# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------

@app.get("/")
def read_root():
    """Endpoint sederhana untuk memeriksa apakah server berjalan."""
    return {"status": "ok", "message": "Groq Full API Capabilities Backend berjalan dengan baik."}