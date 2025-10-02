# main.py
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from groq import Groq

# 1. Muat variabel lingkungan
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Inisialisasi FastAPI
app = FastAPI(
    title="Groq Chatbot Backend",
    description="Backend FastAPI untuk streaming respons chatbot dari Groq API menggunakan SSE."
)

# 3. Konfigurasi CORS
# Anda harus mengganti "http://localhost:3000" dengan URL frontend React Anda
origins = [
    "http://localhost:3000", 
    "https://your-frontend-domain.vercel.app", # Contoh URL Vercel produksi
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Inisialisasi Groq Client
if not GROQ_API_KEY:
    # Jika kunci tidak ditemukan, berikan error saat startup
    print("WARNING: GROQ_API_KEY tidak ditemukan. Harap atur di file .env")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error saat inisialisasi Groq client: {e}")
        groq_client = None


# 5. Endpoint Streaming Chat
@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Menerima pesan, memanggil Groq API, dan mengembalikan respons secara streaming menggunakan SSE.
    """
    if not groq_client:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Server API key not configured."}
        )

    try:
        # Asumsi frontend mengirim JSON dengan kunci 'message' dan opsional 'model'
        data = await request.json()
        user_message = data.get("message")
        
        # Menggunakan model default Mixtral yang cepat, atau ambil dari request
        model_name = data.get("model", "mixtral-8x7b-32768") 
        
        if not user_message:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Pesan ('message') tidak boleh kosong."}
            )

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Request body harus JSON yang valid."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Kesalahan internal server: {e}"}
        )


    # Generator untuk menghasilkan chunk respons Groq sebagai event SSE
    async def groq_response_generator():
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Anda adalah asisten AI yang ramah, ringkas, dan sangat cepat."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True, # Penting untuk streaming
            )
            
            # Kirim setiap potongan konten yang masuk
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    # Setiap 'yield' akan menjadi 'data:' dalam event SSE
                    yield content
                    
        except Exception as e:
            # Kirim event error (ini memerlukan penanganan khusus di frontend)
            error_msg = f"API Error: {str(e)}"
            print(error_msg)
            # Karena ini EventSourceResponse, kita hanya bisa mengirim string data
            yield json.dumps({"error": error_msg})


    # Menggunakan EventSourceResponse untuk mengirimkan data streaming (SSE)
    # Media type text/event-stream adalah standar untuk SSE
    return EventSourceResponse(groq_response_generator(), media_type="text/event-stream")


# 6. Endpoint Health Check
@app.get("/")
def read_root():
    """Endpoint sederhana untuk memeriksa apakah server berjalan."""
    return {"status": "ok", "message": "Groq Chatbot Backend berjalan dengan baik di FastAPI."}