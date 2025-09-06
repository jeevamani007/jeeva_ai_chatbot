from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import traceback
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio


# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("AIzaSyCic8d5-4Iwa04e9_DLRWae4dxtDNFnPkI") or "AIzaSyCic8d5-4Iwa04e9_DLRWae4dxtDNFnPkI"
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ChatMessage(BaseModel):
    message: str
    history: list = []

class ChatResponse(BaseModel):
    response: str
    history: list

conversation_history = {}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "gemini_key_set": bool(GEMINI_API_KEY)})

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_data: ChatMessage):
    try:
        user_message = chat_data.message
        session_id = "default"

        if session_id not in conversation_history:
            conversation_history[session_id] = []

        conversation_history[session_id].append(f"You: {user_message}")
        prompt = "\n".join(conversation_history[session_id]) + "\nAssistant:"

        # Simulate typing delay
        await asyncio.sleep(0.3)

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        reply = response.text

        conversation_history[session_id].append(f"Assistant: {reply}")

        return ChatResponse(response=reply, history=conversation_history[session_id])

    except Exception as e:
        print("Chat Error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

@app.delete("/chat/clear")
async def clear_chat():
    global conversation_history
    conversation_history = {}
    return {"message": "Chat history cleared"}
