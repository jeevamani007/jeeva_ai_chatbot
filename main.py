from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import traceback
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyCic8d5-4Iwa04e9_DLRWae4dxtDNFnPkI"
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment")

genai.configure(api_key=GEMINI_API_KEY)

# System prompt (always applied)
SYSTEM_PROMPT = """
You are Jeeva's AI assistant powered by Gemini.
Always respond politely, provide helpful answers, and keep answers concise.
If the user asks about the date or time, respond with the current date/time.
"""

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ChatMessage(BaseModel):
    message: str
    history: list = []

class ChatResponse(BaseModel):
    response: str
    history: list
    session_id: str

# Conversation storage
conversation_history = {}
MAX_HISTORY = 15

# Root route
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Health check
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "gemini_key_set": bool(GEMINI_API_KEY)})

# Chat route
@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_data: ChatMessage):
    try:
        user_message = chat_data.message

        # Multi-user session
        session_id = request.cookies.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Save user message
        conversation_history[session_id].append(f"You: {user_message}")
        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY:]

        # Handle date/time request locally
        if "date" in user_message.lower() or "time" in user_message.lower():
            now = datetime.now()
            reply = f"Today is {now.strftime('%B %d, %Y')} and the time is {now.strftime('%H:%M:%S')}."
            conversation_history[session_id].append(f"Assistant: {reply}")
        else:
            # Prompt for Gemini
            prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(conversation_history[session_id]) + "\nAssistant:"
            await asyncio.sleep(0.3)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await asyncio.to_thread(model.generate_content, prompt)
            reply = response.text
            conversation_history[session_id].append(f"Assistant: {reply}")

        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY:]

        response_data = ChatResponse(
            response=reply,
            history=conversation_history[session_id],
            session_id=session_id
        )
        json_response = JSONResponse(content=response_data.dict())
        json_response.set_cookie(key="session_id", value=session_id)
        return json_response

    except Exception as e:
        print("Chat Error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

# Clear chat
@app.delete("/chat/clear")
async def clear_chat():
    global conversation_history
    conversation_history = {}
    return {"message": "Chat history cleared"}
