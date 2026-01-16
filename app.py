from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import traceback
from memory_store import (
    get_user_state,
    get_user_memory,
    append_user_memory,
    clear_user_memory,
    clear_user_state,
    save_user_job
)
from OpenAi import chat_with_user

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------

app = FastAPI(title="AI Aura Career Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    sessionId: str

class JobSuggestion(BaseModel):
    title: str
    company: str
    location: str
    redirect_url: str

class ChatResponse(BaseModel):
    assistantText: str
    mode: str  # "chat" | "searching" | "results" | "no_results"
    jobs: List[JobSuggestion]
    debug: dict = {}  # optional, for dev

class UserCredentials(BaseModel):
    username: str
    password: str  # plaintext for now, but you should hash for production

class AuthResponse(BaseModel):
    sessionId: str
    message: str

# -------------------------------------------------------------------
# In-memory user store (demo only)
# -------------------------------------------------------------------
USER_STORE = {}  # username -> {"password": str, "session_id": str}

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/", summary="Health check endpoint")
async def health_check():
    return {"status": "ok"}

# -------------------------------------------------------------------
# Sign-up
# -------------------------------------------------------------------

@app.post("/signup", response_model=AuthResponse, summary="Create a new account")
async def signup(credentials: UserCredentials):
    username = credentials.username.lower()
    if username in USER_STORE:
        raise HTTPException(status_code=400, detail="Username already exists")

    session_id = str(uuid.uuid4())
    USER_STORE[username] = {"password": credentials.password, "session_id": session_id}

    # Initialize memory/state for new user
    clear_user_memory(session_id)
    clear_user_state(session_id)

    return AuthResponse(sessionId=session_id, message="Sign-up successful")

# -------------------------------------------------------------------
# Login
# -------------------------------------------------------------------

@app.post("/login", response_model=AuthResponse, summary="Login existing user")
async def login(credentials: UserCredentials):
    username = credentials.username.lower()
    user = USER_STORE.get(username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session_id = user["session_id"]
    return AuthResponse(sessionId=session_id, message="Login successful")

# -------------------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, summary="Send a message to AI Aura")
async def chat(request: ChatRequest):
    session_id = request.sessionId
    user_message = (request.message or "").strip()

    if not user_message:
        return ChatResponse(
            assistantText="Please send a message.",
            mode="chat",
            jobs=[],
            debug={"reason": "empty user message"}
        )

    # 1️⃣ Store user message safely
    try:
        append_user_memory(session_id, "user", user_message)
    except Exception:
        pass  # fail silently

    # 2️⃣ Load conversation history safely
    try:
       conversation_history = get_user_memory(session_id) or []
    except Exception:
       conversation_history = []
    try:
       state = get_user_state(session_id)
    except Exception:
        state = {}


    # 3️⃣ Call AI Aura backend logic
    try:
       response = await chat_with_user(
           session_id=session_id,
           user_message=user_message,
           conversation_history=conversation_history  # explicitly pass history
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] chat_with_user failed:\n{tb}")
        response = {
            "assistantText": "Oops, something went wrong. Please try again later.",
            "mode": "chat",
            "jobs": [],
            "debug": {"error": str(e), "traceback": tb}
        }

    # 4️⃣ Store assistant reply safely
    assistant_text = response.get("assistantText") or "Sorry, I couldn't generate a response."
    append_user_memory(session_id, "assistant", assistant_text)

    # 5️⃣ Save returned jobs into user's history
    for job in response.get("jobs") or []:
        save_user_job(session_id, job)

    # 5️⃣ Normalize job suggestions safely
    normalized_jobs: List[JobSuggestion] = []
    for job in response.get("jobs") or []:
        try:
            normalized_jobs.append(JobSuggestion(
                title=str(job.get("title") or ""),
                company=(str(job.get("company", {}).get("display_name"))
                         if isinstance(job.get("company"), dict)
                         else str(job.get("company") or "")),
                location=(str(job.get("location", {}).get("display_name"))
                      if isinstance(job.get("location"), dict)
                      else str(job.get("location") or "")),
                redirect_url=str(job.get("redirect_url") or "")
            ))
        except Exception:
            continue

    # 6️⃣ Return structured response
    return ChatResponse(
        assistantText=assistant_text,
        mode=str(response.get("mode") or "chat"),
        jobs=normalized_jobs,
        debug=response.get("debug") or {}
    )

# -------------------------------------------------------------------
# Reset session
# -------------------------------------------------------------------

@app.post("/reset-chat/{session_id}", summary="Reset chat memory and state")
async def reset_chat(session_id: str):
    try:
        clear_user_memory(session_id)
        clear_user_state(session_id)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] reset_chat failed:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")
    return {"status": "reset"}