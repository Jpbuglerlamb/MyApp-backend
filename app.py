from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import traceback

from OpenAi import chat_with_user
from memory_store import get_memory, append_message, clear_memory, clear_state

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

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/", summary="Health check endpoint")
async def health_check():
    return {"status": "ok"}

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
        append_message(session_id, "user", user_message)
    except Exception:
        pass  # fail silently if memory store fails

    # 2️⃣ Load conversation history safely
    try:
        conversation_history = get_memory(session_id) or []
    except Exception:
        conversation_history = []

    # 3️⃣ Call AI Aura backend logic
    try:
        response = await chat_with_user(
            user_id=session_id,
            user_message=user_message,
            conversation_history=conversation_history  # <-- corrected param name
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
    try:
        append_message(session_id, "assistant", assistant_text)
    except Exception:
        pass

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
        clear_memory(session_id)
        clear_state(session_id)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] reset_chat failed:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")
    return {"status": "reset"}