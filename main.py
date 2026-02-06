#main.py
from fastapi import FastAPI
from api.chat import router as chat_router
from api.auth import router as auth_router
from core.database import engine, Base
from fastapi import Depends
from core.auth_utils import get_current_user_id

app = FastAPI(title="AI Aura")

app.include_router(auth_router)
app.include_router(chat_router)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
def health():
    return {"status": "ok"}

from fastapi import Depends
from core.auth_utils import get_current_user_id

@app.get("/me")
def get_me(user_id: str = Depends(get_current_user_id)):
    return {"userId": user_id, "message": "Token is valid!"}