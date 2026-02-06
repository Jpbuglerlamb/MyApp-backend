# app/ai/client.py
from openai import AsyncOpenAI
from settings import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY)