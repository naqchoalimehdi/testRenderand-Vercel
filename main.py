from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os

# Load API Key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Load LangChain Chat Model (ChatGroq)
chat_model = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

# Request schema
class ChatRequest(BaseModel):
    message: str

# Chatbot Route
@app.post("/chat")
async def chat(request: ChatRequest):
    response = chat_model.invoke(request.message)
    return {"reply": response.content}
