from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path

# Create FastAPI instance
app = FastAPI(title="English-Igbo Translator API")

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load translation model
model_name = "Helsinki-NLP/opus-mt-en-ig"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Input format
class TextInput(BaseModel):
    text: str

# Serve index.html when visiting the root URL
@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    html_path = Path(__file__).parent.parent / "index.html"
    return html_path.read_text(encoding="utf-8")

# Translation endpoint
@app.post("/translate")
def translate_text(data: TextInput):
    inputs = tokenizer(data.text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translated_text": result}

# Optional root check
@app.get("/health")
def health_check():
    return {"status": "running", "message": "Translator API is live"}
