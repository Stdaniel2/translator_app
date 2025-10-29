from fastapi import FastAPI
from pydantic import BaseModel
from .translator import translate_text

app = FastAPI(title="English–Igbo Translator API")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the English–Igbo Translator API"}

@app.post("/translate")
def translate(input_data: TextInput):
    result = translate_text(input_data.text)
    return {"translated_text": result}
