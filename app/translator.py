from .model_loader import model, tokenizer

def translate_text(text: str) -> str:
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    translated = model.generate(**inputs, max_length=128)
    
    # Decode the output
    return tokenizer.decode(translated[0], skip_special_tokens=True)
