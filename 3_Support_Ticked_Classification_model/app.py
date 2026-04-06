from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
import re
from contextlib import asynccontextmanager

# Load NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TicketInput(BaseModel):
    description: str

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        
    print("Loading models...")
    with open('models/model.pkl', 'rb') as f:
        ml_models['model'] = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        ml_models['vectorizer'] = pickle.load(f)
    with open('models/encoder.pkl', 'rb') as f:
        ml_models['encoder'] = pickle.load(f)
    print("Models loaded successfully")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Helper for text cleaning
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.post("/predict")
async def predict_priority(ticket: TicketInput):
    cleaned = clean_text(ticket.description)
    
    # Vectorize
    vectorizer = ml_models['vectorizer']
    X_tfidf = vectorizer.transform([cleaned])
    
    # Predict
    model = ml_models['model']
    pred_enc = model.predict(X_tfidf)
    
    # Decode
    encoder = ml_models['encoder']
    priority = encoder.inverse_transform(pred_enc)[0]
    
    return {"priority": priority}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
