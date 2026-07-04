"""
FastAPI Backend for Kazakh NER Model
=====================================
Provides a REST endpoint for Named Entity Recognition
on Kazakh text using the fine-tuned XLM-RoBERTa model.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8080

Endpoints:
    POST /predict  — Extract entities from Kazakh text
    GET  /health   — Health check
    GET  /info     — Model information
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import uvicorn

app = FastAPI(
    title="Kazakh NER API",
    description="Named Entity Recognition for Kazakh Language using XLM-RoBERTa",
    version="1.0.0",
)

MODEL_PATH = "./ner-kazakh-best"

# Load model at startup
try:
    ner_pipeline = pipeline(
        "ner",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        aggregation_strategy="simple",
    )
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")
    model_loaded = False


class TextRequest(BaseModel):
    text: str


class Entity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


class PredictionResponse(BaseModel):
    text: str
    entities: list[Entity]
    count: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/info")
def info():
    return {
        "model": "xlm-roberta-base (fine-tuned)",
        "dataset": "KazNERD",
        "num_labels": 51,
        "language": "Kazakh",
        "f1_score": 0.873,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    results = ner_pipeline(request.text)

    entities = [
        Entity(
            entity_group=r["entity_group"],
            word=r["word"],
            score=round(r["score"], 4),
            start=r["start"],
            end=r["end"],
        )
        for r in results
    ]

    return PredictionResponse(
        text=request.text,
        entities=entities,
        count=len(entities),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
