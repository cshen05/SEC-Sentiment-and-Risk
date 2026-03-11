

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.model import predict


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Paragraph text to classify")


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of paragraph texts to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="REST API for SEC paragraph risk and sentiment classification",
)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_dir": str(settings.model_dir),
        "api_version": settings.api_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_single(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    return predict(text)


@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch(request: PredictBatchRequest):
    cleaned_texts = [text.strip() for text in request.texts if text and text.strip()]
    if not cleaned_texts:
        raise HTTPException(status_code=400, detail="Batch input must contain at least one non-empty text.")

    return [predict(text) for text in cleaned_texts]