from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import settings


LABEL_MAP = {
    0: "negative_risk",
    1: "neutral",
    2: "positive_outlook",
}


class FinBertRiskModel:
    """Wrapper around the fine-tuned FinBERT model used for risk classification."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

        # Load model (prefer fine-tuned local model if present)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_dir)
        except Exception:
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict_text(self, text: str) -> Dict:
        """Run inference on a single paragraph of text."""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=settings.max_length,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(probs.argmax())

        return {
            "label": LABEL_MAP[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {
                LABEL_MAP[i]: float(probs[i])
                for i in range(len(probs))
            },
        }


# Load model once at startup
model = FinBertRiskModel()


def predict(text: str) -> Dict:
    """Convenience wrapper for API usage."""
    return model.predict_text(text)
