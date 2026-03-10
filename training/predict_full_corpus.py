

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow running as: python3 -m training.predict_full_corpus
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import get_connection

MODEL_DIR = PROJECT_ROOT / "models" / "finbert_risk_classifier"
BATCH_SIZE = 32
MAX_LENGTH = 256

ID2LABEL = {
    0: "negative_risk",
    1: "neutral",
    2: "positive_outlook",
}


def get_device() -> torch.device:
    """Choose the best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_dir: Path):
    """Load the saved fine-tuned FinBERT model and tokenizer."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Saved model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = get_device()
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded model from {model_dir}")
    print(f"[INFO] Using device: {device}")

    return tokenizer, model, device


def fetch_paragraph_rows() -> pd.DataFrame:
    """Load all paragraph rows that need model predictions."""
    conn = get_connection()
    query = """
        SELECT id, text
        FROM paragraphs
        ORDER BY id
    """

    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("No paragraphs found in the database.")

    print(f"[INFO] Loaded {len(df)} paragraphs from the database")
    return df


def predict_batch(texts, tokenizer, model, device: torch.device):
    """Generate class predictions and confidence scores for a batch of texts."""
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)
        confs, pred_ids = torch.max(probs, dim=-1)

    pred_ids = pred_ids.detach().cpu().numpy()
    confs = confs.detach().cpu().numpy()

    labels = [ID2LABEL[int(i)] for i in pred_ids]
    return labels, confs


def predict_dataframe(df: pd.DataFrame, tokenizer, model, device: torch.device) -> pd.DataFrame:
    """Run batched inference across the full paragraph dataframe."""
    predictions = []
    confidences = []

    texts = df["text"].fillna("").astype(str).tolist()
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_labels, batch_confs = predict_batch(batch_texts, tokenizer, model, device)

        predictions.extend(batch_labels)
        confidences.extend(batch_confs.tolist())

        print(f"[INFO] Predicted rows {start + 1}-{end} of {total}")

    result = df.copy()
    result["prediction"] = predictions
    result["confidence"] = confidences
    return result


def write_predictions_to_db(df: pd.DataFrame) -> None:
    """Write model predictions and confidence scores back to the database."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        for row in df.itertuples(index=False):
            cur.execute(
                """
                UPDATE paragraphs
                SET prediction = %s,
                    confidence = %s
                WHERE id = %s
                """,
                (row.prediction, float(row.confidence), int(row.id)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    print(f"[INFO] Wrote predictions for {len(df)} paragraphs back to the database")


def print_prediction_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of model outputs."""
    counts = df["prediction"].value_counts().to_dict()
    avg_conf = df.groupby("prediction")["confidence"].mean().round(4).to_dict()

    print(f"[INFO] Prediction counts: {counts}")
    print(f"[INFO] Average confidence by class: {avg_conf}")



def main() -> None:
    tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR)
    df = fetch_paragraph_rows()
    result_df = predict_dataframe(df, tokenizer, model, device)
    print_prediction_summary(result_df)
    write_predictions_to_db(result_df[["id", "prediction", "confidence"]])


if __name__ == "__main__":
    main()