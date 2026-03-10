

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Allow running as: python3 -m training.train_finbert
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "annotation_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_DIR = MODELS_DIR / "finbert_risk_classifier"
METRICS_PATH = MODELS_DIR / "finbert_metrics.pkl"
MODEL_NAME = "ProsusAI/finbert"

LABEL_MAP = {
    "negative_risk": 0,
    "neutral": 1,
    "positive_outlook": 2,
}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load manually labeled rows and encode class labels."""
    df = pd.read_csv(csv_path)
    df = df[df["manual_label"].notna()].copy()
    df["manual_label"] = df["manual_label"].astype(str).str.strip()

    if df.empty:
        raise ValueError("No manually labeled rows found in annotation_sample.csv")

    df["label"] = df["manual_label"].map(LABEL_MAP)

    if df["label"].isna().any():
        bad_labels = sorted(df.loc[df["label"].isna(), "manual_label"].unique())
        raise ValueError(f"Found unexpected manual labels: {bad_labels}")

    return df


def build_splits(df: pd.DataFrame):
    """Create stratified train/test splits from the manually labeled data."""
    train_df, test_df = train_test_split(
        df[["text", "label", "manual_label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_dataset(tokenizer, dataset: Dataset) -> Dataset:
    """Tokenize text for transformer training."""

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    tokenized = dataset.map(_tokenize, batched=True)
    tokenized = tokenized.remove_columns([col for col in tokenized.column_names if col not in {"label", "input_ids", "attention_mask", "token_type_ids"}])
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    """Compute evaluation metrics for Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Annotated dataset not found: {DATA_PATH}")

    df = load_dataset(DATA_PATH)
    train_df, test_df = build_splits(df)

    print(f"[INFO] Loaded {len(df)} manually labeled rows")
    print(f"[INFO] Full label counts: {df['manual_label'].value_counts().to_dict()}")
    print(f"[INFO] Train size: {len(train_df)}")
    print(f"[INFO] Train label counts: {train_df['manual_label'].value_counts().to_dict()}")
    print(f"[INFO] Test size: {len(test_df)}")
    print(f"[INFO] Test label counts: {test_df['manual_label'].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False)

    train_dataset = tokenize_dataset(tokenizer, train_dataset)
    test_dataset = tokenize_dataset(tokenizer, test_dataset)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = predictions.label_ids

    accuracy = accuracy_score(true_ids, pred_ids)
    report = classification_report(
        true_ids,
        pred_ids,
        target_names=["negative_risk", "neutral", "positive_outlook"],
        zero_division=0,
    )
    cm = confusion_matrix(true_ids, pred_ids)

    pred_label_counts = pd.Series([INVERSE_LABEL_MAP[i] for i in pred_ids]).value_counts().to_dict()

    print(f"[INFO] Predicted label counts: {pred_label_counts}")
    print(f"[INFO] Accuracy: {accuracy:.4f}")
    print("\n[INFO] Classification report:\n")
    print(report)
    print("[INFO] Confusion matrix:")
    print(cm)

    test_results = pd.DataFrame(
        {
            "text": test_df["text"].values,
            "true_label": [INVERSE_LABEL_MAP[i] for i in true_ids],
            "pred_label": [INVERSE_LABEL_MAP[i] for i in pred_ids],
        }
    )
    mistakes = test_results[test_results["true_label"] != test_results["pred_label"]]

    print("\n[INFO] Sample misclassifications:")
    if mistakes.empty:
        print("None")
    else:
        print(mistakes.head(10).to_string(index=False, max_colwidth=120))

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    metrics_payload = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "test_results": test_results,
    }
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics_payload, f)

    print(f"\n[INFO] Saved FinBERT model to {MODEL_OUTPUT_DIR}")
    print(f"[INFO] Saved FinBERT metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()