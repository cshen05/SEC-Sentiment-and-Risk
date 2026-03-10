import pickle
import sys
from pathlib import Path

from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Allow running as: python3 -m training.train_baseline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "annotation_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "risk_baseline.pkl"

LABEL_MAP = {
    "negative_risk": 0,
    "neutral": 1,
    "positive_outlook": 2,
}

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def decode_label_counts(values) -> dict[str, int]:
    """Convert encoded label counts to readable label names."""
    counts = Counter(values)
    return {
        INVERSE_LABEL_MAP[label_id]: counts.get(label_id, 0)
        for label_id in sorted(INVERSE_LABEL_MAP)
    }


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the annotated dataset and keep only rows with manual labels."""
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


def train_baseline(df: pd.DataFrame):
    """Train a TF-IDF + Logistic Regression baseline classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "classification_report": classification_report(
            y_test,
            preds,
            target_names=["negative_risk", "neutral", "positive_outlook"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "full_label_counts": df["manual_label"].value_counts().to_dict(),
        "train_label_counts": decode_label_counts(y_train),
        "test_label_counts": decode_label_counts(y_test),
        "predicted_label_counts": decode_label_counts(preds),
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "test_results": pd.DataFrame(
            {
                "text": X_test.values,
                "true_label": [INVERSE_LABEL_MAP[label] for label in y_test],
                "pred_label": [INVERSE_LABEL_MAP[label] for label in preds],
            }
        ),
    }


def save_model(model, vectorizer, label_map: dict[str, int], output_path: Path) -> None:
    """Persist the trained model, vectorizer, and label mapping."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "vectorizer": vectorizer,
                "label_map": label_map,
                "inverse_label_map": {v: k for k, v in label_map.items()},
            },
            f,
        )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Annotated dataset not found: {DATA_PATH}")

    df = load_dataset(DATA_PATH)
    results = train_baseline(df)

    metrics = results["metrics"]

    print(f"[INFO] Loaded {len(df)} manually labeled rows")
    print(f"[INFO] Full label counts: {metrics['full_label_counts']}")
    print(f"[INFO] Train size: {metrics['train_size']}")
    print(f"[INFO] Train label counts: {metrics['train_label_counts']}")
    print(f"[INFO] Test size: {metrics['test_size']}")
    print(f"[INFO] Test label counts: {metrics['test_label_counts']}")
    print(f"[INFO] Predicted label counts: {metrics['predicted_label_counts']}")
    print(f"[INFO] Accuracy: {metrics['accuracy']:.4f}")
    print("\n[INFO] Classification report:\n")
    print(metrics["classification_report"])
    print("[INFO] Confusion matrix:")
    print(metrics["confusion_matrix"])

    mistakes = results["test_results"]
    mistakes = mistakes[mistakes["true_label"] != mistakes["pred_label"]]
    print("\n[INFO] Sample misclassifications:")
    if mistakes.empty:
        print("None")
    else:
        print(mistakes.head(10).to_string(index=False, max_colwidth=120))

    save_model(results["model"], results["vectorizer"], LABEL_MAP, MODEL_PATH)
    print(f"\n[INFO] Saved baseline model to {MODEL_PATH}")


if __name__ == "__main__":
    main()