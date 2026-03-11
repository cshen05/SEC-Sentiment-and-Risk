

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model import predict


VALID_LABELS = {"negative_risk", "neutral", "positive_outlook"}


def test_predict_returns_expected_schema() -> None:
    result = predict("The company may face increased regulatory scrutiny.")

    assert set(result.keys()) == {"label", "confidence", "probabilities"}
    assert result["label"] in VALID_LABELS
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

    probabilities = result["probabilities"]
    assert isinstance(probabilities, dict)
    assert set(probabilities.keys()) == VALID_LABELS
    assert all(isinstance(value, float) for value in probabilities.values())
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_predict_handles_positive_outlook_style_text() -> None:
    result = predict(
        "Continuing to bring to market compelling cloud-based experiences that generate increasing traffic and market share."
    )

    assert result["label"] in VALID_LABELS
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_handles_neutral_style_text() -> None:
    result = predict(
        "The company depends on the performance of carriers, wholesalers, retailers and other resellers."
    )

    assert result["label"] in VALID_LABELS
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_handles_long_text_with_truncation() -> None:
    long_text = " ".join([
        "The company may not be able to attract and retain talented employees." for _ in range(200)
    ])

    result = predict(long_text)

    assert result["label"] in VALID_LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6