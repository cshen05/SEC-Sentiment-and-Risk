import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_dir" in data
    assert "api_version" in data



def test_predict_returns_valid_schema() -> None:
    payload = {
        "text": "The company may not be able to attract and retain talented employees."
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == {"label", "confidence", "probabilities"}
    assert data["label"] in {"negative_risk", "neutral", "positive_outlook"}
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0

    probabilities = data["probabilities"]
    assert set(probabilities.keys()) == {
        "negative_risk",
        "neutral",
        "positive_outlook",
    }
    assert all(isinstance(value, float) for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6



def test_predict_rejects_empty_text() -> None:
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Input text cannot be empty."



def test_predict_batch_returns_list_of_predictions() -> None:
    payload = {
        "texts": [
            "The company may face increased regulatory scrutiny.",
            "Continuing to enhance the attractiveness of our cloud platforms.",
        ]
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert set(item.keys()) == {"label", "confidence", "probabilities"}
        assert item["label"] in {"negative_risk", "neutral", "positive_outlook"}
        assert 0.0 <= item["confidence"] <= 1.0
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-6



def test_predict_batch_rejects_empty_inputs() -> None:
    response = client.post("/predict/batch", json={"texts": ["   ", ""]})

    assert response.status_code == 400
    assert response.json()["detail"] == "Batch input must contain at least one non-empty text."
