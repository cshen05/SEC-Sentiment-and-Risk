# SEC Sentiment and Risk Analysis

A production-style NLP pipeline that extracts risk disclosures from SEC filings and classifies their sentiment using transformer models. The system ingests filings, processes paragraph-level disclosures, predicts risk sentiment using a fine‑tuned FinBERT model, and exposes results through both batch pipelines and a FastAPI service.

---

# Project Overview

Public companies disclose potential business risks in their SEC filings (10‑K and 10‑Q). These disclosures often contain hundreds of paragraphs describing regulatory, operational, financial, and strategic risks.

Manually reviewing these disclosures is time‑consuming and difficult to scale.

This project builds a complete machine learning system that:

• Ingests SEC filings
• Extracts and stores risk paragraphs
• Labels and trains NLP models
• Runs sentiment classification
• Summarizes results per filing
• Serves predictions through an API

The goal is to demonstrate how financial NLP models can be built and deployed in a production‑ready pipeline.

---

# Features

• Automated SEC filing ingestion
• Paragraph‑level risk extraction
• Manual + weak labeling pipeline
• Baseline model (logistic regression)
• Transformer model (FinBERT)
• Batch inference pipeline
• Risk summary generation
• FastAPI inference service
• Dockerized deployment
• Automated testing
• CI pipeline with GitHub Actions

---

# Project Architecture

```
                SEC EDGAR
                    │
                    ▼
           build_corpus.py
                    │
                    ▼
          PostgreSQL Database
                    │
                    ▼
        Training / Labeling
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Baseline Model       FinBERT Model
 (Logistic Regression)  (Transformer)
          │                   │
          └─────────┬─────────┘
                    ▼
        predict_full_corpus.py
                    │
                    ▼
        summarize_predictions.py
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Batch Pipeline          FastAPI Service
 pipeline.batch_inference     app.main
```

---

# Repository Structure

```
SEC-Sentiment-and-Risk

app/
    config.py
    model.py
    main.py

pipeline/
    build_corpus.py
    load_corpus_to_db.py
    batch_inference.py

training/
    train_baseline.py
    train_finbert.py
    predict_full_corpus.py
    summarize_predictions.py

notebooks/
    00_annotation_review.ipynb
    01_label_analysis.ipynb
    02_corpus_analysis.ipynb
    03_baseline_results_analysis.ipynb
    04_finbert_model_analysis.ipynb

data/
models/

logs/

scripts/

tests/
    test_api.py
    test_model.py
    test_pipeline.py

Dockerfile
docker-compose.yml
requirements.txt
README.md
```

---

# Dataset

Source: SEC EDGAR

The corpus consists of paragraphs extracted from the **Risk Factors** sections of:

• Form 10‑K
• Form 10‑Q

Each paragraph is labeled into one of three categories:

```
negative_risk
neutral
positive_outlook
```

Example:

```
"The company may not be able to attract and retain talented employees."

→ negative_risk
```

---

# Labeling Strategy

Two labeling methods were used.

### Manual Annotation

A curated subset of paragraphs was manually labeled to provide high‑quality training examples.

### Weak Labeling

Heuristic rules and patterns were used to generate additional labels at scale.

This hybrid approach improves dataset size while preserving label quality.

---

# Models

## Baseline Model

Algorithm:

```
TF‑IDF + Logistic Regression
```

Purpose:

• Provide a simple benchmark
• Validate dataset quality

---

## Transformer Model

Model:

```
FinBERT
```

FinBERT is a transformer pretrained on financial text and fine‑tuned for sentiment classification.

Advantages:

• Financial domain knowledge
• Better understanding of contextual risk language

---

# Model Evaluation

| Model | Accuracy | F1 Score |
|------|------|------|
| Baseline | ~0.75 | ~0.73 |
| FinBERT | ~0.87 | ~0.86 |

FinBERT significantly improves classification of nuanced financial risk language.

---

# Analysis Notebooks

```
02_corpus_analysis.ipynb
```

Explores:

• dataset distribution
• paragraph lengths
• label balance

```
03_baseline_results_analysis.ipynb
```

Evaluates:

• baseline performance
• error analysis

```
04_finbert_model_analysis.ipynb
```

Evaluates:

• transformer results
• confidence distribution

---

# API

The project exposes a FastAPI service for real‑time inference.

### Start the API

```
uvicorn app.main:app --reload
```

### Endpoint

```
POST /predict
```

Example request:

```
{
  "text": "The company may face regulatory scrutiny in international markets."
}
```

Example response:

```
{
  "label": "negative_risk",
  "confidence": 0.92,
  "probabilities": {
    "negative_risk": 0.92,
    "neutral": 0.06,
    "positive_outlook": 0.02
  }
}
```

---

# Batch Pipeline

The system can also run full‑corpus inference.

```
python -m pipeline.batch_inference
```

Pipeline steps:

1. Fetch SEC filings
2. Store paragraphs in database
3. Run FinBERT inference
4. Generate risk summaries

Outputs:

```
data/predictions.csv
reports/filing_summary.csv
reports/top_risk_paragraphs.csv
```

---

# Docker Deployment

Build the containers:

```
docker compose up --build
```

Services:

• PostgreSQL database
• FastAPI inference service

API becomes available at:

```
http://localhost:8000
```

Swagger documentation:

```
http://localhost:8000/docs
```

---

# Testing

The project includes automated tests covering the API, model inference, and data pipeline.

Run tests with:

```
pytest
```

Test coverage includes:

• API endpoints
• model predictions
• pipeline summary logic

---

# Continuous Integration

GitHub Actions automatically runs the test suite on every push.

Workflow file:

```
.github/workflows/ci.yml
```

Pipeline steps:

1. Install dependencies
2. Run pytest

---

# Example Output

Example filing summary:

| Company | Filing | Risk Paragraphs | High Confidence Risks |
|------|------|------|------|
| Apple | 10‑K | 120 | 34 |
| Microsoft | 10‑Q | 85 | 22 |

Example top risk paragraph:

```
"Supply chain disruptions could materially affect the company's operations."
```

---

# Future Improvements

Possible extensions:

• Topic modeling of risk categories
• Risk trend analysis over time
• Financial event correlation
• Dashboard visualization
• Multi‑company comparative analysis

---

# Technologies Used

Python
PyTorch
Transformers (HuggingFace)
FastAPI
PostgreSQL
Docker
Pytest
GitHub Actions

---

# Author

Connor Shen

University of Texas at Austin
Economics + Statistics & Data Science

---

# License

MIT License

