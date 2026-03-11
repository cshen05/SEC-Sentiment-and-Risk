# SEC Sentiment and Risk Analysis

An **end‑to‑end NLP and deep learning system** that ingests SEC filings, extracts risk disclosures, classifies their sentiment using financial transformer models, and surfaces portfolio‑level risk insights through APIs, dashboards, and automated pipelines.

This project demonstrates how to design, train, deploy, and monitor a **production‑style financial NLP platform** combining machine learning, data engineering, and backend deployment.

---

# Project Overview

Public companies disclose hundreds of paragraphs of potential business risks in their SEC filings (10‑K and 10‑Q). These disclosures often contain critical information about:

• operational risks  
• regulatory exposure  
• macroeconomic threats  
• supply chain vulnerabilities  

Manually reviewing these disclosures is slow and difficult to scale.

This project builds a system that automatically:

• ingests SEC filings from EDGAR  
• extracts paragraph‑level risk disclosures  
• trains NLP models to classify risk sentiment  
• performs large‑scale inference on filings  
• generates company‑level risk summaries  
• exposes predictions through APIs and dashboards

The result is a **production‑style machine learning pipeline for financial document analysis**.

---

# Quickstart

Run the full project locally in three steps.

### 1. Start the infrastructure

```
docker compose up --build
```

This launches:

• PostgreSQL database  
• FastAPI inference service

---

### 2. Run the batch pipeline

```
python -m pipeline.batch_inference
```

This will:

1. ingest SEC filings  
2. extract risk paragraphs  
3. run FinBERT inference  
4. generate filing‑level summaries

---

### 3. Launch the dashboard

```
streamlit run dashboard/app.py
```

---

# System Architecture

The system combines **data ingestion, NLP modeling, batch processing, and real‑time inference**.

```
              SEC EDGAR
                   │
                   ▼
           Filing Ingestion
          (build_corpus.py)
                   │
                   ▼
         Paragraph Extraction
                   │
                   ▼
          PostgreSQL Database
                   │
                   ▼
            Model Training
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Baseline Model          FinBERT Model
 (Logistic Regression)   (Transformer)
        │                     │
        └──────────┬──────────┘
                   ▼
            Batch Inference
        (predict_full_corpus)
                   │
                   ▼
           Risk Aggregation
       (summarize_predictions)
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
  FastAPI Service        Streamlit Dashboards
   Real‑time NLP           Monitoring + Analysis
```

---

# Key Features

## Automated SEC Filing Ingestion

The pipeline retrieves filings directly from **SEC EDGAR** and extracts the Risk Factors section.

Supported filings:

• Form 10‑K  
• Form 10‑Q

Each filing is converted into **paragraph‑level training examples**.

---

## Risk Sentiment Classification

Each paragraph is classified into one of three categories:

```
negative_risk
neutral
positive_outlook
```

Example:

```
"The company may not be able to attract and retain skilled employees."

→ negative_risk
```

---

# Machine Learning Models

## Baseline Model

Algorithm:

```
TF‑IDF + Logistic Regression
```

Purpose:

• establish a benchmark
• validate dataset quality

---

## Transformer Model

Model:

```
FinBERT
```

FinBERT is a financial domain transformer pretrained on financial text.

Advantages:

• financial language understanding  
• contextual interpretation of risk statements  
• improved classification accuracy

---

# Model Performance

| Model | Accuracy | F1 Score |
|------|------|------|
| Baseline | ~0.75 | ~0.73 |
| FinBERT | ~0.87 | ~0.86 |

The transformer model significantly improves detection of nuanced financial risk language.

---

# Batch Inference Pipeline

The project supports **large‑scale inference across entire filing corpora**.

Run:

```
python -m pipeline.batch_inference
```

Pipeline steps:

1. ingest SEC filings  
2. store paragraphs in PostgreSQL  
3. run FinBERT predictions  
4. generate risk summaries

Outputs:

```
data/predictions.csv
reports/filing_summary.csv
reports/top_risk_paragraphs.csv
```

---

# FastAPI Inference Service

The trained model is deployed through a **FastAPI service** for real‑time predictions.

Start the API:

```
uvicorn app.main:app --reload
```

Example request

```
POST /predict

{
  "text": "The company may face regulatory scrutiny in international markets."
}
```

Example response

```
{
  "label": "negative_risk",
  "confidence": 0.92
}
```

API documentation:

```
http://localhost:8000/docs
```

---

# Dashboards

Interactive dashboards built with **Streamlit** visualize:

• model evaluation results  
• prediction distributions  
• portfolio risk signals  
• model monitoring metrics  
• newly detected SEC filings

Run:

```
streamlit run dashboard/app.py
```

---

# Repository Structure

```
SEC-Sentiment-and-Risk

app/                FastAPI inference service
pipeline/           SEC ingestion and batch pipelines
training/           model training and prediction
notebooks/          analysis and experimentation

data/               datasets and generated reports
models/             trained models

scripts/            pipeline runners
logs/               execution logs

tests/              automated tests

Dockerfile
docker-compose.yml
README.md
```

---

# Testing

Automated tests ensure reliability of the API, model inference, and data pipeline.

Run tests with:

```
pytest
```

Coverage includes:

• API endpoints  
• model predictions  
• pipeline aggregation logic

---

# Continuous Integration

GitHub Actions automatically runs the test suite on every push.

Workflow file:

```
.github/workflows/ci.yml
```

Pipeline steps:

1. install dependencies  
2. run pytest

---

# Technologies Used

Python  
PyTorch  
HuggingFace Transformers  
FastAPI  
Streamlit  
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