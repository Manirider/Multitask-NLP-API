# Production-Ready Multi-Task NLP API

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-blueviolet.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)

**A production-grade NLP system serving Sentiment Analysis, Named Entity Recognition (NER), and Question Answering (QA) via a single shared Transformer model.**

---

## 🚀 Project Overview

This project implements a scalable, multi-task NLP architecture designed for high-performance inference. Instead of deploying three separate models (which would triple memory usage and maintenance), we utilize a **shared DistilBERT encoder** with lightweight task-specific heads. 

The system is fully containerized, tracks experiments via **MLflow**, and exports optimized **quantized ONNX models** for low-latency inference using **FastAPI**.

### Key Features
- **Multi-Task Learning**: Shared encoder architecture reduces memory footprint by ~60% compared to separate models.
- **Production Inference**: Serves **Int8 Quantized ONNX** models via ONNX Runtime for <50ms latency on CPU.
- **Experiment Tracking**: Full MLflow integration for metrics (`f1`, `accuracy`, `loss`) and artifact versioning.
- **Robust Pipeline**: Automated data preprocessing, validation split handling, and round-robin training loop.
- **Observability**: Prometheus metrics for API latency and throughput monitoring.
- **Infrastructure-as-Code**: One-command deployment via Docker Compose with health checks.

---

## 🏗️ System Architecture

The system uses a hard-parameter sharing approach where the lower layers (DistilBERT) are shared across all tasks, while the top layers (Heads) specialized for each specific task.

```ascii
                    ┌─────────────────────────┐
  Input Text ──────►│  DistilBERT Encoder     │ ◄─── Shared Weights
                    │  (6 layers, 768 dim)    │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┼─────────────┐
                   ▼             ▼             ▼
             ┌──────────┐  ┌──────────┐  ┌──────────┐
             │ Sentiment│  │   NER    │  │    QA    │
             │   Head   │  │   Head   │  │   Head   │
             └────┬─────┘  └────┬─────┘  └────┬─────┘
                  ▼             ▼             ▼
            {"sentiment":   [{"entity":      {"answer":
             "positive"}      "ORG"}]         "Paris"}
```

---

## 🛠️ Tech Stack

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| **Core** | Python 3.10, PyTorch | Industry standard for deep learning research and production. |
| **Model** | Hugging Face Transformers | State-of-the-art pre-trained models (DistilBERT). |
| **Training** | Custom Loop + MLflow | Full control over multi-task batching; centralized experiment tracking. |
| **Serving** | FastAPI + Uvicorn | High-performance ASGI framework with auto-generated Swagger docs. |
| **Inference** | ONNX Runtime (Quantized) | Faster inference on CPU, reduced model size (260MB → 65MB). |
| **Ops** | Docker, Docker Compose | Reproducible environments; "works on my machine" guarantee. |
| **Monitoring** | Prometheus | Real-time metrics scraping for latency/throughput. |

---

## 📂 Project Structure

```bash
MULTITASK-NLP-API/
├── src/
│   ├── api/                 # FastAPI endpoints & schemas
│   ├── modeling/            # PyTorch model architecture
│   ├── utils/               # Helpers for MLflow, ONNX, Logging
│   ├── config.py            # Configuration management (Pydantic)
│   ├── preprocess.py        # Data cleaning & tokenization pipeline
│   └── train.py             # Multi-task training loop & export
├── tests/                   # Pytest integration & unit tests
├── docker-compose.yml       # Service orchestration
├── Dockerfile               # Multi-stage build definition
└── requirements.txt         # Production dependencies
```

---

## ⚡ Setup & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multitask-nlp-api.git
cd multitask-nlp-api
```

### 2. Configure Environment
Create a `.env` file from the example:
```bash
cp .env.example .env
```
*Note: The default settings in `.env.example` are tuned for a standard laptop (16GB RAM).*

### 3. Build & Run
Launch the entire stack (Training + API + MLflow + Monitoring):
```bash
docker-compose up --build
```
*The system will automatically download datasets, train the model (approx. 40-60 mins on CPU), export to ONNX, and start the API.*

---

## 🔌 API Endpoints

Once running, the API is available at `http://localhost:8000`.

### Documentation
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 1. Sentiment Analysis
**POST** `/predict/sentiment`
```bash
curl -X POST http://localhost:8000/predict/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The implementation was seamless and very robust."}'
```
**Response:**
```json
{
  "text": "The implementation was seamless and very robust.",
  "sentiment": "positive",
  "score": 0.98
}
```

### 2. Named Entity Recognition
**POST** `/predict/ner`
```bash
curl -X POST http://localhost:8000/predict/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc. is planning to open a new office in London."}'
```
**Response:**
```json
{
  "text": "Apple Inc. is planning to open a new office in London.",
  "entities": [
    {"text": "Apple Inc.", "type": "ORG", "start_char": 0, "end_char": 10},
    {"text": "London", "type": "LOC", "start_char": 47, "end_char": 53}
  ]
}
```

### 3. Question Answering
**POST** `/predict/qa`
```bash
curl -X POST http://localhost:8000/predict/qa \
  -H "Content-Type: application/json" \
  -d '{"context": "MLflow is an open source platform for the machine learning lifecycle.", "question": "What is MLflow?"}'
```
**Response:**
```json
{
  "answer": "an open source platform for the machine learning lifecycle",
  "start_char": 10,
  "end_char": 68,
  "score": 0.95
}
```

---

## 📊 MLflow & Monitoring

### Experiment Tracking (MLflow)
Access the MLflow UI at [http://localhost:5000](http://localhost:5000).
- **Metrics**: Track `loss`, `sentiment_accuracy`, `ner_f1`, and `qa_f1` over epochs.
- **Artifacts**: View saved models (`model.onnx`, `model.quant.onnx`) and configuration (`metrics.json`).

### Prometheus Integration
Metrics are exposed at `http://localhost:8000/metrics`.
- `api_requests_total`: Total request count.
- `api_request_latency_seconds`: Histogram of response times.

---

## 💡 Design Decisions

1.  **Shared Encoder (Hard Parameter Sharing)**:
    *   **Decision**: We used a single DistilBERT base for all three tasks.
    *   **Reasoning**: This forces the model to learn generalized features applicable to syntax (NER), semantics (Sentiment), and comprehension (QA). It heavily optimizes memory throughput, allowing three functional endpoints to be served by a single loaded model.

2.  **Why ONNX Runtime?**:
    *   **Decision**: Export PyTorch models to ONNX and apply dynamic quantization (Int8).
    *   **Reasoning**: PyTorch is excellent for training but heavy for simple inference. ONNX Runtime provides a standard, optimized execution provider. Quantization reduced the model size from ~260MB to ~65MB, significantly speeding up container startup and reducing RAM usage.

3.  **Round-Robin Training**:
    *   **Decision**: Training batches cycle through tasks (Batch 1: Sentiment, Batch 2: NER, Batch 3: QA).
    *   **Reasoning**: This prevents "catastrophic forgetting" where the model optimizes for one task at the expense of others. It ensures balanced gradient updates across the shared encoder.

---

## 🔮 Future Improvements

- **GPU Support**: Add `nvidia-docker` support for CUDA acceleration in training and inference.
- **Redis Caching**: Implement a cache layer for frequent queries (e.g., common QA pairs).
- **Asynchronous Batching**: Use `batches` in FastAPI to group incoming requests for higher throughput.

---

## 📄 License
This project is licensed under the MIT License.
