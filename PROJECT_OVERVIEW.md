# Project Overview: Production-Ready Multi-Task NLP API

**Role**: Machine Learning Engineer / MLOps Architect  
**Tech Stack**: Python, PyTorch, FastAPI, Docker, MLflow, ONNX, Prometheus

---

## 🎯 Problem Statement
In modern NLP applications, deploying separate models for every task (sentiment, NER, QA) is inefficient. It leads to:
- **High Memory Usage**: Loading 3+ BERT models consumes significant RAM (1GB+ per model).
- **Maintenance Overhead**: Managing multiple training pipelines and deployment services.
- **Latency**: No shared computation between related tasks.

## ✅ Solution
I engineered a **Multi-Task Learning (MTL) system** that unifies three distinct NLP tasks into a single deployment pipeline. By utilizing a **shared DistilBERT encoder**, the system extracts language features once and routes them to lightweight, task-specific heads.

The result is a single container that serves Sentiment Analysis, Named Entity Recognition, and Question Answering with **60% less memory usage** than a multi-model approach.

## 🔑 Key Achievements

### 1. Architecture Optimization
- **Hard Parameter Sharing**: Implemented a shared backbone architecture in PyTorch.
- **Quantization**: Exported the trained model to **ONNX Int8**, reducing the artifact size from **260MB to 65MB** and boosting CPU inference speed by **3x**.

### 2. MLOps Best Practices
- **Experiment Tracking**: Integrated **MLflow** to log metrics (`qa_f1`, `ner_f1`), hyperparameters, and model versions automatically during training.
- **Reproducibility**: Fully containerized environment using **Docker Compose** ensures identical behavior across development and production.
- **Observability**: Added **Prometheus** middleware to the FastAPI app to expose real-time request latency and throughput metrics.

### 3. Production Readiness
- **Robust API**: REST API built with **FastAPI** including Pydantic data validation and auto-generated documentation (`/docs`).
- **Automated Pipeline**: End-to-end script handles data download, cleaning, validation splitting, training, evaluation, and export without manual intervention.
- **Zero-Downtime Design**: Configuration of Docker health checks ensures dependent services (like the API) only start when upstream services (MLflow) are healthy.

---

## 🚀 Impact
This project demonstrates the ability to take a machine learning concept from **research code to a production-grade service**. It moves beyond simple "model.fit()" tutorials by addressing real-world engineering challenges like memory constraints, latency optimization, and automated infrastructure.
