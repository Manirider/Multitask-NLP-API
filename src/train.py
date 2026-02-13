import argparse
import json
import os
import re
import string
import sys

import mlflow
import torch
import torch.nn as nn
from seqeval.metrics import f1_score as ner_f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

from src.config import get_settings
from src.modeling.datasets import NERDataset, QADataset, SentimentDataset
from src.modeling.multitask_model import MultiTaskDistilBert
from src.utils.logging_utils import configure_logger, get_logger
from src.utils.mlflow_utils import setup_mlflow
from src.utils.onnx_utils import export_to_onnx, quantize_onnx_model

configure_logger()
logger = get_logger(__name__)


class ONNXWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.sentiment_head = model.sentiment_head
        self.ner_head = model.ner_head
        self.qa_head = model.qa_head

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]
        sentiment_logits = self.sentiment_head(cls_output)

        ner_logits = self.ner_head(sequence_output)

        qa_logits = self.qa_head(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return sentiment_logits, ner_logits, start_logits, end_logits


def evaluate_sentiment(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, task_name="sentiment")
            preds = torch.argmax(outputs["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_ner(model, dataloader, device, label_map):

    model.eval()
    true_labels = []
    pred_labels = []

    id2label = {v: k for k, v in label_map.items()} if label_map else {}
    if not id2label:
        labels_list = [
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
        ]
        id2label = {i: label for i, label in enumerate(labels_list)}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, task_name="ner")
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=2)

            for i in range(labels.shape[0]):
                p_labels = []
                t_labels = []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        t_labels.append(id2label.get(labels[i, j].item(), "O"))
                        p_labels.append(id2label.get(predictions[i, j].item(), "O"))
                true_labels.append(t_labels)
                pred_labels.append(p_labels)

    return ner_f1_score(true_labels, pred_labels)


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):

    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def evaluate_qa(model, dataloader, device, tokenizer):

    model.eval()
    exact_match = 0
    f1_total = 0.0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids,
                attention_mask,
                task_name="qa",
                start_positions=start_positions,
                end_positions=end_positions,
            )

            start_logits, end_logits = outputs["logits"]

            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)

            for i in range(input_ids.shape[0]):
                pred_s = pred_start[i].item()
                pred_e = pred_end[i].item()
                true_s = start_positions[i].item()
                true_e = end_positions[i].item()

                pred_tokens = input_ids[i][pred_s : pred_e + 1]
                true_tokens = input_ids[i][true_s : true_e + 1]

                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                true_text = tokenizer.decode(true_tokens, skip_special_tokens=True)

                if normalize_answer(pred_text) == normalize_answer(true_text):
                    exact_match += 1
                f1_total += compute_f1(pred_text, true_text)
                total += 1

    em_score = (exact_match / total) * 100.0 if total > 0 else 0.0
    f1_score = (f1_total / total) * 100.0 if total > 0 else 0.0
    return em_score, f1_score


def train(args):
    try:
        settings = get_settings()
        setup_mlflow(
            tracking_uri=settings.MLFLOW_TRACKING_URI,
            experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = DistilBertTokenizerFast.from_pretrained(settings.MODEL_NAME)

        logger.info("Loading datasets...")
        data_dir = os.path.join("data", "processed")

        try:
            sentiment_ds = SentimentDataset(
                os.path.join(data_dir, "sentiment_train.json"),
                tokenizer,
                args.max_seq_length,
            )
            ner_ds = NERDataset(
                os.path.join(data_dir, "ner_train.json"),
                tokenizer,
                args.max_seq_length,
            )
            qa_ds = QADataset(
                os.path.join(data_dir, "qa_train.json"),
                tokenizer,
                args.max_seq_length,
            )
        except FileNotFoundError:
            logger.error("Data files not found. Please run src/preprocess.py first.")
            sys.exit(1)

        train_dataloaders = {
            "sentiment": DataLoader(
                sentiment_ds, batch_size=args.batch_size, shuffle=True
            ),
            "ner": DataLoader(ner_ds, batch_size=args.batch_size, shuffle=True),
            "qa": DataLoader(qa_ds, batch_size=args.batch_size, shuffle=True),
        }

        model = MultiTaskDistilBert(model_name=settings.MODEL_NAME)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

        total_steps = sum(len(dl) for dl in train_dataloaders.values()) * args.epochs
        if total_steps == 0:
            logger.error("No training data found. Exiting.")
            sys.exit(1)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_seq_length": args.max_seq_length,
                    "model_name": settings.MODEL_NAME,
                    "device": str(device),
                }
            )

            logger.info("Starting training...")
            global_step = 0
            all_metrics = {}

            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0.0
                epoch_steps = 0

                iterators = {task: iter(dl) for task, dl in train_dataloaders.items()}
                steps_per_epoch = sum(len(dl) for dl in train_dataloaders.values())
                progress_bar = tqdm(
                    total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}"
                )

                while iterators:
                    for task in list(iterators.keys()):
                        try:
                            batch = next(iterators[task])
                        except StopIteration:
                            del iterators[task]
                            continue

                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        if task == "sentiment":
                            labels = batch["labels"].to(device)
                            outputs = model(
                                input_ids, attention_mask, task_name=task, labels=labels
                            )
                        elif task == "ner":
                            labels = batch["labels"].to(device)
                            outputs = model(
                                input_ids, attention_mask, task_name=task, labels=labels
                            )
                        elif task == "qa":
                            start_positions = batch["start_positions"].to(device)
                            end_positions = batch["end_positions"].to(device)
                            outputs = model(
                                input_ids,
                                attention_mask,
                                task_name=task,
                                start_positions=start_positions,
                                end_positions=end_positions,
                            )

                        loss = outputs["loss"]
                        if loss is not None:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                            epoch_loss += loss.item()
                            epoch_steps += 1
                            global_step += 1

                            if global_step % 100 == 0:
                                mlflow.log_metric(
                                    "train_loss", loss.item(), step=global_step
                                )

                        progress_bar.update(1)

                progress_bar.close()
                avg_loss = epoch_loss / max(epoch_steps, 1)
                logger.info(f"Epoch {epoch + 1} — avg_loss: {avg_loss:.4f}")
                mlflow.log_metrics(
                    {"avg_train_loss": avg_loss, "epoch": epoch + 1}, step=global_step
                )
                all_metrics[f"epoch_{epoch + 1}_avg_loss"] = avg_loss

            logger.info("Loading validation datasets...")
            try:
                sentiment_val = SentimentDataset(
                    os.path.join(data_dir, "sentiment_validation.json"),
                    tokenizer,
                    args.max_seq_length,
                )
                ner_val = NERDataset(
                    os.path.join(data_dir, "ner_validation.json"),
                    tokenizer,
                    args.max_seq_length,
                )
                qa_val = QADataset(
                    os.path.join(data_dir, "qa_validation.json"),
                    tokenizer,
                    args.max_seq_length,
                )
                val_dataloaders = {
                    "sentiment": DataLoader(sentiment_val, batch_size=args.batch_size),
                    "ner": DataLoader(ner_val, batch_size=args.batch_size),
                    "qa": DataLoader(qa_val, batch_size=args.batch_size),
                }
            except FileNotFoundError:
                logger.warning(
                    "Validation data not found, using training data for eval."
                )
                val_dataloaders = train_dataloaders

            logger.info("Running evaluation...")

            sentiment_acc = evaluate_sentiment(
                model, val_dataloaders["sentiment"], device
            )
            logger.info(f"Sentiment Accuracy: {sentiment_acc:.4f}")
            mlflow.log_metric("sentiment_accuracy", sentiment_acc)
            all_metrics["sentiment_accuracy"] = sentiment_acc

            ner_f1 = evaluate_ner(model, val_dataloaders["ner"], device, {})
            logger.info(f"NER F1: {ner_f1:.4f}")
            mlflow.log_metric("ner_f1", ner_f1)
            all_metrics["ner_f1"] = ner_f1

            qa_em, qa_f1 = evaluate_qa(model, val_dataloaders["qa"], device, tokenizer)
            logger.info(f"QA Exact Match: {qa_em:.4f}")
            logger.info(f"QA F1: {qa_f1:.4f}")
            mlflow.log_metric("qa_exact_match", qa_em)
            mlflow.log_metric("qa_f1", qa_f1)
            all_metrics["qa_exact_match"] = qa_em
            all_metrics["qa_f1"] = qa_f1

            metrics_path = "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
            logger.info("Logged metrics.json artifact.")

            model_dir = "saved_model"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            mlflow.log_artifacts(model_dir, artifact_path="model")
            logger.info("Logged PyTorch model artifact.")

            logger.info("Exporting to ONNX...")
            model.to("cpu")
            onnx_wrapper = ONNXWrapper(model)
            onnx_wrapper.eval()

            dummy_input_ids = torch.randint(0, 1000, (1, args.max_seq_length))
            dummy_mask = torch.ones((1, args.max_seq_length), dtype=torch.long)

            onnx_dir = "onnx_export"
            os.makedirs(onnx_dir, exist_ok=True)
            onnx_path = os.path.join(onnx_dir, "model.onnx")

            export_to_onnx(
                onnx_wrapper,
                (dummy_input_ids, dummy_mask),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits", "ner_logits", "start_logits", "end_logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"},
                    "ner_logits": {0: "batch_size", 1: "sequence_length"},
                    "start_logits": {0: "batch_size", 1: "sequence_length"},
                    "end_logits": {0: "batch_size", 1: "sequence_length"},
                },
            )

            quantized_path = os.path.join(onnx_dir, "model.quant.onnx")
            quantize_onnx_model(onnx_path, quantized_path)

            mlflow.log_artifacts(onnx_dir, artifact_path="onnx")
            logger.info("Logged ONNX artifacts (model.onnx + model.quant.onnx).")

            logger.info(f"Training complete. MLflow Run ID: {run.info.run_id}")

    except Exception:
        logger.exception("Training failed with error")
        sys.exit(1)


if __name__ == "__main__":
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train Multi-Task NLP Model")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=settings.LEARNING_RATE)
    parser.add_argument("--max_seq_length", type=int, default=settings.MAX_SEQ_LENGTH)
    args = parser.parse_args()

    train(args)