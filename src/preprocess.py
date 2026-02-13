import json
import logging
import os

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_SAMPLES = 10000
DATA_DIR = os.path.join("data", "processed")


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(data, filename):
    filepath = os.path.join(DATA_DIR, filename)
    logger.info(f"Saving {len(data)} samples to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_sentiment():
    logger.info("Processing Sentiment Analysis (GLUE SST2) data...")
    dataset = load_dataset("glue", "sst2", split="train")

    processed_data = []
    for i, item in tqdm(enumerate(dataset), total=min(len(dataset), MAX_SAMPLES)):
        if i >= MAX_SAMPLES:
            break
        processed_data.append({"text": item["sentence"], "label": item["label"]})

    save_json(processed_data, "sentiment_train.json")

    val_dataset = load_dataset("glue", "sst2", split="validation")
    val_data = []
    for item in val_dataset:
        val_data.append({"text": item["sentence"], "label": item["label"]})
    save_json(val_data, "sentiment_validation.json")


def process_ner():
    logger.info("Processing NER (CoNLL-2003) data...")
    try:
        dataset = load_dataset("conll2003", split="train")

        processed_data = []
        for i, item in tqdm(enumerate(dataset), total=min(len(dataset), MAX_SAMPLES)):
            if i >= MAX_SAMPLES:
                break
            processed_data.append({"tokens": item["tokens"], "tags": item["ner_tags"]})

        save_json(processed_data, "ner_train.json")

        val_dataset = load_dataset("conll2003", split="validation")
        val_data = []
        for item in val_dataset:
            val_data.append({"tokens": item["tokens"], "tags": item["ner_tags"]})
        save_json(val_data, "ner_validation.json")

    except Exception as e:
        logger.error(f"Failed to load CoNLL-2003: {e}")
        logger.warning("Generating synthetic NER data as fallback...")
        generate_synthetic_ner()


def generate_synthetic_ner():

    processed_data = []

    for _i in range(100):
        processed_data.append(
            {
                "tokens": ["John", "Doe", "lives", "in", "New", "York", "."],
                "tags": [1, 2, 0, 0, 5, 6, 0],
            }
        )
    save_json(processed_data, "ner_train.json")
    save_json(processed_data, "ner_validation.json")


def process_qa():
    logger.info("Processing QA (SQuAD) data...")
    dataset = load_dataset("squad", split="train")

    processed_data = []
    for i, item in tqdm(enumerate(dataset), total=min(len(dataset), MAX_SAMPLES)):
        if i >= MAX_SAMPLES:
            break

        answers = item["answers"]

        processed_data.append(
            {
                "context": item["context"],
                "question": item["question"],
                "answers": {
                    "text": answers["text"],
                    "answer_start": answers["answer_start"],
                },
            }
        )

    save_json(processed_data, "qa_train.json")

    val_dataset = load_dataset("squad", split="validation")
    val_data = []
    for item in val_dataset:
        answers = item["answers"]
        val_data.append(
            {
                "context": item["context"],
                "question": item["question"],
                "answers": {
                    "text": answers["text"],
                    "answer_start": answers["answer_start"],
                },
            }
        )
    save_json(val_data, "qa_validation.json")


def main():
    ensure_dir(DATA_DIR)

    try:
        process_sentiment()
        process_ner()
        process_qa()
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
