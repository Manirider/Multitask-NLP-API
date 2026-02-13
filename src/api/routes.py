from fastapi import APIRouter, HTTPException
from .schemas import (
    SentimentRequest, SentimentResponse,
    NERRequest, NERResponse, Entity,
    QARequest, QAResponse,
)
import numpy as np
import logging
import onnxruntime
from transformers import DistilBertTokenizerFast

logger = logging.getLogger(__name__)

router = APIRouter()

onnx_session: onnxruntime.InferenceSession = None
tokenizer: DistilBertTokenizerFast = None

id_to_label_ner = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
}

def _get_model():
    if onnx_session is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return onnx_session

def _get_tokenizer():
    if tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Tokenizer not initialized")
    return tokenizer

def _run_inference(session, input_ids, attention_mask):

    input_feed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return session.run(None, input_feed)

@router.post("/predict/sentiment", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    session = _get_model()
    tok = _get_tokenizer()

    inputs = tok(request.text, return_tensors="np",
                 truncation=True, padding=True)
    outputs = _run_inference(
        session, inputs["input_ids"], inputs["attention_mask"])

    logits = outputs[0]
    probabilities = np.exp(logits) / \
        np.sum(np.exp(logits), axis=-1, keepdims=True)
    predicted_class_id = int(np.argmax(probabilities, axis=-1)[0])
    score = float(probabilities[0][predicted_class_id])
    sentiment = "positive" if predicted_class_id == 1 else "negative"

    return SentimentResponse(text=request.text, sentiment=sentiment, score=score)

@router.post("/predict/ner", response_model=NERResponse)
async def predict_ner(request: NERRequest):
    session = _get_model()
    tok = _get_tokenizer()

    inputs = tok(request.text, return_tensors="np",
                 truncation=True, padding=True,
                 return_offsets_mapping=True)

    offset_mapping = inputs.pop("offset_mapping")[0]
    outputs = _run_inference(
        session, inputs["input_ids"], inputs["attention_mask"])

    ner_logits = outputs[1]
    predictions = np.argmax(ner_logits, axis=-1)[0]

    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    current_entity = None

    for i, (token, label_id) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        label = id_to_label_ner.get(int(label_id), "O")
        char_start = int(offset_mapping[i][0])
        char_end = int(offset_mapping[i][1])

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": request.text[char_start:char_end],
                "type": label[2:],
                "score": 1.0,
                "start_char": char_start,
                "end_char": char_end,
            }
        elif label.startswith("I-") and current_entity:
            if label[2:] == current_entity["type"]:
                current_entity["text"] = request.text[current_entity["start_char"]:char_end]
                current_entity["end_char"] = char_end
            else:
                entities.append(current_entity)
                current_entity = None
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    api_entities = [
        Entity(text=e["text"], type=e["type"],
               score=e["score"], start_char=e["start_char"], end_char=e["end_char"])
        for e in entities
    ]

    return NERResponse(text=request.text, entities=api_entities)

@router.post("/predict/qa", response_model=QAResponse)
async def predict_qa(request: QARequest):
    session = _get_model()
    tok = _get_tokenizer()

    inputs = tok(
        request.question, request.context,
        return_tensors="np", truncation="only_second", padding=True,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    outputs = _run_inference(
        session, inputs["input_ids"], inputs["attention_mask"])

    start_logits = outputs[2]
    end_logits = outputs[3]

    start_index = int(np.argmax(start_logits))
    end_index = int(np.argmax(end_logits))

    if end_index < start_index:
        end_index = start_index

    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]
    answer = tok.decode(answer_tokens, skip_special_tokens=True)
    score = float(np.max(start_logits) + np.max(end_logits))

    start_char = int(offset_mapping[start_index][0])
    end_char = int(offset_mapping[end_index][1])

    return QAResponse(answer=answer, start_char=start_char, end_char=end_char, score=score)

@router.get("/health")
async def health_check():
    return {"status": "ok"}
