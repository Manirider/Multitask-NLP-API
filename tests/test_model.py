def test_model_forward_sentiment(model, tokenizer):
    text = "This is a positive test."
    inputs = tokenizer(text, return_tensors="pt")

    # Task specific forward
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        task_name="sentiment",
    )

    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 2)


def test_model_forward_ner(model, tokenizer):
    text = "John Doe lives in New York."
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        task_name="ner",
    )

    seq_len = inputs["input_ids"].shape[1]
    # Default 9 labels from the class init
    assert outputs["logits"].shape == (1, seq_len, 9)


def test_model_forward_qa(model, tokenizer):
    question = "Who?"
    context = "Me."
    inputs = tokenizer(question, context, return_tensors="pt")

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        task_name="qa",
    )

    # Returns (start_logits, end_logits) tuple
    assert isinstance(outputs["logits"], tuple)
    assert len(outputs["logits"]) == 2

    start_logits, end_logits = outputs["logits"]
    seq_len = inputs["input_ids"].shape[1]

    assert start_logits.shape == (1, seq_len)
    assert end_logits.shape == (1, seq_len)
