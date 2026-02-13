import os
from unittest.mock import patch

import pytest

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_experiment"


@pytest.fixture
def client():
    with (
        patch("src.main.setup_mlflow"),
        patch("src.main.get_latest_run_id", return_value=None),
    ):
        from fastapi.testclient import TestClient

        from src.main import app

        with TestClient(app) as c:
            yield c


@pytest.fixture
def model():
    from src.modeling.multitask_model import MultiTaskDistilBert

    return MultiTaskDistilBert(model_name="distilbert-base-uncased")


@pytest.fixture
def tokenizer():
    from transformers import DistilBertTokenizerFast

    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
