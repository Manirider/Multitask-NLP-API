from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from transformers import DistilBertTokenizerFast
import mlflow
import os

from src.api.routes import router as api_router
from src.api.metrics import MetricsMiddleware
from src.utils.mlflow_utils import setup_mlflow, get_latest_run_id
from src.utils.onnx_utils import create_onnx_session
from src.config import get_settings
from src.utils.logging_utils import configure_logger, get_logger
import src.api.routes as routes_module

configure_logger()
logger = get_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting up API service...")

    setup_mlflow(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
    )

    run_id = get_latest_run_id(settings.MLFLOW_EXPERIMENT_NAME)

    if run_id:
        logger.info(f"Found latest run: {run_id}")
        local_dir = "downloaded_artifacts"
        os.makedirs(local_dir, exist_ok=True)

        try:
            onnx_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="onnx/model.quant.onnx",
                dst_path=local_dir,
            )
            logger.info(f"Downloaded ONNX model to {onnx_path}")

            session = create_onnx_session(onnx_path)
            routes_module.onnx_session = session
            logger.info("ONNX session loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
    else:
        logger.warning("No successful run found in MLflow. Model not loaded.")

    try:
        logger.info("Loading tokenizer...")
        routes_module.tokenizer = DistilBertTokenizerFast.from_pretrained(
            settings.MODEL_NAME
        )
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")

    yield

    logger.info("Shutting down API service.")

app = FastAPI(title=settings.APP_NAME,
              version=settings.APP_VERSION, lifespan=lifespan)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
app.add_middleware(MetricsMiddleware)
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Multi-Task NLP API is running. Visit /docs for documentation."}
