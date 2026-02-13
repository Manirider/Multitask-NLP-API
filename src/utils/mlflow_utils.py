import mlflow
import os
import logging

logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name=None, tracking_uri=None):
    if tracking_uri is None:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000")
    if experiment_name is None:
        experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "multitask_nlp_experiment")

    logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Setting MLflow experiment to: {experiment_name}")
    mlflow.set_experiment(experiment_name)

def get_latest_run_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
        filter_string="status = 'FINISHED'",
    )

    if runs.empty:
        return None

    return runs.iloc[0].run_id
