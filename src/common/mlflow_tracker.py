"""
MLflow Integration for tracking Federation rounds and models.
"""
import os
import json
from common.logger import setup_logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = setup_logger("MLflowTracker")


class FederationTracker:
    def __init__(self, tracking_uri: str = None, experiment_name: str = "PmadFederation"):
        self.enabled = MLFLOW_AVAILABLE
        if not self.enabled:
            logger.warning("MLflow not installed. Tracking disabled.")
            return

        tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Start a new run
        self.run = mlflow.start_run()
        logger.info(f"MLflow tracking started. Run ID: {self.run.info.run_id}")

    def log_round(self, round_id: int, metrics: dict, num_agents: int):
        if not self.enabled:
            return

        # Log metrics with round as step
        mlflow.log_metrics({
            "avg_accuracy": metrics.get("accuracy", 0.0),
            "avg_loss": metrics.get("loss", 0.0),
            "agents_participating": num_agents
        }, step=round_id)

    def log_final_model(self, global_model: dict):
        if not self.enabled:
            return

        # Register final model as an artifact
        try:
            with open("final_model.json", "w") as f:
                json.dump(global_model, f)
            mlflow.log_artifact("final_model.json")
            logger.info("Final model registered in MLflow.")
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
        finally:
            mlflow.end_run()
