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
    def __init__(
        self, tracking_uri: str = None, experiment_name: str = "PmadFederation"
    ):
        self.enabled = MLFLOW_AVAILABLE
        if not self.enabled:
            logger.warning("MLflow not installed. Tracking disabled.")
            return

        tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Start a new run
        self.run = mlflow.start_run()
        logger.info(f"MLflow tracking started. Run ID: {self.run.info.run_id}")

    def log_setup(self, params: dict, tags: dict = None):
        if not self.enabled:
            return

        # Log Hyperparameters
        mlflow.log_params(params)

        # Log contextual tags
        if tags:
            mlflow.set_tags(tags)
        else:
            mlflow.set_tags(
                {
                    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
                    "task": "Federated Logistic Regression",
                    "framework": "SPADE + Flask + PySpark/Scikit-Learn",
                }
            )

    def log_round(
        self, round_id: int, metrics: dict, num_agents: int, agent_metrics: list = None
    ):
        if not self.enabled:
            return

        # Prepare payload
        payload = {
            "avg_accuracy": metrics.get("accuracy", 0.0),
            "avg_loss": metrics.get("loss", 0.0),
            "avg_precision": metrics.get("precision", 0.0),
            "avg_recall": metrics.get("recall", 0.0),
            "agents_participating": num_agents,
        }

        # Dynamically inject individual node metrics
        if agent_metrics:
            for update in agent_metrics:
                node = update.get("sender_id", "unknown_agent").split("@")[0]
                node_m = update.get("metrics", {})
                payload[f"{node}_accuracy"] = node_m.get("accuracy", 0.0)
                payload[f"{node}_loss"] = node_m.get("loss", 0.0)
                payload[f"{node}_precision"] = node_m.get("precision", 0.0)
                payload[f"{node}_recall"] = node_m.get("recall", 0.0)

        # Log all metrics with round as step
        mlflow.log_metrics(payload, step=round_id)

    def log_privacy(self, round_id: int, min_remaining: float):
        if not self.enabled:
            return
        mlflow.log_metrics(
            {"min_privacy_remaining": min_remaining},
            step=round_id,
        )

    def log_final_model(self, global_model: dict, metrics: dict = None):
        if not self.enabled:
            return

        try:
            # Save model artifact
            with open("final_model.json", "w") as f:
                json.dump(global_model, f, indent=4)
            mlflow.log_artifact("final_model.json")

            # Save run configuration
            config = {
                "dataset": "UCI Breast Cancer Wisconsin",
                "features": len(global_model.get("weights", [])),
                "differential_privacy": "Active",
                "architecture": "FedAvg via Scikit/PySpark",
            }
            with open("run_config.json", "w") as f:
                json.dump(config, f, indent=4)
            mlflow.log_artifact("run_config.json")

            # Register model in MLflow Model Registry
            model_name = "FederatedGlobalModel"
            run_id = self.run.info.run_id
            artifact_uri = f"runs:/{run_id}/final_model.json"

            try:
                result = mlflow.register_model(
                    artifact_uri, model_name
                )
                version = result.version
                logger.info(
                    f"Model registered: {model_name} "
                    f"v{version}"
                )

                # Tag the version with final metrics
                client = mlflow.tracking.MlflowClient()
                if metrics:
                    for k, v in metrics.items():
                        client.set_model_version_tag(
                            model_name,
                            version,
                            k,
                            f"{v:.4f}",
                        )
                client.set_model_version_tag(
                    model_name, version, "run_id", run_id
                )
            except Exception as reg_err:
                logger.warning(
                    f"Model registry unavailable: {reg_err}. "
                    f"Artifact still logged."
                )

            logger.info(
                "Final model and config registered in MLflow."
            )
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
        finally:
            mlflow.end_run()

    @staticmethod
    def get_model_history():
        """Retrieve all registered model versions."""
        if not MLFLOW_AVAILABLE:
            return []
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                "name='FederatedGlobalModel'"
            )
            history = []
            for v in versions:
                entry = {
                    "version": v.version,
                    "run_id": v.run_id,
                    "status": v.status,
                    "created": str(v.creation_timestamp),
                    "tags": dict(v.tags) if v.tags else {},
                }
                history.append(entry)
            return sorted(
                history,
                key=lambda x: int(x["version"]),
                reverse=True,
            )
        except Exception:
            return []
