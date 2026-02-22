"""
Local Trainer for Federated Learning.

Wraps scikit-learn SGDClassifier (logistic regression via SGD) for local
training on agent-private data. Only model parameters leave the agent.

Privacy Invariant: raw data D_i is NEVER exposed. Only (coef, intercept,
num_samples, metrics) are returned.
"""

import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from common.logger import setup_logger
from common.privacy import clip_weights, add_dp_noise
from common.data_loader import get_agent_data

logger = setup_logger("LocalTrainer")


class LocalTrainer:
    """Federated-compatible local trainer using logistic regression (SGD)."""

    def __init__(self):
        self.model = SGDClassifier(
            loss="log_loss",
            max_iter=5,
            warm_start=True,
            random_state=42,
            learning_rate="constant",
            eta0=0.1,
        )
        self._fitted = False
        self.dp_epsilon = float(os.environ.get("DP_EPSILON", 1.0))
        self.dp_clip_norm = float(os.environ.get("DP_CLIP_NORM", 1.0))

    def set_weights(self, coef: list, intercept: list) -> None:
        """Apply global model weights (FedAvg update) before local training."""
        self.model.coef_ = np.array([coef], dtype=np.float64)
        self.model.intercept_ = np.array(intercept, dtype=np.float64)
        self.model.classes_ = np.array([0, 1])
        self._fitted = True

    def train(self, agent_id: str, global_weights: dict = None) -> dict:
        """
        Train on local data and return ONLY model parameters + metrics.

        Args:
            agent_id: Identifier of the agent string.
            global_weights: Optional dict with 'weights' and 'intercept'
                            from coordinator's global model.

        Returns:
            dict with keys: weights, intercept, num_samples, metrics
        """
        # Parse agent index from agent_id (e.g., "agent1" -> 0)
        try:
            agent_index = int(str(agent_id).replace("agent", "")) - 1
        except ValueError:
            agent_index = 0

        # Get data shard securely
        X_train, X_test, y_train, y_test = get_agent_data(
            agent_index, total_agents=5, non_iid=True
        )

        # Apply global model if provided (warm start for FedAvg)
        if global_weights and "weights" in global_weights:
            self.set_weights(
                global_weights["weights"],
                global_weights.get("intercept", [0.0]),
            )

        # Local training
        self.model.partial_fit(X_train, y_train, classes=[0, 1])
        self._fitted = True

        # Compute local metrics (on test data for realistic tracking)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        loss = float(log_loss(y_test, y_proba, labels=[0, 1]))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))

        logger.info(
            f"Local training complete: acc={accuracy:.4f}, loss={loss:.4f}, prec={precision:.4f}, rec={recall:.4f}"
        )

        weights = self.model.coef_[0].tolist()
        intercept = self.model.intercept_.tolist()

        if self.dp_epsilon > 0.0:
            logger.info(
                f"Applying DP noise (epsilon={self.dp_epsilon}, clip={self.dp_clip_norm})"
            )
            weights = clip_weights(weights, self.dp_clip_norm)
            weights = add_dp_noise(weights, self.dp_epsilon, self.dp_clip_norm)
            intercept = clip_weights(intercept, self.dp_clip_norm)
            intercept = add_dp_noise(intercept, self.dp_epsilon, self.dp_clip_norm)

        # Privacy boundary: only parameters and aggregate metrics leave
        return {
            "weights": weights,
            "intercept": intercept,
            "num_samples": len(y_train),
            "metrics": {
                "accuracy": accuracy,
                "loss": loss,
                "precision": precision,
                "recall": recall,
            },
        }
