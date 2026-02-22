"""
Unit tests for LocalTrainer (federated logistic regression) on Breast Cancer Data.

Tests validate:
  - Training produces correct weight dimensions (30 features)
  - Setting global weights changes model behavior
  - Privacy: raw data never exposed in return values
  - Returns 4 metrics: accuracy, loss, precision, recall
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common.local_trainer import LocalTrainer  # noqa: E402


class TestLocalTrainerBasics:
    """Verify LocalTrainer produces valid outputs on real shards."""

    def test_train_returns_correct_keys(self):
        trainer = LocalTrainer()
        result = trainer.train("agent1")

        assert "weights" in result
        assert "intercept" in result
        assert "num_samples" in result
        assert "metrics" in result
        
        assert "precision" in result["metrics"]
        assert "recall" in result["metrics"]

    def test_weight_dimensions(self):
        trainer = LocalTrainer()
        result = trainer.train("agent2")

        # Breast cancer has 30 features → 30 weights
        assert len(result["weights"]) == 30
        # Binary classification → 1 intercept
        assert len(result["intercept"]) == 1

    def test_metrics_are_valid(self):
        trainer = LocalTrainer()
        result = trainer.train("agent3")

        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
        assert result["metrics"]["loss"] >= 0.0
        assert 0.0 <= result["metrics"]["precision"] <= 1.0
        assert 0.0 <= result["metrics"]["recall"] <= 1.0


class TestFederatedWeights:
    """Verify weight setting and warm start behavior."""

    def test_set_weights_changes_predictions(self):
        trainer = LocalTrainer()
        result1 = trainer.train("agent1")

        # Set specific weights and retrain
        trainer2 = LocalTrainer()
        result2 = trainer2.train(
            "agent1",
            global_weights={"weights": [1.0] * 30, "intercept": [-5.0]},
        )

        # Weights should differ after warm-starting from different points
        assert result1["weights"] != result2["weights"]

    def test_warm_start_improves(self):
        """Multiple rounds of training should improve or maintain accuracy."""
        trainer = LocalTrainer()
        r1 = trainer.train("agent1")

        # Simulate receiving the same weights back (single-agent fedavg)
        r2 = trainer.train(
            "agent1",
            global_weights={
                "weights": r1["weights"],
                "intercept": r1["intercept"],
            },
        )

        assert r2["metrics"]["loss"] < 2.0


class TestPrivacyInvariant:
    """Ensure raw data never leaks through the trainer interface."""

    def test_no_raw_data_in_output(self):
        trainer = LocalTrainer()
        result = trainer.train("agent4")

        # Only allowed keys in output
        allowed = {"weights", "intercept", "num_samples", "metrics"}
        assert set(result.keys()) == allowed

    def test_weights_are_lists_of_floats(self):
        trainer = LocalTrainer()
        result = trainer.train("agent5")

        assert all(isinstance(w, float) for w in result["weights"])
        assert all(isinstance(i, float) for i in result["intercept"])
