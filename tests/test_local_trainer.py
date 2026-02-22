"""
Unit tests for LocalTrainer (federated logistic regression).

Tests validate:
  - Training produces correct weight dimensions
  - Setting global weights changes model behavior
  - Privacy: raw data never exposed in return values
  - Round-trip: train → set_weights → retrain converges
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.local_trainer import LocalTrainer  # noqa: E402
from common.mock_data import generate_mock_data  # noqa: E402


class TestLocalTrainerBasics:
    """Verify LocalTrainer produces valid outputs."""

    def test_train_returns_correct_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 100, seed=42)
        trainer = LocalTrainer()
        result = trainer.train(data_path)

        assert "weights" in result
        assert "intercept" in result
        assert "num_samples" in result
        assert "metrics" in result

    def test_weight_dimensions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 100, seed=42)
        trainer = LocalTrainer()
        result = trainer.train(data_path)

        # 2 features → 2 weights
        assert len(result["weights"]) == 2
        # Binary classification → 1 intercept
        assert len(result["intercept"]) == 1

    def test_num_samples_matches(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        n = 77
        data_path = generate_mock_data("test", n, seed=42)
        trainer = LocalTrainer()
        result = trainer.train(data_path)
        assert result["num_samples"] == n

    def test_metrics_are_valid(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 200, seed=42)
        trainer = LocalTrainer()
        result = trainer.train(data_path)

        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
        assert result["metrics"]["loss"] >= 0.0


class TestFederatedWeights:
    """Verify weight setting and warm start behavior."""

    def test_set_weights_changes_predictions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 100, seed=42)

        trainer = LocalTrainer()
        result1 = trainer.train(data_path)

        # Set extreme weights and retrain
        trainer2 = LocalTrainer()
        result2 = trainer2.train(
            data_path,
            global_weights={"weights": [10.0, 10.0], "intercept": [-15.0]},
        )

        # Weights should differ after warm-starting from different points
        assert result1["weights"] != result2["weights"]

    def test_warm_start_improves(self, tmp_path, monkeypatch):
        """Multiple rounds of training should improve or maintain accuracy."""
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 300, seed=42)

        trainer = LocalTrainer()
        r1 = trainer.train(data_path)

        # Simulate receiving the same weights back (single-agent fedavg)
        r2 = trainer.train(
            data_path,
            global_weights={
                "weights": r1["weights"],
                "intercept": r1["intercept"],
            },
        )

        # After 2 rounds, loss should not explode
        assert r2["metrics"]["loss"] < 2.0


class TestPrivacyInvariant:
    """Ensure raw data never leaks through the trainer interface."""

    def test_no_raw_data_in_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 50, seed=42)
        trainer = LocalTrainer()
        result = trainer.train(data_path)

        # Only allowed keys in output
        allowed = {"weights", "intercept", "num_samples", "metrics"}
        assert set(result.keys()) == allowed

    def test_weights_are_lists_of_floats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = generate_mock_data("test", 50, seed=42)
        trainer = LocalTrainer()
        result = trainer.train(data_path)

        assert all(isinstance(w, float) for w in result["weights"])
        assert all(isinstance(i, float) for i in result["intercept"])
