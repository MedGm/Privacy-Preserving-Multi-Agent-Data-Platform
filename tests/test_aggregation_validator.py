"""Tests for aggregation sanity checks."""

from common.aggregation_validator import validate_update


def test_valid_update_accepted():
    update = {
        "weights": [0.1, 0.2, -0.3],
        "intercept": [0.05],
        "sender_id": "agent1",
    }
    is_valid, reason = validate_update(update)
    assert is_valid is True
    assert reason is None


def test_nan_rejected():
    update = {
        "weights": [0.1, float("nan"), -0.3],
        "intercept": [0.05],
        "sender_id": "agent_bad",
    }
    is_valid, reason = validate_update(update)
    assert is_valid is False
    assert "NaN" in reason


def test_inf_rejected():
    update = {
        "weights": [0.1, float("inf"), -0.3],
        "intercept": [0.05],
        "sender_id": "agent_bad",
    }
    is_valid, reason = validate_update(update)
    assert is_valid is False
    assert "Inf" in reason or "NaN" in reason


def test_large_norm_rejected():
    # Create a weight vector with L2 norm > 10.0
    update = {
        "weights": [100.0] * 30,
        "intercept": [0.05],
        "sender_id": "agent_big",
    }
    is_valid, reason = validate_update(update)
    assert is_valid is False
    assert "L2 norm" in reason


def test_normal_norm_accepted():
    update = {
        "weights": [0.5] * 30,
        "intercept": [0.05],
        "sender_id": "agent_ok",
    }
    is_valid, reason = validate_update(update)
    assert is_valid is True
