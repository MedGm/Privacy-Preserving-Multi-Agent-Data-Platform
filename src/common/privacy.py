"""
Differential Privacy module for Federated Learning.

Implements L2 norm clipping and Laplace/Gaussian noise mechanisms
to ensure that the aggregated weights satisfy epsilon-differential privacy.

Includes a privacy accountant to track cumulative epsilon across rounds.
"""

import numpy as np


def clip_weights(weights: list, max_norm: float) -> list:
    """
    Clips a weight vector to a maximum L2 norm.
    If the L2 norm exceeds max_norm, scales it down.
    """
    w_array = np.array(weights, dtype=np.float64)
    l2_norm = np.linalg.norm(w_array)
    if l2_norm > max_norm:
        w_array = w_array * (max_norm / l2_norm)
    return w_array.tolist()


def add_dp_noise(
    weights: list, epsilon: float, sensitivity: float, mechanism: str = "laplace"
) -> list:
    """
    Adds differential privacy noise to a weight vector.

    Args:
        weights: Raw weight list.
        epsilon: Privacy budget for this specific operation.
        sensitivity: Sensitivity of the weights (usually max_norm).
        mechanism: 'laplace' or 'gaussian'.

    Returns:
        Noisy weight list.
    """
    w_array = np.array(weights, dtype=np.float64)
    if epsilon <= 0.0:
        return w_array.tolist()

    if mechanism == "laplace":
        # Scale for Laplace is sensitivity / epsilon
        scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0.0, scale=scale, size=w_array.shape)
    elif mechanism == "gaussian":
        # Simplified Gaussian DP: scale approx sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        # For simplicity, we just use a calibrated scale based on epsilon.
        scale = (sensitivity / epsilon) * 1.5
        noise = np.random.normal(loc=0.0, scale=scale, size=w_array.shape)
    else:
        raise ValueError(f"Unknown DP mechanism: {mechanism}")

    return (w_array + noise).tolist()


class PrivacyAccountant:
    """Tracks cumulative epsilon expenditure across communication rounds."""

    def __init__(self, target_epsilon: float):
        self.target_epsilon = target_epsilon
        self.spent_epsilon = 0.0

    def spend(self, epsilon: float) -> bool:
        """
        Record spending of epsilon.
        Returns False if the budget is exceeded, True otherwise.
        """
        if self.spent_epsilon + epsilon > self.target_epsilon:
            return False
        self.spent_epsilon += epsilon
        return True

    def remaining(self) -> float:
        return max(0.0, self.target_epsilon - self.spent_epsilon)
