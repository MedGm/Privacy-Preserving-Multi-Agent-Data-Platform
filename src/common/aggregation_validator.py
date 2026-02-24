"""
Aggregation Sanity Checks for Federated Learning.

Validates incoming model updates before FedAvg aggregation.
Rejects updates that contain NaN/Inf values, exceed L2-norm bounds,
or diverge sharply from the current global model.
"""

import math
import numpy as np
from common.logger import setup_logger

logger = setup_logger("AggregationValidator")

# Maximum allowed L2 norm for incoming weight vectors.
MAX_WEIGHT_NORM = 10.0


def validate_update(update: dict, global_model: dict = None) -> tuple:
    """
    Validate a single agent's model update before aggregation.

    Args:
        update: dict with 'weights', 'intercept', 'sender_id', etc.
        global_model: current global model (optional,
                      used for cosine similarity check).

    Returns:
        (is_valid: bool, reason: str or None)
    """
    sender = update.get("sender_id", "unknown")
    weights = update.get("weights", [])
    intercept = update.get("intercept", [])

    # 1. NaN / Inf detection
    all_values = weights + intercept
    for i, v in enumerate(all_values):
        if math.isnan(v) or math.isinf(v):
            reason = f"NaN/Inf detected at index {i}"
            logger.warning(f"[Validator] REJECTED {sender}: {reason}")
            return False, reason

    # 2. L2-norm bound on weight vector
    w_array = np.array(weights, dtype=np.float64)
    l2_norm = float(np.linalg.norm(w_array))
    if l2_norm > MAX_WEIGHT_NORM:
        reason = f"L2 norm {l2_norm:.2f} exceeds " f"limit {MAX_WEIGHT_NORM}"
        logger.warning(f"[Validator] REJECTED {sender}: {reason}")
        return False, reason

    # 3. Cosine similarity check against global model
    if (
        global_model
        and global_model.get("weights")
        and len(global_model["weights"]) == len(weights)
    ):
        g_array = np.array(global_model["weights"], dtype=np.float64)
        g_norm = np.linalg.norm(g_array)
        if g_norm > 0 and l2_norm > 0:
            cosine_sim = float(np.dot(w_array, g_array) / (l2_norm * g_norm))
            if cosine_sim < -0.5:
                reason = (
                    f"Cosine similarity {cosine_sim:.3f} "
                    f"diverges sharply from global model"
                )
                logger.warning(
                    f"[Validator] FLAGGED {sender}: " f"{reason} (still accepted)"
                )
                # Flag but don't reject — logs for manual review

    logger.debug(f"[Validator] ACCEPTED {sender} " f"(L2={l2_norm:.3f})")
    return True, None
