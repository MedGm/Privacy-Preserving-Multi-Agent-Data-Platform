"""
Unit tests for Differential Privacy Module.

Tests validate:
  - L2 norm clipping behavior
  - Noise addition changes weights
  - Privacy budget tracking semantics
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common.privacy import clip_weights, add_dp_noise, PrivacyAccountant  # noqa: E402


def test_clip_weights_no_op():
    weights = [0.5, 0.5]
    # Norm is sqrt(0.5) < 1.0, so no clipping
    clipped = clip_weights(weights, max_norm=1.0)
    assert np.allclose(clipped, weights)

def test_clip_weights_scales_down():
    weights = [3.0, 4.0]
    # Norm is 5.0. If max_norm=1.0, should be [0.6, 0.8]
    clipped = clip_weights(weights, max_norm=1.0)
    assert np.allclose(clipped, [0.6, 0.8])

def test_add_dp_noise_changes_weights():
    weights = [1.0, 1.0]
    noisy = add_dp_noise(weights, epsilon=0.1, sensitivity=1.0)
    assert not np.allclose(weights, noisy)

def test_add_dp_noise_zero_epsilon_no_op():
    weights = [1.0, 1.0]
    noisy = add_dp_noise(weights, epsilon=0.0, sensitivity=1.0)
    assert np.allclose(weights, noisy)

def test_privacy_accountant():
    accountant = PrivacyAccountant(target_epsilon=1.0)
    assert accountant.remaining() == 1.0
    
    # Spend 0.5
    assert accountant.spend(0.5) is True
    assert accountant.remaining() == 0.5
    
    # Try to spend another 0.6
    assert accountant.spend(0.6) is False
    assert accountant.remaining() == 0.5
    
    # Spend remaining exactly
    assert accountant.spend(0.5) is True
    assert accountant.remaining() == 0.0
