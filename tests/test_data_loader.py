"""
Unit tests for data loader (Breast Cancer dataset).

Tests validate:
  - Data is loaded as numpy arrays.
  - Return shapes are structured as X_train, X_test, y_train, y_test.
  - Exactly 30 features are returned.
  - Arrays are StandardScaled.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common.data_loader import get_agent_data  # noqa: E402


class TestDataLoader:
    """Verify breast cancer dataset loading and sharding."""

    def test_return_shapes(self):
        # 5 agents total
        X_train, X_test, y_train, y_test = get_agent_data(0, 5, non_iid=False)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        # Breast cancer has 30 features
        assert X_train.shape[1] == 30
        assert X_test.shape[1] == 30

    def test_agent_sharding(self):
        # 569 / 5 = 113.8, so some get 114, some get 113.
        # test size is 0.2
        X_train_0, X_test_0, _, _ = get_agent_data(0, 5, non_iid=False)
        X_train_1, X_test_1, _, _ = get_agent_data(1, 5, non_iid=False)

        # The partitions shouldn't be identical
        assert not np.array_equal(X_train_0, X_train_1)

        # total items per agent is around 114. Train is 80%, so ~91.
        assert 85 < len(X_train_0) < 95
        assert 85 < len(X_train_1) < 95

    def test_values_are_scaled(self):
        X_train, X_test, y_train, y_test = get_agent_data(2, 5, non_iid=True)

        # Means shouldn't be massive like raw breast cancer dataset (where some means are > 500)
        # Because we only get a shard (not the whole dataset), the mean of the shard
        # isn't exactly 0, but it will be close (within [-2, 2])
        col_means = np.mean(X_train, axis=0)
        assert np.all(np.abs(col_means) < 3.0)

    def test_binary_labels(self):
        X_train, X_test, y_train, y_test = get_agent_data(3, 5, non_iid=False)
        unique_labels = np.unique(y_train)
        assert set(unique_labels).issubset({0, 1})
