"""
Unit tests for mock data generation.

Tests validate:
  - CSV is created with expected schema (feature_1, feature_2, label)
  - Agent isolation: each agent gets its own file path
  - Correct number of rows generated
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.mock_data import generate_mock_data  # noqa: E402


class TestMockDataGeneration:
    """Verify mock dataset generation correctness."""

    def test_file_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = generate_mock_data("test_agent", 10)
        assert os.path.exists(path)

    def test_csv_header(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = generate_mock_data("test_agent", 5)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["feature_1", "feature_2", "label"]

    def test_correct_row_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        num = 42
        path = generate_mock_data("test_agent", num)
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        assert len(rows) == num

    def test_agent_isolation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p1 = generate_mock_data("alice", 5)
        p2 = generate_mock_data("bob", 5)
        assert p1 != p2
        assert "alice" in p1
        assert "bob" in p2

    def test_label_values(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = generate_mock_data("test_agent", 200)
        labels = set()
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.add(int(row["label"]))
        assert labels == {0, 1}
