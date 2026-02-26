"""
Tests for the PyTorchTrainer and LimitedImageDataset.
"""

import os
import tempfile
import pytest
from PIL import Image

torch_available = False
try:
    import torch
    from agent.pytorch_trainer import PyTorchTrainer, LimitedImageDataset
    torch_available = True
except ImportError:
    pass

@pytest.fixture
def mock_dataset():
    temp_dir = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(temp_dir.name, "NORMAL"))
    os.makedirs(os.path.join(temp_dir.name, "PNEUMONIA"))
    
    # Create fake images
    for i in range(10):
        img_normal = Image.new("RGB", (64, 64), color="black")
        img_normal.save(os.path.join(temp_dir.name, "NORMAL", f"normal_{i}.jpeg"))
        img_pneumo = Image.new("RGB", (64, 64), color="white")
        img_pneumo.save(os.path.join(temp_dir.name, "PNEUMONIA", f"pneumo_{i}.jpeg"))
        
    yield temp_dir.name
    temp_dir.cleanup()

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
def test_limited_dataset(mock_dataset):
    dataset = LimitedImageDataset(mock_dataset, max_samples=10)
    assert len(dataset) <= 10
    assert len(dataset.samples) == 10
    
    img, lbl = dataset[0]
    assert img is not None
    assert lbl in [0, 1]

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
def test_pytorch_trainer_initialization(mock_dataset):
    # Just need base dir, _load_data automatically looks for "train" folder.
    mock_base = os.path.dirname(mock_dataset)
    trainer = PyTorchTrainer(data_dir=mock_base)
    assert trainer.model is not None

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
def test_train_interface(mock_dataset):
    mock_base = os.path.dirname(mock_dataset)
    trainer = PyTorchTrainer(data_dir=mock_base)
    res = trainer.train("test_agent_1", {"weights": []})
    assert "weights" in res
    assert "intercept" in res
    assert "metrics" in res
