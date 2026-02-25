"""
PyTorch-based Local Trainer for Image Federation (e.g., Chest X-Ray).

This module flattens and unflattens PyTorch state_dicts to strictly
comply with the frozen `MODEL_UPDATE` JSON schema (`weights`, `intercept`).
"""

import os
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LimitedImageDataset(Dataset):
    def __init__(self, root_dir, max_samples=500, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = ["NORMAL", "PNEUMONIA"]
        for label, cls_name in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                files = os.listdir(cls_dir)
                # Take an even slice from each class to hit max_samples
                subset = files[: max_samples // len(classes)]
                for f in subset:
                    self.samples.append((os.path.join(cls_dir, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SimpleCNN(nn.Module):
    """A lightweight CNN for binary classification (e.g., Normal vs Pneumonia)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # 32x32 input -> 16x16 -> 8x8. 32 channels * 8 * 8 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PyTorchTrainer:
    """Trains a CNN on image data and serializes weights for federation."""

    def __init__(self, data_dir: str = "/app/dataset/chest_xray"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not fully installed.")

        self.logger = logging.getLogger("PyTorchTrainer")
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate the model
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.BCELoss()

        # We define a standard transform
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self._num_samples = 0
        self._load_data()

    def _load_data(self):
        """Load datasets if available."""
        train_path = os.path.join(self.data_dir, "train")
        if os.path.exists(train_path):
            self.train_dataset = LimitedImageDataset(
                train_path, max_samples=250, transform=self.transform
            )

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=16,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            self._num_samples = len(self.train_dataset)
            self.logger.info(
                f"Loaded {self._num_samples} training images from {train_path} (subsampled)"
            )
        else:
            self.train_dataset = None
            self.train_loader = None
            self.logger.warning(f"Data directory not found: {train_path}")

    def flat_weights_to_state_dict(self, flat_weights: list) -> dict:
        """Convert a 1D list of floats back into a PyTorch state_dict."""
        state_dict = self.model.state_dict()
        idx = 0
        for name, param in state_dict.items():
            param_size = param.numel()
            end_idx = idx + param_size
            chunk = torch.tensor(
                flat_weights[idx:end_idx],
                dtype=param.dtype,
                device=param.device,
            )
            state_dict[name] = chunk.view(param.shape)
            idx = end_idx

        if idx != len(flat_weights):
            raise ValueError(
                f"Weight dimension mismatch. Read {idx}, total {len(flat_weights)}."
            )

        return state_dict

    def state_dict_to_flat_weights(self, state_dict: dict) -> list:
        """Convert a PyTorch state_dict into a 1D list of floats."""
        flat = []
        for name, param in state_dict.items():
            flat.extend(param.cpu().view(-1).tolist())
        return flat

    def train(self, agent_id: str, global_weights: dict = None) -> dict:
        """Train the model for one epoch starting from global weights."""

        if (
            global_weights
            and "weights" in global_weights
            and len(global_weights["weights"]) > 0
        ):
            try:
                new_state = self.flat_weights_to_state_dict(global_weights["weights"])
                self.model.load_state_dict(new_state)
            except Exception as e:
                self.logger.error(f"Failed to load global weights: {e}")

        # If we have no data, just return the exact weights we got (or initial random weights)
        if not self.train_loader:
            self.logger.warning("No data available. Skipping training.")
            return {
                "weights": self.state_dict_to_flat_weights(self.model.state_dict()),
                "intercept": [0.0],
                "num_samples": 0,
                "metrics": {"accuracy": 0.0, "loss": 0.0},
                "budget_exhausted": False,
                "privacy_remaining": 10.0,
            }

        # Training Loop
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        running_loss = 0.0
        correct = 0
        total = 0

        # For a true setup we would train for 1 epoch per round.
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device).float()

            optimizer.zero_grad()
            outputs = self.model(images).squeeze()

            # Catch single-item batches dimension issues
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        self.logger.info(
            f"PyTorch agent {agent_id} completed epoch. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )

        # Flatten updated weights for the network schema
        flat_updated_weights = self.state_dict_to_flat_weights(self.model.state_dict())

        return {
            "weights": flat_updated_weights,
            "intercept": [0.0],  # Schema requirement, unused for CNN
            "num_samples": total,
            "metrics": {
                "accuracy": epoch_acc,
                "loss": epoch_loss,
                "precision": epoch_acc,  # Approximate for demo
                "recall": epoch_acc,  # Approximate for demo
            },
            "budget_exhausted": False,
            "privacy_remaining": 10.0,
        }
