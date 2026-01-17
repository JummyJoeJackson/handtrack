"""
Training Script for ASL Landmark Model

Handles the complete pipeline:
1. Load CSV data
2. Convert to NumPy arrays
3. Apply normalization
4. Convert to PyTorch tensors
5. Train the model
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from model import ASLLandmarkMLP, ASLLandmarkNet, normalize_landmarks, ASL_CLASSES


class ASLLandmarkDataset(Dataset):
    """
    Dataset class that handles CSV loading and normalization.

    Expected CSV format:
    - Columns: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
    - Or: 63 coordinate columns + 1 label column
    """

    def __init__(self, csv_path: str, apply_normalization: bool = True):
        """
        Args:
            csv_path: Path to the CSV file with landmark data
            apply_normalization: Whether to normalize landmarks (recommended)
        """
        self.apply_normalization = apply_normalization

        # Load CSV
        df = pd.read_csv(csv_path)

        # Check if label is first column (main.py format) or last column
        if df.columns[0] == 'label':
            # main.py format: label is first column
            self.labels = df.iloc[:, 0].values
            self.landmarks = df.iloc[:, 1:].values.astype(np.float32)
        else:
            # Alternative format: label is last column
            self.labels = df.iloc[:, -1].values
            self.landmarks = df.iloc[:, :-1].values.astype(np.float32)

        # Create label to index mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        print(f"Loaded {len(self.landmarks)} samples")
        print(f"Classes ({self.num_classes}): {unique_labels}")

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Get raw landmarks as NumPy array
        landmarks = self.landmarks[idx]

        # Step 2: Apply normalization (NumPy operation)
        if self.apply_normalization:
            landmarks = normalize_landmarks(landmarks)

        # Step 3: Convert to PyTorch tensor
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

        # Step 4: Convert label to index tensor
        label_idx = self.label_to_idx[self.labels[idx]]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return landmarks_tensor, label_tensor


def train_model(
    csv_path: str,
    model_type: str = "mlp",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    save_path: str = None,
    device: str = None
):
    """
    Train the ASL landmark model.

    Args:
        csv_path: Path to training CSV file
        model_type: "mlp" or "cnn"
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data for validation
        save_path: Where to save the trained model
        device: "cuda" or "cpu" (auto-detected if None)
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    # Note: main.py already normalizes data (center + scale), so skip extra normalization
    dataset = ASLLandmarkDataset(csv_path, apply_normalization=False)

    # Split into train/validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    if model_type == "cnn":
        model = ASLLandmarkNet(num_classes=dataset.num_classes)
    else:
        model = ASLLandmarkMLP(num_classes=dataset.num_classes)

    model = model.to(device)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for landmarks, labels in train_loader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for landmarks, labels in val_loader:
                landmarks = landmarks.to(device)
                labels = labels.to(device)

                outputs = model(landmarks)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss/len(val_loader):.4f} "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': model_type,
                    'num_classes': dataset.num_classes,
                    'label_to_idx': dataset.label_to_idx,
                    'idx_to_label': dataset.idx_to_label,
                }, save_path)
                print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model, dataset.idx_to_label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ASL Landmark Model")
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"],
                        help="Model type: mlp (recommended) or cnn")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save", type=str, default="models/asl_model.pth",
                        help="Path to save trained model")

    args = parser.parse_args()

    # Create models directory if needed
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    train_model(
        csv_path=args.csv,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save
    )
