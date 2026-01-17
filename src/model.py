import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASLLandmarkNet(nn.Module):
    def __init__(self, num_landmarks: int = 21, num_coords: int = 3, num_classes: int = 29):
        super(ASLLandmarkNet, self).__init__()

        self.num_landmarks = num_landmarks
        self.num_coords = num_coords

        self.conv1 = nn.Conv1d(num_coords, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_landmarks, self.num_coords)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
        return predicted, confidence


class ASLLandmarkMLP(nn.Module):
    def __init__(self, num_landmarks: int = 21, num_coords: int = 3, num_classes: int = 29):
        super(ASLLandmarkMLP, self).__init__()

        input_features = num_landmarks * num_coords

        self.model = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
        return predicted, confidence


ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    landmarks = np.array(landmarks).reshape(21, 3)

    wrist = landmarks[0]
    centered = landmarks - wrist

    scale = np.linalg.norm(centered[9])
    if scale > 0:
        centered = centered / scale

    return centered.flatten()


def get_model(model_type: str = "cnn", num_classes: int = 29, pretrained_path: str = None) -> nn.Module:
    if model_type == "mlp":
        model = ASLLandmarkMLP(num_classes=num_classes)
    else:
        model = ASLLandmarkNet(num_classes=num_classes)

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))

    return model


if __name__ == "__main__":
    print("Testing ASLLandmarkNet (1D CNN)...")
    model = ASLLandmarkNet()
    x = torch.randn(4, 63)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting ASLLandmarkMLP...")
    model_mlp = ASLLandmarkMLP()
    out_mlp = model_mlp(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_mlp.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_mlp.parameters()):,}")

    print("\nTesting prediction...")
    pred, conf = model.predict(x)
    print(f"  Predictions: {pred}")
    print(f"  Confidences: {conf}")
    print(f"  Letters: {[ASL_CLASSES[p] for p in pred.tolist()]}")
