from pathlib import Path

import torch
from torch import nn

from ml_project.model import SimpleClassifier


def train():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training on: {device}")

    model = SimpleClassifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.CrossEntropyLoss()

    # Synthetic classification dataset
    X = torch.randn(1000, 10, device=device)

    y = (X[:, 0] + X[:, 1] > 0).long()

    for epoch in range(20):
        model.train()

        optimizer.zero_grad()

        predictions = model(X)

        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1}/20 "
            f"Loss: {loss.item():.4f}"
        )

    # Evaluate on the training data for this first exercise.
    model.eval()

    with torch.no_grad():
        predictions = model(X)
        accuracy = (
            (predictions.argmax(dim=1) == y)
            .float()
            .mean()
            .item()
        )

    print(f"Training accuracy: {accuracy:.4f}")

    # Save checkpoint
    Path("artifacts").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        "artifacts/model.pth",
    )

    print("Model saved successfully!")


if __name__ == "__main__":
    train()