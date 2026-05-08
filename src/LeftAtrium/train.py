import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.LeftAtrium.torchdataset import HeartDataset


EPOCHS = 30
BATCH_SIZE = 4
LR = 0.001
MODEL_PATH = "model_leftatrium.pth"


def train_model(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    dataset = HeartDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"Model existent încărcat: {MODEL_PATH}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n==============================")
    print("LEFT ATRIUM TRAINING")
    print("==============================")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("==============================\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (images, masks) in enumerate(dataloader):
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {i}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} completed | Average loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel salvat: {MODEL_PATH}")

    return model