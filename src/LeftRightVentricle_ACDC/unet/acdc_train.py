import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.LeftRightVentricle_ACDC.unet.acdc_torchdataset import ACDCDataset
from src.LeftRightVentricle_ACDC.acdc_datasetloader import load_acdc_dataset


EPOCHS = 10
BATCH_SIZE = 4
LR = 0.001

DATA_ROOT = "../../data/raw/LRVentricle/training"

MODEL_LV_PATH = "model_lv.pth"
MODEL_RV_PATH = "model_rv.pth"


def train_acdc(X, Y, model_name, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    dataset = ACDCDataset(X, Y, size=(512, 428))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet()

    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
        print(f"Model existent încărcat: {model_name}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n==============================")
    print(f"ACDC TRAINING: {model_name}")
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
                    f"{model_name} | "
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {i}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"{model_name} | Epoch {epoch + 1}/{epochs} completed | Average loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_name)
    print(f"\nModel salvat: {model_name}")

    return model


def main():
    print("\n==============================")
    print("LOADING ACDC DATASET")
    print("==============================")

    X_lv, Y_lv = load_acdc_dataset(DATA_ROOT, target="LV")
    X_rv, Y_rv = load_acdc_dataset(DATA_ROOT, target="RV")

    print("\n==============================")
    print("START TRAINING LV")
    print("==============================")
    train_acdc(X_lv, Y_lv, MODEL_LV_PATH)

    print("\n==============================")
    print("START TRAINING RV")
    print("==============================")
    train_acdc(X_rv, Y_rv, MODEL_RV_PATH)

    print("\n==============================")
    print("ACDC TRAINING COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()