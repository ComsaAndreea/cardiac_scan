import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.attention_unet import AttentionUNet
from src.LeftRightVentricle.acdc_torchdataset import ACDCDataset
from src.LeftRightVentricle.acdc_datasetloader import load_acdc_dataset


EPOCHS = 30
BATCH_SIZE = 4
LR = 0.001
TARGET_SIZE = (512, 448)

ACDC_DATA_ROOT = Path(
    os.environ.get(
        "ACDC_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "ResourcesACDC")
    )
)

DATA_ROOT = ACDC_DATA_ROOT / "training"

MODEL_LV_PATH = PROJECT_ROOT / "model_lv_attention.pth"
MODEL_RV_PATH = PROJECT_ROOT / "model_rv_attention.pth"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def train_acdc_attention(X, Y, model_path, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    device = get_device()

    dataset = ACDCDataset(X, Y, size=TARGET_SIZE)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = AttentionUNet().to(device)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model existent încărcat: {model_path}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n==============================")
    print(f"ACDC ATTENTION U-NET TRAINING: {model_path.name}")
    print("==============================")
    print(f"Data root: {DATA_ROOT}")
    print(f"Model path: {model_path}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("==============================\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"{model_path.name} | "
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {i}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(
            f"{model_path.name} | "
            f"Epoch {epoch + 1}/{epochs} completed | "
            f"Average loss: {avg_loss:.4f}"
        )

    torch.save(model.state_dict(), model_path)
    print(f"\nModel salvat: {model_path}")

    return model


def main():
    print("\n==============================")
    print("LOADING ACDC DATASET")
    print("==============================")
    print(f"ACDC_DATA_ROOT: {ACDC_DATA_ROOT}")
    print(f"Training folder: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Nu există folderul de training: {DATA_ROOT}\n"
            "În Colab structura trebuie să fie: /content/data/ResourcesACDC/training"
        )

    X_lv, Y_lv = load_acdc_dataset(str(DATA_ROOT), target="LV")
    X_rv, Y_rv = load_acdc_dataset(str(DATA_ROOT), target="RV")

    print("\n==============================")
    print("START TRAINING LV - ATTENTION U-NET")
    print("==============================")
    train_acdc_attention(X_lv, Y_lv, MODEL_LV_PATH)

    print("\n==============================")
    print("START TRAINING RV - ATTENTION U-NET")
    print("==============================")
    train_acdc_attention(X_rv, Y_rv, MODEL_RV_PATH)

    print("\n==============================")
    print("ACDC ATTENTION U-NET TRAINING COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()