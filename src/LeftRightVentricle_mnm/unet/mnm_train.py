import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.LeftRightVentricle_mnm.mnm_datasetloader import load_mnm_dataset
from src.LeftRightVentricle_mnm.mnm_torchdataset import MNMDataset


EPOCHS = 10
BATCH_SIZE = 4
LR = 0.001
TARGET_SIZE = (512, 448)

MNM_DATA_ROOT = Path(
    os.environ.get(
        "MNM_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "ResourcesMNM")
    )
)

TRAIN_DIR = MNM_DATA_ROOT / "training"

MODEL_LV_PATH = PROJECT_ROOT / "model_lv_mnm_unet.pth"
MODEL_RV_PATH = PROJECT_ROOT / "model_rv_mnm_unet.pth"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def train_model(X, Y, model_path, target_name):
    device = get_device()

    dataset = MNMDataset(X, Y, size=TARGET_SIZE)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = UNet().to(device)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model existent încărcat: {model_path}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n==============================")
    print(f"MNM U-NET TRAINING - {target_name}")
    print("==============================")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Model path: {model_path}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("==============================\n")

    for epoch in range(EPOCHS):
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
                    f"{target_name} | Epoch {epoch + 1}/{EPOCHS} | "
                    f"Step {i}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"{target_name} | Epoch {epoch + 1}/{EPOCHS} completed | Avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model salvat: {model_path}")


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Nu există folderul de training: {TRAIN_DIR}\n"
            "Structură așteptată: /content/data/ResourcesMNM/training/images și labels"
        )

    print(f"MNM_DATA_ROOT: {MNM_DATA_ROOT}")

    X_lv, Y_lv = load_mnm_dataset(TRAIN_DIR, target="LV")
    X_rv, Y_rv = load_mnm_dataset(TRAIN_DIR, target="RV")

    train_model(X_lv, Y_lv, MODEL_LV_PATH, "LV")
    train_model(X_rv, Y_rv, MODEL_RV_PATH, "RV")


if __name__ == "__main__":
    main()