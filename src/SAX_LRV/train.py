import os
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.SAX_LRV.models import get_model
from src.SAX_LRV.sax_datasetloader import load_sax_dataset
from src.SAX_LRV.sax_torchdataset import SAXLRVDataset


MODEL_TYPE = os.environ.get("MODEL_TYPE", "unet")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
LR = float(os.environ.get("LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
PATIENCE = int(os.environ.get("PATIENCE", "8"))

TARGET_SIZE = (512, 512)
NUM_CLASSES = 4

DATA_ROOT = Path(
    os.environ.get(
        "SAX_DATA_ROOT",
        str("/kaggle/input/datasets/comsaandreea/heart-sax-mri/CombinedSAX_ED_split")
            #PROJECT_ROOT.parent / "data" / "CombinedSAX_ED_split")
    )
)

TRAIN_DIR = DATA_ROOT / "training"
VAL_DIR = DATA_ROOT / "validation"

MODEL_DIR = PROJECT_ROOT / "models" / "SAX_LRV"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "experiments" / "SAX_LRV"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / f"{MODEL_TYPE}_sax_lrv_multiclass_last.pth"
BEST_MODEL_PATH = MODEL_DIR / f"{MODEL_TYPE}_sax_lrv_multiclass_best.pth"
TRAIN_LOG_CSV = LOG_DIR / f"{MODEL_TYPE}_training_log.csv"

CHECKPOINT_PATH = MODEL_DIR / f"{MODEL_TYPE}_sax_lrv_checkpoint.pth"

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train() if train_mode else model.eval()

    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            loss = criterion(outputs, masks)

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def save_log(log_rows):
    with open(TRAIN_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "is_best"
            ]
        )
        writer.writeheader()
        writer.writerows(log_rows)


def train():
    device = get_device()

    print("\n==============================")
    print("SAX LRV MULTICLASS TRAINING")
    print("==============================")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Validation dir: {VAL_DIR}")
    print(f"Max epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Patience: {PATIENCE}")
    print(f"Last model: {MODEL_PATH}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Training log: {TRAIN_LOG_CSV}")
    print("==============================\n")

    X_train, Y_train = load_sax_dataset(TRAIN_DIR)
    X_val, Y_val = load_sax_dataset(VAL_DIR)

    train_dataset = SAXLRVDataset(
        X_train,
        Y_train,
        size=TARGET_SIZE,
        augment=True
    )

    val_dataset = SAXLRVDataset(
        X_val,
        Y_val,
        size=TARGET_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = get_model(MODEL_TYPE, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    log_rows = []

    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        log_rows = checkpoint.get("log_rows", [])

        print(f"Continuing from epoch {start_epoch}")
        print(f"Best epoch so far: {best_epoch}")
        print(f"Best val loss so far: {best_val_loss:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=True
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=False
        )

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        torch.save(model.state_dict(), MODEL_PATH)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "log_rows": log_rows,
            "model_type": MODEL_TYPE,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved at: {CHECKPOINT_PATH}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "is_best": int(is_best)
        })

        save_log(log_rows)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Best epoch: {best_epoch} | "
            f"Best val loss: {best_val_loss:.4f}"
        )

        if is_best:
            print("Best model updated.")

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n==============================")
    print("TRAINING FINISHED")
    print("==============================")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print(f"Last model saved at: {MODEL_PATH}")
    print(f"Training log saved at: {TRAIN_LOG_CSV}")
    print("==============================")


if __name__ == "__main__":
    train()