import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.SAX_LRV.models import get_model
from src.SAX_LRV.sax_datasetloader import find_image_label_pairs, load_nifti_with_spacing
from src.utils.preprocessing import normalize_image, pad_to_size


MODEL_TYPE = os.environ.get("MODEL_TYPE", "unetpp")
TARGET_SIZE = (512, 512)
NUM_CLASSES = 4

DATA_ROOT = Path(
    os.environ.get(
        "SAX_DATA_ROOT",
        str(PROJECT_ROOT / "data" / "CombinedSAX_ED_split")
    )
)

TEST_DIR = DATA_ROOT / "testing"

MODEL_PATH = PROJECT_ROOT / "src" / "SAX_LRV" / "models neoptimizat" / f"normal_{MODEL_TYPE}_sax_lrv_multiclass_last_clean.pth"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_volume(model, volume, device):
    preds = []

    model.eval()

    for i in range(volume.shape[0]):
        img = pad_to_size(volume[i], TARGET_SIZE, value=0)
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1)

        preds.append(pred.squeeze().detach().cpu().numpy().astype(np.uint8))

    return np.array(preds)


def prepare_volume(volume):
    return np.array([pad_to_size(volume[i], TARGET_SIZE, value=0) for i in range(volume.shape[0])])


def prepare_mask(mask):
    return np.array([pad_to_size(mask[i], TARGET_SIZE, value=0) for i in range(mask.shape[0])])


def choose_slice(mask):
    areas = [np.sum(mask[i] > 0) for i in range(mask.shape[0])]
    return int(np.argmax(areas))


def show_case(image_volume, gt_volume, pred_volume, patient_name):
    slice_idx = choose_slice(gt_volume)

    image_slice = image_volume[slice_idx]
    gt_slice = gt_volume[slice_idx]
    pred_slice = pred_volume[slice_idx]

    print("\n==============================")
    print("SAX LRV VISUAL TEST")
    print("==============================")
    print(f"Model: {MODEL_TYPE}")
    print(f"Patient/file: {patient_name}")
    print(f"Slice: {slice_idx}")
    print("Labels: 0=background, 1=RV, 2=myocardium, 3=LV")
    print("==============================\n")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, cmap="jet", vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap="jet", vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(pred_slice, cmap="jet", alpha=0.4, vmin=0, vmax=3)
    axes[3].set_title("Overlay Prediction")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    device = get_device()
    print("Using device:", device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul: {MODEL_PATH}")

    pairs = find_image_label_pairs(TEST_DIR)
    image_path, label_path = random.choice(pairs)

    volume, _ = load_nifti_with_spacing(image_path)
    gt_volume, _ = load_nifti_with_spacing(label_path)

    image_volume = prepare_volume(volume)
    gt_volume = prepare_mask(gt_volume)

    model = get_model(MODEL_TYPE, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    pred_volume = predict_volume(model, volume, device)

    show_case(
        image_volume=image_volume,
        gt_volume=gt_volume,
        pred_volume=pred_volume,
        patient_name=image_path.name
    )


if __name__ == "__main__":
    main()