import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.SAX_LRV.models import get_model
from src.SAX_LRV.sax_datasetloader import load_nifti_with_spacing
from src.utils.preprocessing import normalize_image, pad_to_size


TARGET_SIZE = (512, 512)
NUM_CLASSES = 4
N_EXAMPLES = int(os.environ.get("N_EXAMPLES", "6"))
MODEL_TYPE = os.environ.get("MODEL_TYPE", "unetpp")

DATA_ROOT = Path(
    os.environ.get(
        "SAX_DATA_ROOT",
        #"/kaggle/working/data/CombinedSAX_ED_split"
        PROJECT_ROOT/"data"/"CombinedSAX_ED_split"
    )
)

TEST_DIR = DATA_ROOT / "testing"

MODEL_DIR = PROJECT_ROOT /"src"/"SAX_LRV"/ "models neoptimizat"
MODEL_PATH = MODEL_DIR / f"normal_{MODEL_TYPE}_sax_lrv_multiclass_best_clean.pth"

OUTPUT_DIR = PROJECT_ROOT /"src"/"SAX_LRV"/"experiments"/ "prediction_examples" / MODEL_TYPE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LABEL_NAMES = {
    0: "Background",
    1: "RV",
    2: "MYO",
    3: "LV",
}


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def find_pairs(split_dir):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    image_files = list(images_dir.glob("*.nii")) + list(images_dir.glob("*.nii.gz"))

    pairs = []

    for image_path in sorted(image_files):
        name = image_path.name

        if name.endswith(".nii.gz"):
            label_name = name.replace(".nii.gz", "_gt.nii.gz")
        else:
            label_name = name.replace(".nii", "_gt.nii")

        label_path = labels_dir / label_name

        if label_path.exists():
            pairs.append((image_path, label_path))

    if len(pairs) == 0:
        raise FileNotFoundError(f"Nu am găsit perechi image-label în {split_dir}")

    return pairs


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul: {MODEL_PATH}")

    model = get_model(MODEL_TYPE, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model


def predict_slice(model, image_slice, device):
    img = pad_to_size(image_slice, TARGET_SIZE, value=0)
    img = normalize_image(img)

    tensor = torch.tensor(img, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1)

    pred = pred.squeeze().cpu().numpy().astype(np.uint8)

    return img, pred


def prepare_mask_slice(mask_slice):
    mask = pad_to_size(mask_slice, TARGET_SIZE, value=0)
    return mask.astype(np.uint8)


def choose_informative_slice(mask_volume):
    best_idx = 0
    best_pixels = 0

    for i in range(mask_volume.shape[0]):
        pixels = np.sum(mask_volume[i] > 0)

        if pixels > best_pixels:
            best_pixels = pixels
            best_idx = i

    return best_idx


def make_mask_rgb(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    # 1 = RV = blue
    rgb[mask == 1] = [0.0, 0.25, 1.0]

    # 2 = MYO = yellow
    rgb[mask == 2] = [1.0, 0.85, 0.0]

    # 3 = LV = red
    rgb[mask == 3] = [1.0, 0.0, 0.0]

    return rgb


def save_example(image, gt, pred, patient_name, slice_idx):
    gt_rgb = make_mask_rgb(gt)
    pred_rgb = make_mask_rgb(pred)

    overlay = np.stack([image, image, image], axis=-1)
    overlay = overlay.astype(np.float32)

    pred_area = pred > 0
    overlay[pred_area] = 0.55 * overlay[pred_area] + 0.45 * pred_rgb[pred_area]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{patient_name} | slice {slice_idx} | "
        "RV=blue, MYO=yellow, LV=red",
        fontsize=11
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{patient_name}_slice{slice_idx}_{MODEL_TYPE}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    random.seed(42)

    device = get_device()
    model = load_model(device)

    pairs = find_pairs(TEST_DIR)

    selected_pairs = random.sample(
        pairs,
        min(N_EXAMPLES, len(pairs))
    )

    print("\n==============================")
    print("GENERATING PREDICTION EXAMPLES")
    print("==============================")
    print(f"Model: {MODEL_TYPE}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Examples: {len(selected_pairs)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("==============================\n")

    for image_path, label_path in selected_pairs:
        volume, _ = load_nifti_with_spacing(image_path)
        mask_volume, _ = load_nifti_with_spacing(label_path)

        slice_idx = choose_informative_slice(mask_volume)

        image_slice = volume[slice_idx]
        mask_slice = mask_volume[slice_idx]

        image_prepared, pred = predict_slice(
            model,
            image_slice,
            device
        )

        gt = prepare_mask_slice(mask_slice)

        patient_name = image_path.name
        patient_name = patient_name.replace(".nii.gz", "")
        patient_name = patient_name.replace(".nii", "")

        save_example(
            image=image_prepared,
            gt=gt,
            pred=pred,
            patient_name=patient_name,
            slice_idx=slice_idx
        )

    print("\nDone.")


if __name__ == "__main__":
    main()