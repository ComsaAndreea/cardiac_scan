import os
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from src.SAX_LRV.models import get_model
from src.SAX_LRV.sax_datasetloader import load_nifti_with_spacing
from src.utils.preprocessing import normalize_image, pad_to_size


TARGET_SIZE = (512, 512)
NUM_CLASSES = 4

DATA_ROOT = Path(
    os.environ.get(
        "SAX_DATA_ROOT",
        "/kaggle/working/data/CombinedSAX_ED_split"
    )
)

TEST_DIR = DATA_ROOT / "testing"

MODEL_DIR = PROJECT_ROOT / "src"/ "SAX_LRV" / "models neoptimizat"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "SAX_LRV"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "test_results_all_models.csv"
SUMMARY_CSV = OUTPUT_DIR / "test_summary_all_models.csv"

MODELS = {
    "unet": MODEL_DIR / "normal_unet_sax_lrv_multiclass_best_clean.pth",
    "attention_unet": MODEL_DIR / "normal_attention_unet_sax_lrv_multiclass_best_clean.pth",
    "unetpp": MODEL_DIR / "normal_unetpp_sax_lrv_multiclass_best_clean.pth",
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

    print(f"Perechi test găsite: {len(pairs)}")
    return pairs


def load_model(model_type, model_path, device):
    model = get_model(model_type, num_classes=NUM_CLASSES).to(device)

    if not model_path.exists():
        raise FileNotFoundError(f"Lipsește modelul: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def predict_volume(model, volume, device):
    preds = []

    for i in range(volume.shape[0]):
        img = volume[i]

        img = pad_to_size(img, TARGET_SIZE, value=0)
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1)

        pred = pred.squeeze().cpu().numpy().astype(np.uint8)
        preds.append(pred)

    return np.array(preds)


def prepare_gt(mask):
    masks = []

    for i in range(mask.shape[0]):
        m = pad_to_size(mask[i], TARGET_SIZE, value=0)
        masks.append(m.astype(np.uint8))

    return np.array(masks)


def dice_score(gt, pred, label):
    gt_bin = gt == label
    pred_bin = pred == label

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    denominator = gt_bin.sum() + pred_bin.sum()

    if denominator == 0:
        return 1.0

    return (2 * intersection) / denominator


def precision_score(gt, pred, label):
    gt_bin = gt == label
    pred_bin = pred == label

    tp = np.logical_and(gt_bin, pred_bin).sum()
    fp = np.logical_and(~gt_bin, pred_bin).sum()

    return tp / (tp + fp + 1e-8)


def recall_score(gt, pred, label):
    gt_bin = gt == label
    pred_bin = pred == label

    tp = np.logical_and(gt_bin, pred_bin).sum()
    fn = np.logical_and(gt_bin, ~pred_bin).sum()

    return tp / (tp + fn + 1e-8)


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall + 1e-8)


def compute_volume_ml(mask, label, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    slice_thickness_mm = spacing[2]

    voxels = np.sum(mask == label)
    volume_mm3 = voxels * pixel_area_mm2 * slice_thickness_mm

    return volume_mm3 / 1000.0


def compute_max_area_cm2(mask, label, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]

    max_pixels = 0

    for i in range(mask.shape[0]):
        pixels = np.sum(mask[i] == label)
        max_pixels = max(max_pixels, pixels)

    area_mm2 = max_pixels * pixel_area_mm2
    return area_mm2 / 100.0


def evaluate_label(gt, pred, spacing, label, prefix):
    dice = dice_score(gt, pred, label)
    precision = precision_score(gt, pred, label)
    recall = recall_score(gt, pred, label)
    f1 = f1_score(precision, recall)

    gt_volume = compute_volume_ml(gt, label, spacing)
    pred_volume = compute_volume_ml(pred, label, spacing)

    gt_area = compute_max_area_cm2(gt, label, spacing)
    pred_area = compute_max_area_cm2(pred, label, spacing)

    return {
        f"{prefix}_dice": dice,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,

        f"{prefix}_gt_volume_ml": gt_volume,
        f"{prefix}_pred_volume_ml": pred_volume,
        f"{prefix}_volume_abs_error_ml": abs(pred_volume - gt_volume),
        f"{prefix}_volume_rel_error_percent": abs(pred_volume - gt_volume) / (gt_volume + 1e-8) * 100,

        f"{prefix}_gt_area_cm2": gt_area,
        f"{prefix}_pred_area_cm2": pred_area,
        f"{prefix}_area_abs_error_cm2": abs(pred_area - gt_area),
        f"{prefix}_area_rel_error_percent": abs(pred_area - gt_area) / (gt_area + 1e-8) * 100,
    }


def evaluate_model(model_type, model_path, test_pairs, device):
    print("\n==============================")
    print(f"Evaluating: {model_type}")
    print("==============================")

    model = load_model(model_type, model_path, device)

    rows = []

    for image_path, label_path in tqdm(test_pairs):
        volume, spacing = load_nifti_with_spacing(image_path)
        gt, _ = load_nifti_with_spacing(label_path)

        gt = prepare_gt(gt)
        pred = predict_volume(model, volume, device)

        patient = image_path.name.replace(".nii.gz", "").replace(".nii", "")

        row = {
            "model": model_type,
            "patient": patient,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "num_slices": volume.shape[0],
        }

        # ACDC / M&Ms labels:
        # 1 = RV, 2 = MYO, 3 = LV
        row.update(evaluate_label(gt, pred, spacing, label=1, prefix="rv"))
        row.update(evaluate_label(gt, pred, spacing, label=2, prefix="myo"))
        row.update(evaluate_label(gt, pred, spacing, label=3, prefix="lv"))

        row["lv_rv_gt_volume_ratio"] = row["lv_gt_volume_ml"] / (row["rv_gt_volume_ml"] + 1e-8)
        row["lv_rv_pred_volume_ratio"] = row["lv_pred_volume_ml"] / (row["rv_pred_volume_ml"] + 1e-8)
        row["lv_rv_volume_ratio_abs_error"] = abs(
            row["lv_rv_pred_volume_ratio"] - row["lv_rv_gt_volume_ratio"]
        )

        row["lv_rv_gt_area_ratio"] = row["lv_gt_area_cm2"] / (row["rv_gt_area_cm2"] + 1e-8)
        row["lv_rv_pred_area_ratio"] = row["lv_pred_area_cm2"] / (row["rv_pred_area_cm2"] + 1e-8)
        row["lv_rv_area_ratio_abs_error"] = abs(
            row["lv_rv_pred_area_ratio"] - row["lv_rv_gt_area_ratio"]
        )

        rows.append(row)

    return rows


def make_summary(df):
    metric_cols = [
        "rv_dice", "myo_dice", "lv_dice",
        "rv_precision", "myo_precision", "lv_precision",
        "rv_recall", "myo_recall", "lv_recall",
        "rv_f1", "myo_f1", "lv_f1",

        "rv_volume_abs_error_ml",
        "myo_volume_abs_error_ml",
        "lv_volume_abs_error_ml",

        "rv_volume_rel_error_percent",
        "myo_volume_rel_error_percent",
        "lv_volume_rel_error_percent",

        "rv_area_abs_error_cm2",
        "myo_area_abs_error_cm2",
        "lv_area_abs_error_cm2",

        "rv_area_rel_error_percent",
        "myo_area_rel_error_percent",
        "lv_area_rel_error_percent",

        "lv_rv_volume_ratio_abs_error",
        "lv_rv_area_ratio_abs_error",
    ]

    summary = df.groupby("model")[metric_cols].agg(["mean", "std", "min", "max"])
    return summary


def main():
    device = get_device()

    print("\n==============================")
    print("SAX LRV MULTICLASS TESTING")
    print("==============================")
    print(f"Test dir: {TEST_DIR}")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print("==============================\n")

    test_pairs = find_pairs(TEST_DIR)

    all_rows = []

    for model_type, model_path in MODELS.items():
        if not model_path.exists():
            print(f"Skipping {model_type}, model not found: {model_path}")
            continue

        rows = evaluate_model(model_type, model_path, test_pairs, device)
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise RuntimeError("Nu s-a evaluat niciun model. Verifică modelele .pth.")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)

    summary = make_summary(df)
    summary.to_csv(SUMMARY_CSV)

    print("\n==============================")
    print("TESTING FINISHED")
    print("==============================")
    print(f"Detailed results saved at: {RESULTS_CSV}")
    print(f"Summary saved at: {SUMMARY_CSV}")
    print("==============================\n")

    print(summary)


if __name__ == "__main__":
    main()