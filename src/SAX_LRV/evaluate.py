import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd

from src.SAX_LRV.models import get_model
from src.SAX_LRV.sax_datasetloader import find_image_label_pairs, load_nifti_with_spacing
from src.utils.preprocessing import normalize_image, pad_to_size


MODEL_TYPE = os.environ.get("MODEL_TYPE", "unet")
TARGET_SIZE = (512, 512)
NUM_CLASSES = 4

DATA_ROOT = Path(
    os.environ.get(
        "SAX_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "CombinedSAX_ED_split")
    )
)

TEST_DIR = DATA_ROOT / "testing"

MODEL_PATH = PROJECT_ROOT / "models" / "SAX_LRV" / f"{MODEL_TYPE}_sax_lrv_multiclass.pth"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "SAX_LRV"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / f"{MODEL_TYPE}_sax_lrv_results.csv"
SUMMARY_CSV = OUTPUT_DIR / f"{MODEL_TYPE}_sax_lrv_summary.csv"


LABELS = {
    1: "rv",
    2: "myocardium",
    3: "lv",
}


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


def prepare_mask(mask):
    return np.array([pad_to_size(mask[i], TARGET_SIZE, value=0) for i in range(mask.shape[0])])


def segmentation_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return dice, precision, recall, f1, accuracy


def compute_volume_ml(mask, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    slice_thickness_mm = spacing[2]
    voxels = np.sum(mask > 0)

    return voxels * pixel_area_mm2 * slice_thickness_mm / 1000.0


def compute_max_area_cm2(mask, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]

    max_pixels = 0
    for i in range(mask.shape[0]):
        max_pixels = max(max_pixels, np.sum(mask[i] > 0))

    return max_pixels * pixel_area_mm2 / 100.0


def evaluate_label(gt_volume, pred_volume, spacing, label_id, label_name):
    gt = (gt_volume == label_id).astype(np.uint8)
    pred = (pred_volume == label_id).astype(np.uint8)

    dice, precision, recall, f1, accuracy = segmentation_metrics(gt, pred)

    gt_volume_ml = compute_volume_ml(gt, spacing)
    pred_volume_ml = compute_volume_ml(pred, spacing)

    gt_area_cm2 = compute_max_area_cm2(gt, spacing)
    pred_area_cm2 = compute_max_area_cm2(pred, spacing)

    return {
        f"{label_name}_dice": dice,
        f"{label_name}_precision": precision,
        f"{label_name}_recall": recall,
        f"{label_name}_f1": f1,
        f"{label_name}_accuracy": accuracy,
        f"{label_name}_gt_volume_ml": gt_volume_ml,
        f"{label_name}_pred_volume_ml": pred_volume_ml,
        f"{label_name}_volume_abs_error_ml": abs(pred_volume_ml - gt_volume_ml),
        f"{label_name}_volume_rel_error_percent": abs(pred_volume_ml - gt_volume_ml) / (gt_volume_ml + 1e-8) * 100,
        f"{label_name}_gt_area_cm2": gt_area_cm2,
        f"{label_name}_pred_area_cm2": pred_area_cm2,
        f"{label_name}_area_abs_error_cm2": abs(pred_area_cm2 - gt_area_cm2),
        f"{label_name}_area_rel_error_percent": abs(pred_area_cm2 - gt_area_cm2) / (gt_area_cm2 + 1e-8) * 100,
    }


def main():
    device = get_device()
    print("Using device:", device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul: {MODEL_PATH}")

    pairs = find_image_label_pairs(TEST_DIR)

    model = get_model(MODEL_TYPE, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    rows = []

    for idx, (image_path, label_path) in enumerate(pairs):
        volume, spacing = load_nifti_with_spacing(image_path)
        gt_volume, _ = load_nifti_with_spacing(label_path)

        gt_volume = prepare_mask(gt_volume)
        pred_volume = predict_volume(model, volume, device)

        row = {
            "model": MODEL_TYPE,
            "image": image_path.name,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "num_slices": volume.shape[0],
        }

        for label_id, label_name in LABELS.items():
            row.update(evaluate_label(gt_volume, pred_volume, spacing, label_id, label_name))

        rows.append(row)
        print(f"[{idx + 1}/{len(pairs)}] {image_path.name} done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    metric_cols = [col for col in df.columns if col not in ["model", "image"]]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std", "min", "max"])
    summary.to_csv(SUMMARY_CSV)

    print("\nDone.")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(summary)


if __name__ == "__main__":
    main()