import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd

from src.models.unet import UNet
from src.LeftRightVentricle_ACDC.acdc_datasetloader import load_nifti_with_spacing, read_info_cfg
from src.utils.preprocessing import normalize_image, pad_to_size


MODEL_NAME = "unet"
TARGET_SIZE = (512, 448)

ACDC_DATA_ROOT = Path(
    os.environ.get(
        "ACDC_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "ResourcesACDC")
    )
)

ACDC_SPLIT = os.environ.get("ACDC_EVAL_SPLIT", "testing")
ACDC_DIR = ACDC_DATA_ROOT / ACDC_SPLIT

MODEL_LV_PATH = PROJECT_ROOT / "model_lv.pth"
MODEL_RV_PATH = PROJECT_ROOT / "model_rv.pth"

OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "acdc_unet_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "acdc_unet_summary.csv"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def compute_binary_area(mask_slice, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    pixels = np.sum(mask_slice > 0)
    area_mm2 = pixels * pixel_area_mm2
    return area_mm2 / 100.0


def compute_binary_volume(mask_volume, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    slice_thickness_mm = spacing[2]
    voxel_volume_mm3 = pixel_area_mm2 * slice_thickness_mm
    voxels = np.sum(mask_volume > 0)
    volume_mm3 = voxels * voxel_volume_mm3
    return volume_mm3 / 1000.0


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


def find_acdc_cases():
    if not ACDC_DIR.exists():
        raise FileNotFoundError(f"Nu există folderul ACDC: {ACDC_DIR}")

    cases = []

    for patient_dir in sorted(ACDC_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient = patient_dir.name
        info_path = patient_dir / "Info.cfg"

        if not info_path.exists():
            continue

        ed = read_info_cfg(str(info_path))
        if ed is None:
            continue

        frame = str(ed).zfill(2)

        image_path = patient_dir / f"{patient}_frame{frame}.nii.gz"
        gt_path = patient_dir / f"{patient}_frame{frame}_gt.nii.gz"

        if image_path.exists() and gt_path.exists():
            cases.append((patient, frame, image_path, gt_path))

    if len(cases) == 0:
        raise FileNotFoundError(
            f"Nu am găsit cazuri ACDC cu ground truth în: {ACDC_DIR}\n"
            "Dacă testing nu are *_gt.nii.gz, setează ACDC_EVAL_SPLIT=training."
        )

    return cases


def load_model(model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_volume(model, volume, device):
    pred_slices = []

    model.eval()

    for i in range(volume.shape[0]):
        img = volume[i]
        img = pad_to_size(img, TARGET_SIZE, value=0)
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output)

        pred = prob.squeeze().detach().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)

        pred_slices.append(pred_binary)

    return np.array(pred_slices)


def prepare_gt(gt_volume, label):
    masks = []

    for i in range(gt_volume.shape[0]):
        mask = (gt_volume[i] == label).astype(np.uint8)
        mask = pad_to_size(mask, TARGET_SIZE, value=0)
        masks.append(mask)

    return np.array(masks)


def evaluate_structure(gt, pred, spacing, prefix):
    dice, precision, recall, f1, accuracy = segmentation_metrics(gt, pred)

    gt_volume = compute_binary_volume(gt, spacing)
    pred_volume = compute_binary_volume(pred, spacing)

    gt_area = max(compute_binary_area(gt[i], spacing) for i in range(gt.shape[0]))
    pred_area = max(compute_binary_area(pred[i], spacing) for i in range(pred.shape[0]))

    return {
        f"{prefix}_dice": dice,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_accuracy": accuracy,

        f"{prefix}_gt_volume_ml": gt_volume,
        f"{prefix}_pred_volume_ml": pred_volume,
        f"{prefix}_volume_abs_error_ml": abs(pred_volume - gt_volume),
        f"{prefix}_volume_rel_error_percent": abs(pred_volume - gt_volume) / (gt_volume + 1e-8) * 100,

        f"{prefix}_gt_area_cm2": gt_area,
        f"{prefix}_pred_area_cm2": pred_area,
        f"{prefix}_area_abs_error_cm2": abs(pred_area - gt_area),
        f"{prefix}_area_rel_error_percent": abs(pred_area - gt_area) / (gt_area + 1e-8) * 100,
    }


def main():
    device = get_device()

    print("\n==============================")
    print("ACDC U-NET EVALUATION")
    print("==============================")
    print(f"ACDC root: {ACDC_DATA_ROOT}")
    print(f"Split: {ACDC_SPLIT}")
    print(f"Folder: {ACDC_DIR}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"LV model: {MODEL_LV_PATH}")
    print(f"RV model: {MODEL_RV_PATH}")
    print("==============================\n")

    if not MODEL_LV_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul LV: {MODEL_LV_PATH}")

    if not MODEL_RV_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul RV: {MODEL_RV_PATH}")

    cases = find_acdc_cases()
    print(f"Număr pacienți găsiți: {len(cases)}")

    model_lv = load_model(MODEL_LV_PATH, device)
    model_rv = load_model(MODEL_RV_PATH, device)

    rows = []

    for idx, (patient, frame, image_path, gt_path) in enumerate(cases):
        volume, spacing = load_nifti_with_spacing(str(image_path))
        gt_volume, _ = load_nifti_with_spacing(str(gt_path))

        gt_lv = prepare_gt(gt_volume, label=3)
        gt_rv = prepare_gt(gt_volume, label=1)

        pred_lv = predict_volume(model_lv, volume, device)
        pred_rv = predict_volume(model_rv, volume, device)

        row = {
            "model": MODEL_NAME,
            "patient": patient,
            "frame": frame,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "num_slices": volume.shape[0],
        }

        row.update(evaluate_structure(gt_lv, pred_lv, spacing, "lv"))
        row.update(evaluate_structure(gt_rv, pred_rv, spacing, "rv"))

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

        print(f"[{idx + 1}/{len(cases)}] {patient} done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    metric_cols = [
        "lv_dice", "rv_dice",
        "lv_precision", "rv_precision",
        "lv_recall", "rv_recall",
        "lv_f1", "rv_f1",
        "lv_accuracy", "rv_accuracy",
        "lv_volume_abs_error_ml", "rv_volume_abs_error_ml",
        "lv_volume_rel_error_percent", "rv_volume_rel_error_percent",
        "lv_area_abs_error_cm2", "rv_area_abs_error_cm2",
        "lv_area_rel_error_percent", "rv_area_rel_error_percent",
        "lv_rv_volume_ratio_abs_error",
        "lv_rv_area_ratio_abs_error",
    ]

    summary = df.groupby("model")[metric_cols].agg(["mean", "std", "min", "max"])
    summary.to_csv(SUMMARY_CSV)

    print("\n==============================")
    print("RESULTS SAVED")
    print("==============================")
    print(f"Detailed results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print("==============================\n")

    print(summary)


if __name__ == "__main__":
    main()