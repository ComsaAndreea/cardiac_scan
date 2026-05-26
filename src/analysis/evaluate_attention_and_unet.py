import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.LeftRightVentricle_ACDC.acdc_datasetloader import load_nifti_with_spacing, read_info_cfg
from src.utils.preprocessing import normalize_image, pad_to_size


TARGET_SIZE = (512, 448)

ACDC_DATA_ROOT = Path(
    os.environ.get(
        "ACDC_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "ResourcesACDC")
    )
)

ACDC_SPLIT = os.environ.get("ACDC_EVAL_SPLIT", "testing")
ACDC_DIR = ACDC_DATA_ROOT / ACDC_SPLIT

OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "acdc_model_comparison_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "acdc_model_comparison_summary.csv"


MODELS = {
    "unet": {
        "class": UNet,
        "lv_path": PROJECT_ROOT / "src" / "LeftRightVentricle" / "unet" / "model_lv.pth" ,
        "rv_path": PROJECT_ROOT / "src" / "LeftRightVentricle" / "unet" / "model_rv.pth",
    },
    "attention_unet": {
        "class": AttentionUNet,
        "lv_path": PROJECT_ROOT / "src" / "LeftRightVentricle" / "attention_unet" / "model_lv_attention.pth",
        "rv_path": PROJECT_ROOT / "src" / "LeftRightVentricle" / "attention_unet" / "model_rv_attention.pth",
    }
}


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    return device


def find_acdc_cases():
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
            f"Nu am găsit cazuri ACDC cu ground truth în: {ACDC_DIR}"
        )

    return cases


def load_model(model_class, path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_volume(model, volume, device):
    preds = []

    for i in range(volume.shape[0]):
        img = pad_to_size(volume[i], TARGET_SIZE, value=0)
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output)

        pred = prob.squeeze().detach().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8)

        preds.append(pred)

    return np.array(preds)


def prepare_gt(gt_volume, label):
    masks = []

    for i in range(gt_volume.shape[0]):
        mask = (gt_volume[i] == label).astype(np.uint8)
        mask = pad_to_size(mask, TARGET_SIZE, value=0)
        masks.append(mask)

    return np.array(masks)


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
    voxel_volume_mm3 = pixel_area_mm2 * slice_thickness_mm

    voxels = np.sum(mask > 0)
    return (voxels * voxel_volume_mm3) / 1000.0


def compute_max_area_cm2(mask, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]

    max_pixels = 0
    for i in range(mask.shape[0]):
        pixels = np.sum(mask[i] > 0)
        max_pixels = max(max_pixels, pixels)

    area_mm2 = max_pixels * pixel_area_mm2
    return area_mm2 / 100.0


def evaluate_structure(gt, pred, spacing, prefix):
    dice, precision, recall, f1, accuracy = segmentation_metrics(gt, pred)

    gt_volume = compute_volume_ml(gt, spacing)
    pred_volume = compute_volume_ml(pred, spacing)

    gt_area = compute_max_area_cm2(gt, spacing)
    pred_area = compute_max_area_cm2(pred, spacing)

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


def evaluate_model(model_name, model_info, cases, device):
    lv_path = model_info["lv_path"]
    rv_path = model_info["rv_path"]

    if not lv_path.exists() or not rv_path.exists():
        print(f"Skipping {model_name}: lipsesc modelele.")
        print(f"LV path: {lv_path}")
        print(f"RV path: {rv_path}")
        return []

    print("\n==============================")
    print(f"EVALUATING {model_name}")
    print("==============================")

    model_class = model_info["class"]
    model_lv = load_model(model_class, lv_path, device)
    model_rv = load_model(model_class, rv_path, device)

    rows = []

    for idx, (patient, frame, image_path, gt_path) in enumerate(cases):
        volume, spacing = load_nifti_with_spacing(str(image_path))
        gt_volume, _ = load_nifti_with_spacing(str(gt_path))

        gt_lv = prepare_gt(gt_volume, label=3)
        gt_rv = prepare_gt(gt_volume, label=1)

        pred_lv = predict_volume(model_lv, volume, device)
        pred_rv = predict_volume(model_rv, volume, device)

        row = {
            "model": model_name,
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

        print(f"[{idx + 1}/{len(cases)}] {model_name} - {patient} done")

    return rows


def make_summary(df):
    metric_cols = [
        "lv_dice", "rv_dice",
        "lv_precision", "rv_precision",
        "lv_recall", "rv_recall",
        "lv_f1", "rv_f1",
        "lv_accuracy", "rv_accuracy",

        "lv_volume_abs_error_ml",
        "rv_volume_abs_error_ml",
        "lv_volume_rel_error_percent",
        "rv_volume_rel_error_percent",

        "lv_area_abs_error_cm2",
        "rv_area_abs_error_cm2",
        "lv_area_rel_error_percent",
        "rv_area_rel_error_percent",

        "lv_rv_volume_ratio_abs_error",
        "lv_rv_area_ratio_abs_error",
    ]

    summary = df.groupby("model")[metric_cols].agg(["mean", "std", "min", "max"])
    return summary


def main():
    device = get_device()

    print("\n==============================")
    print("ACDC EVALUATION")
    print("==============================")
    print(f"ACDC root: {ACDC_DATA_ROOT}")
    print(f"Split: {ACDC_SPLIT}")
    print(f"Folder: {ACDC_DIR}")
    print(f"Target size: {TARGET_SIZE}")
    print("==============================\n")

    cases = find_acdc_cases()
    print(f"Număr pacienți găsiți: {len(cases)}")

    all_rows = []

    for model_name, model_info in MODELS.items():
        rows = evaluate_model(model_name, model_info, cases, device)
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise RuntimeError("Nu s-a evaluat niciun model. Verifică fișierele .pth.")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)

    summary = make_summary(df)
    summary.to_csv(SUMMARY_CSV)

    print("\n==============================")
    print("RESULTS SAVED")
    print("==============================")
    print(f"Detailed results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print("==============================\n")

    print("\nSUMMARY:")
    print(summary)


if __name__ == "__main__":
    main()