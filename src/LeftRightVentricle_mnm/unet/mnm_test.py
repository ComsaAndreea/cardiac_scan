import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.unet import UNet
from src.LeftRightVentricle_mnm.mnm_datasetloader import load_nifti_with_spacing, find_image_label_pairs
from src.utils.preprocessing import normalize_image, pad_to_size


TARGET_SIZE = (512, 448)

MNM_DATA_ROOT = Path(
    os.environ.get(
        "MNM_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "ResourcesMNM")
    )
)

TEST_DIR = MNM_DATA_ROOT / "testing"

MODEL_LV_PATH = PROJECT_ROOT / "model_lv_mnm_unet.pth"
MODEL_RV_PATH = PROJECT_ROOT / "model_rv_mnm_unet.pth"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def compute_binary_area(mask_slice, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    pixels = np.sum(mask_slice > 0)
    return pixels * pixel_area_mm2 / 100.0


def compute_binary_volume(mask_volume, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    slice_thickness_mm = spacing[2]
    voxels = np.sum(mask_volume > 0)
    return voxels * pixel_area_mm2 * slice_thickness_mm / 1000.0


def predict_binary_volume(model, volume, device):
    preds = []

    model.eval()

    for i in range(volume.shape[0]):
        img = pad_to_size(volume[i], TARGET_SIZE, value=0)
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output)

        pred = prob.squeeze().detach().cpu().numpy()
        pred = (pred > 0.5).astype(np.float32)
        preds.append(pred)

    return np.array(preds)


def prepare_gt(gt_volume):
    gt_lv = []
    gt_rv = []

    for i in range(gt_volume.shape[0]):
        gt_slice = gt_volume[i]

        lv = (gt_slice == 3).astype(np.float32)
        rv = (gt_slice == 1).astype(np.float32)

        lv = pad_to_size(lv, TARGET_SIZE, value=0)
        rv = pad_to_size(rv, TARGET_SIZE, value=0)

        gt_lv.append(lv)
        gt_rv.append(rv)

    return np.array(gt_lv), np.array(gt_rv)


def choose_slice(gt_lv, gt_rv):
    areas = [
        np.sum(gt_lv[i] > 0) + np.sum(gt_rv[i] > 0)
        for i in range(gt_lv.shape[0])
    ]
    return int(np.argmax(areas))


def show_case(volume, gt_lv, gt_rv, pred_lv, pred_rv, spacing, patient_name):
    slice_idx = choose_slice(gt_lv, gt_rv)

    image_slice = pad_to_size(volume[slice_idx], TARGET_SIZE, value=0)

    gt_lv_slice = gt_lv[slice_idx]
    gt_rv_slice = gt_rv[slice_idx]
    pred_lv_slice = pred_lv[slice_idx]
    pred_rv_slice = pred_rv[slice_idx]

    gt_combined = np.logical_or(gt_lv_slice > 0, gt_rv_slice > 0).astype(np.float32)
    pred_combined = np.logical_or(pred_lv_slice > 0, pred_rv_slice > 0).astype(np.float32)

    area_gt = compute_binary_area(gt_combined, spacing)
    area_pred = compute_binary_area(pred_combined, spacing)

    vol_gt = compute_binary_volume(np.logical_or(gt_lv > 0, gt_rv > 0), spacing)
    vol_pred = compute_binary_volume(np.logical_or(pred_lv > 0, pred_rv > 0), spacing)

    print("\n==============================")
    print("MNM U-NET TEST")
    print("==============================")
    print(f"Patient/file: {patient_name}")
    print(f"Slice afișat: {slice_idx}")
    print(f"Spacing: {spacing}")
    print(f"GT area: {area_gt:.2f} cm²")
    print(f"Pred area: {area_pred:.2f} cm²")
    print(f"GT volume: {vol_gt:.2f} mL")
    print(f"Pred volume: {vol_pred:.2f} mL")
    print("==============================\n")

    gt_overlay = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    gt_overlay[:, :, 1] = gt_lv_slice
    gt_overlay[:, :, 0] = gt_rv_slice

    pred_overlay = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    pred_overlay[:, :, 1] = pred_lv_slice
    pred_overlay[:, :, 0] = pred_rv_slice

    final_overlay = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    final_overlay[:, :, 1] = gt_combined
    final_overlay[:, :, 0] = pred_combined

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground truth\nLV green, RV red")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction\nLV green, RV red")
    axes[2].axis("off")

    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(final_overlay, alpha=0.4)
    axes[3].set_title("Overlay\nGT green, Pred red")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    device = get_device()

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Nu există folderul de test: {TEST_DIR}")

    if not MODEL_LV_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul LV: {MODEL_LV_PATH}")

    if not MODEL_RV_PATH.exists():
        raise FileNotFoundError(f"Nu există modelul RV: {MODEL_RV_PATH}")

    pairs = find_image_label_pairs(TEST_DIR)
    image_path, label_path = random.choice(pairs)

    volume, spacing = load_nifti_with_spacing(image_path)
    gt_volume, _ = load_nifti_with_spacing(label_path)

    gt_lv, gt_rv = prepare_gt(gt_volume)

    model_lv = UNet().to(device)
    model_lv.load_state_dict(torch.load(MODEL_LV_PATH, map_location=device))
    model_lv.eval()

    model_rv = UNet().to(device)
    model_rv.load_state_dict(torch.load(MODEL_RV_PATH, map_location=device))
    model_rv.eval()

    pred_lv = predict_binary_volume(model_lv, volume, device)
    pred_rv = predict_binary_volume(model_rv, volume, device)

    show_case(
        volume=volume,
        gt_lv=gt_lv,
        gt_rv=gt_rv,
        pred_lv=pred_lv,
        pred_rv=pred_rv,
        spacing=spacing,
        patient_name=image_path.name
    )


if __name__ == "__main__":
    main()