import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.attention_unet import AttentionUNet
from src.LeftRightVentricle.acdc_datasetloader import load_nifti_with_spacing, read_info_cfg
from src.utils.preprocessing import normalize_image, pad_to_size


TARGET_SIZE = (512, 448)

ACDC_DATA_ROOT = Path(
    os.environ.get(
        "ACDC_DATA_ROOT",
        str(PROJECT_ROOT.parent / "data" / "raw" / "LRVentricle")
    )
)

ACDC_TEST_SPLIT = os.environ.get("ACDC_TEST_SPLIT", "testing")
ACDC_TEST_DIR = ACDC_DATA_ROOT / ACDC_TEST_SPLIT

MODEL_LV_PATH = PROJECT_ROOT / "LeftRightVentricle" / "attention_unet" / "model_lv_attention.pth"
MODEL_RV_PATH = PROJECT_ROOT / "LeftRightVentricle" / "attention_unet" / "model_rv_attention.pth"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


def compute_binary_area(mask_slice, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    pixels = np.sum(mask_slice > 0)
    return (pixels * pixel_area_mm2) / 100.0


def compute_binary_volume(mask_volume, spacing):
    pixel_area_mm2 = spacing[0] * spacing[1]
    slice_thickness_mm = spacing[2]
    voxel_volume_mm3 = pixel_area_mm2 * slice_thickness_mm
    voxels = np.sum(mask_volume > 0)
    return (voxels * voxel_volume_mm3) / 1000.0


def find_acdc_test_cases():
    if not ACDC_TEST_DIR.exists():
        raise FileNotFoundError(f"Nu există folderul ACDC test: {ACDC_TEST_DIR}")

    cases = []

    for patient_dir in sorted(ACDC_TEST_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient = patient_dir.name
        info_path = patient_dir / "Info.cfg"

        if not info_path.exists():
            continue

        ed = read_info_cfg(str(info_path))
        if ed is None:
            continue

        ed_str = str(ed).zfill(2)

        image_path = patient_dir / f"{patient}_frame{ed_str}.nii.gz"
        mask_path = patient_dir / f"{patient}_frame{ed_str}_gt.nii.gz"

        if image_path.exists() and mask_path.exists():
            cases.append((patient, image_path, mask_path, ed_str))

    if len(cases) == 0:
        raise FileNotFoundError(
            "Nu am găsit niciun pacient ACDC cu imagine + ground truth.\n"
            f"Am căutat în: {ACDC_TEST_DIR}"
        )

    return cases


def predict_binary_volume(model, volume, device):
    pred_slices = []
    model.eval()

    for i in range(volume.shape[0]):
        img = pad_to_size(volume[i], TARGET_SIZE, value=0)
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output)

        pred = prob.squeeze().detach().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)

        pred_slices.append(pred_binary)

    return np.array(pred_slices)


def prepare_ground_truth(gt_volume):
    gt_lv = []
    gt_rv = []

    for i in range(gt_volume.shape[0]):
        gt_slice = gt_volume[i]

        lv_slice = (gt_slice == 3).astype(np.float32)
        rv_slice = (gt_slice == 1).astype(np.float32)

        lv_slice = pad_to_size(lv_slice, TARGET_SIZE, value=0)
        rv_slice = pad_to_size(rv_slice, TARGET_SIZE, value=0)

        gt_lv.append(lv_slice)
        gt_rv.append(rv_slice)

    return np.array(gt_lv), np.array(gt_rv)


def prepare_image_volume(volume):
    padded_volume = []

    for i in range(volume.shape[0]):
        img = pad_to_size(volume[i], TARGET_SIZE, value=0)
        padded_volume.append(img)

    return np.array(padded_volume)


def choose_representative_slice(gt_lv, gt_rv):
    areas = []

    for i in range(gt_lv.shape[0]):
        area = np.sum(gt_lv[i] > 0) + np.sum(gt_rv[i] > 0)
        areas.append(area)

    return int(np.argmax(areas))


def show_acdc_case(volume, gt_lv, gt_rv, pred_lv, pred_rv, spacing, patient_name, frame):
    slice_idx = choose_representative_slice(gt_lv, gt_rv)

    image_slice = volume[slice_idx]

    gt_lv_slice = gt_lv[slice_idx]
    gt_rv_slice = gt_rv[slice_idx]

    pred_lv_slice = pred_lv[slice_idx]
    pred_rv_slice = pred_rv[slice_idx]

    gt_combined_slice = np.logical_or(gt_lv_slice > 0, gt_rv_slice > 0).astype(np.float32)
    pred_combined_slice = np.logical_or(pred_lv_slice > 0, pred_rv_slice > 0).astype(np.float32)

    gt_combined_volume = np.logical_or(gt_lv > 0, gt_rv > 0).astype(np.float32)
    pred_combined_volume = np.logical_or(pred_lv > 0, pred_rv > 0).astype(np.float32)

    area_gt = compute_binary_area(gt_combined_slice, spacing)
    area_pred = compute_binary_area(pred_combined_slice, spacing)

    volume_gt = compute_binary_volume(gt_combined_volume, spacing)
    volume_pred = compute_binary_volume(pred_combined_volume, spacing)

    lv_area_gt = compute_binary_area(gt_lv_slice, spacing)
    lv_area_pred = compute_binary_area(pred_lv_slice, spacing)

    rv_area_gt = compute_binary_area(gt_rv_slice, spacing)
    rv_area_pred = compute_binary_area(pred_rv_slice, spacing)

    lv_volume_gt = compute_binary_volume(gt_lv, spacing)
    lv_volume_pred = compute_binary_volume(pred_lv, spacing)

    rv_volume_gt = compute_binary_volume(gt_rv, spacing)
    rv_volume_pred = compute_binary_volume(pred_rv, spacing)

    print("\n==============================")
    print("ACDC ATTENTION U-NET TEST")
    print("==============================")
    print(f"Pacient random: {patient_name}")
    print(f"Split folosit: {ACDC_TEST_SPLIT}")
    print(f"Frame ED: {frame}")
    print(f"Volume shape după padding: {volume.shape}")
    print(f"Spacing: {spacing}")
    print(f"Slice afișat: {slice_idx}")
    print("------------------------------")
    print("Combined LV + RV")
    print(f"Arie Ground Truth:  {area_gt:.2f} cm²")
    print(f"Arie Prediction:    {area_pred:.2f} cm²")
    print(f"Volum Ground Truth: {volume_gt:.2f} mL")
    print(f"Volum Prediction:   {volume_pred:.2f} mL")
    print("------------------------------")
    print("LV")
    print(f"Arie LV Ground Truth:  {lv_area_gt:.2f} cm²")
    print(f"Arie LV Prediction:    {lv_area_pred:.2f} cm²")
    print(f"Volum LV Ground Truth: {lv_volume_gt:.2f} mL")
    print(f"Volum LV Prediction:   {lv_volume_pred:.2f} mL")
    print("------------------------------")
    print("RV")
    print(f"Arie RV Ground Truth:  {rv_area_gt:.2f} cm²")
    print(f"Arie RV Prediction:    {rv_area_pred:.2f} cm²")
    print(f"Volum RV Ground Truth: {rv_volume_gt:.2f} mL")
    print(f"Volum RV Prediction:   {rv_volume_pred:.2f} mL")
    print("==============================\n")

    gt_overlay = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    gt_overlay[:, :, 1] = gt_lv_slice
    gt_overlay[:, :, 0] = gt_rv_slice

    pred_overlay = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    pred_overlay[:, :, 1] = pred_lv_slice
    pred_overlay[:, :, 0] = pred_rv_slice

    final_overlay = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    final_overlay[:, :, 1] = gt_combined_slice
    final_overlay[:, :, 0] = pred_combined_slice

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground truth\nLV green, RV red")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Attention U-Net prediction\nLV green, RV red")
    axes[2].axis("off")

    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(final_overlay, alpha=0.4)
    axes[3].set_title("Overlay\nGT green, Pred red")
    axes[3].axis("off")

    plt.suptitle(
        f"Attention U-Net | {patient_name} | ED frame {frame}\n"
        f"GT area={area_gt:.2f} cm² | Pred area={area_pred:.2f} cm² | "
        f"GT volume={volume_gt:.2f} mL | Pred volume={volume_pred:.2f} mL"
    )

    plt.tight_layout()
    plt.show()


def test_acdc_attention():
    device = get_device()

    if not MODEL_LV_PATH.exists():
        raise FileNotFoundError(
            f"Nu există modelul LV Attention: {MODEL_LV_PATH}\n"
            "Antrenează mai întâi modelul LV Attention."
        )

    if not MODEL_RV_PATH.exists():
        raise FileNotFoundError(
            f"Nu există modelul RV Attention: {MODEL_RV_PATH}\n"
            "Antrenează mai întâi modelul RV Attention."
        )

    cases = find_acdc_test_cases()
    patient, image_path, mask_path, frame = random.choice(cases)

    volume, spacing = load_nifti_with_spacing(str(image_path))
    gt_volume, _ = load_nifti_with_spacing(str(mask_path))

    padded_volume = prepare_image_volume(volume)
    gt_lv, gt_rv = prepare_ground_truth(gt_volume)

    model_lv = AttentionUNet().to(device)
    model_lv.load_state_dict(torch.load(MODEL_LV_PATH, map_location=device))
    model_lv.eval()

    model_rv = AttentionUNet().to(device)
    model_rv.load_state_dict(torch.load(MODEL_RV_PATH, map_location=device))
    model_rv.eval()

    pred_lv = predict_binary_volume(model_lv, volume, device)
    pred_rv = predict_binary_volume(model_rv, volume, device)

    show_acdc_case(
        volume=padded_volume,
        gt_lv=gt_lv,
        gt_rv=gt_rv,
        pred_lv=pred_lv,
        pred_rv=pred_rv,
        spacing=spacing,
        patient_name=patient,
        frame=frame
    )


if __name__ == "__main__":
    test_acdc_attention()