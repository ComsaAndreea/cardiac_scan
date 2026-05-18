import os
import torch
import numpy as np
import pandas as pd
import cv2

from src.models.unet import UNet
from src.LeftRightVentricle.acdc_datasetloader import load_nifti_with_spacing, read_info_cfg
from src.analysis.metrics import compute_volume, compute_area


# =============================
# Dice Score
# =============================
def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)


# =============================
# Diameter (max width)
# =============================
def compute_diameter(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0
    y_coords = coords[:, 0]
    return y_coords.max() - y_coords.min()


# =============================
# Interpretare medicală
# =============================
def interpret(lrvr, area_ratio, diameter_ratio):

    print("\n==============================")
    print("📊 INTERPRETARE MEDICALĂ")
    print("==============================")

    # LRVR
    print(f"\n🔹 LRVR: {lrvr:.2f}")
    if lrvr > 1.3:
        print("⚠️ LV predomină → risc insuficiență cardiacă")
    elif lrvr < 0.8:
        print("⚠️ RV predomină → posibilă patologie pulmonară")
    else:
        print("✅ LRVR normal")

    # Area ratio
    print(f"\n🔹 LV/RV Area Ratio: {area_ratio:.2f}")
    if area_ratio > 1.3:
        print("⚠️ posibilă coarctație aortei")
    else:
        print("✅ normal")

    # Diameter ratio
    print(f"\n🔹 RV/LV Diameter Ratio: {diameter_ratio:.2f}")
    if diameter_ratio > 1.0:
        print("⚠️ RV dilatat (posibil PE)")
    elif diameter_ratio > 0.9:
        print("⚠️ borderline patologic")
    else:
        print("✅ normal")

    print("\n==============================\n")


# =============================
# MAIN EVALUATION
# =============================
def evaluate_acdc():

    root = "../../data/raw/LRVentricle/training"

    model_lv = UNet()
    model_lv.load_state_dict(torch.load("../../../model_lv.pth"))
    model_lv.eval()

    model_rv = UNet()
    model_rv.load_state_dict(torch.load("../../../model_rv.pth"))
    model_rv.eval()

    results = []

    patients = sorted(os.listdir(root))

    for patient in patients:

        patient_path = os.path.join(root, patient)
        info_path = os.path.join(patient_path, "Info.cfg")

        if not os.path.exists(info_path):
            continue

        # 📌 citim ED frame corect
        ed = read_info_cfg(info_path)
        ed_str = str(ed).zfill(2)

        img_path = os.path.join(patient_path, f"{patient}_frame{ed_str}.nii.gz")
        mask_path = os.path.join(patient_path, f"{patient}_frame{ed_str}_gt.nii.gz")

        volume, spacing = load_nifti_with_spacing(img_path)
        gt_volume, _ = load_nifti_with_spacing(mask_path)

        pred_lv = []
        pred_rv = []

        gt_lv_resized = []
        gt_rv_resized = []

        # 🔁 slice loop
        for i in range(volume.shape[0]):

            img = volume[i]

            img_resized = cv2.resize(img, (256, 256))
            img_resized = (img_resized - np.min(img_resized)) / (
                np.max(img_resized) - np.min(img_resized) + 1e-8
            )

            tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()

            with torch.no_grad():
                out_lv = model_lv(tensor)
                out_rv = model_rv(tensor)

            pred_lv_slice = torch.sigmoid(out_lv).squeeze().numpy()
            pred_rv_slice = torch.sigmoid(out_rv).squeeze().numpy()

            pred_lv.append((pred_lv_slice > 0.5).astype(float))
            pred_rv.append((pred_rv_slice > 0.5).astype(float))

            # GT
            gt_slice = gt_volume[i]

            gt_lv = (gt_slice == 3).astype(float)
            gt_rv = (gt_slice == 1).astype(float)

            gt_lv = cv2.resize(gt_lv, (256, 256), interpolation=cv2.INTER_NEAREST)
            gt_rv = cv2.resize(gt_rv, (256, 256), interpolation=cv2.INTER_NEAREST)

            gt_lv_resized.append(gt_lv)
            gt_rv_resized.append(gt_rv)

        pred_lv = np.array(pred_lv)
        pred_rv = np.array(pred_rv)
        gt_lv = np.array(gt_lv_resized)
        gt_rv = np.array(gt_rv_resized)

        # =============================
        # METRICS
        # =============================

        dice_lv = dice_score(pred_lv, gt_lv)
        dice_rv = dice_score(pred_rv, gt_rv)

        vol_lv_pred = compute_volume(pred_lv, spacing)
        vol_rv_pred = compute_volume(pred_rv, spacing)

        vol_lv_gt = compute_volume(gt_lv, spacing)
        vol_rv_gt = compute_volume(gt_rv, spacing)

        lrvr_pred = vol_lv_pred / (vol_rv_pred + 1e-8)
        lrvr_gt = vol_lv_gt / (vol_rv_gt + 1e-8)

        lrvr_error = abs(lrvr_pred - lrvr_gt)

        # 🟡 AREA (slice central)
        slice_idx = volume.shape[0] // 2

        area_lv = compute_area(pred_lv[slice_idx], spacing)
        area_rv = compute_area(pred_rv[slice_idx], spacing)

        area_ratio = area_lv / (area_rv + 1e-8)

        # 🔵 DIAMETER
        diam_lv = compute_diameter(pred_lv[slice_idx])
        diam_rv = compute_diameter(pred_rv[slice_idx])

        diameter_ratio = diam_rv / (diam_lv + 1e-8)

        # =============================
        # SAVE
        # =============================
        results.append({
            "patient": patient,
            "dice_lv": dice_lv,
            "dice_rv": dice_rv,
            "lrvr_pred": lrvr_pred,
            "lrvr_gt": lrvr_gt,
            "lrvr_error": lrvr_error,
            "area_ratio": area_ratio,
            "diameter_ratio": diameter_ratio
        })

        print(f"\n{patient}")
        print(f"Dice LV: {dice_lv:.3f} | Dice RV: {dice_rv:.3f}")

        interpret(lrvr_pred, area_ratio, diameter_ratio)

    # 📊 CSV
    df = pd.DataFrame(results)
    os.makedirs("../../../experiments", exist_ok=True)
    df.to_csv("../../experiments/lrv_results_unet.csv", index=False)

    print("✔ Results saved: experiments/lrv_results_unet.csv")


if __name__ == "__main__":
    evaluate_acdc()