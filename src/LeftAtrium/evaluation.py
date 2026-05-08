import os
import torch
import numpy as np
import pandas as pd
import cv2

from src.models.unet import UNet
from src.LeftAtrium.datasetloader import load_nifti_with_spacing
from src.analysis.metrics import compute_area, compute_volume


# 🧠 Dice Score
def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)


def evaluate_left_atrium():

    images_dir = "../../data/raw/LeftAtrium/imagesTr"
    masks_dir = "../../data/raw/LeftAtrium/labelsTr"

    # 🤖 model
    model = UNet()
    model.load_state_dict(torch.load("../../model_leftatrium.pth"))
    model.eval()

    results = []

    files = sorted(os.listdir(images_dir))

    for file in files:

        if not file.endswith(".nii"):
            continue

        image_path = os.path.join(images_dir, file)
        mask_path = os.path.join(masks_dir, file)

        volume, spacing = load_nifti_with_spacing(image_path)
        gt_volume, _ = load_nifti_with_spacing(mask_path)

        pred_volume = []
        gt_resized = []

        # 🔁 procesăm fiecare slice
        for i in range(volume.shape[0]):

            # ---- IMAGE ----
            img = volume[i]

            img_resized = cv2.resize(img, (256, 256))
            img_resized = (img_resized - np.min(img_resized)) / (
                np.max(img_resized) - np.min(img_resized) + 1e-8
            )

            tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()

            # ---- PREDICTION ----
            with torch.no_grad():
                out = model(tensor)

            pred = torch.sigmoid(out).squeeze().numpy()
            pred_binary = (pred > 0.5).astype(float)

            pred_volume.append(pred_binary)

            # ---- GT (IMPORTANT: resize corect) ----
            gt_slice = gt_volume[i]
            gt_slice_resized = cv2.resize(
                gt_slice,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            )

            gt_resized.append(gt_slice_resized)

        pred_volume = np.array(pred_volume)
        gt_volume_bin = (np.array(gt_resized) > 0).astype(float)

        # 🧮 ===== METRICS =====

        # Dice
        dice = dice_score(pred_volume, gt_volume_bin)

        # Volum
        vol_pred = compute_volume(pred_volume, spacing)
        vol_gt = compute_volume(gt_volume_bin, spacing)
        vol_error = abs(vol_pred - vol_gt)

        # Arie (pe slice central)
        slice_idx = pred_volume.shape[0] // 2

        area_pred = compute_area(pred_volume[slice_idx], spacing)
        area_gt = compute_area(gt_volume_bin[slice_idx], spacing)
        area_error = abs(area_pred - area_gt)

        # salvare rezultate
        results.append({
            "patient": file,
            "dice": dice,
            "area_pred": area_pred,
            "area_gt": area_gt,
            "area_error": area_error,
            "vol_pred": vol_pred,
            "vol_gt": vol_gt,
            "vol_error": vol_error
        })

        print(f"{file}")
        print(f"  Dice: {dice:.4f}")
        print(f"  Area error: {area_error:.2f} cm²")
        print(f"  Volume error: {vol_error:.2f} mL\n")

    # 📊 salvare CSV
    df = pd.DataFrame(results)
    os.makedirs("../../experiments", exist_ok=True)
    df.to_csv("../../experiments/left_atrium_results.csv", index=False)

    print("✔ Results saved: experiments/left_atrium_results.csv")


if __name__ == "__main__":
    evaluate_left_atrium()