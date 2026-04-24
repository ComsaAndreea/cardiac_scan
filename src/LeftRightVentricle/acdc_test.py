import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.models.unet import UNet
from src.analysis.metrics import compute_volume
from src.LeftRightVentricle.acdc_datasetloader import load_nifti_with_spacing, read_info_cfg


def test_acdc():

    patient = "patient001"
    base = f"../../data/raw/LRVentricle/training/{patient}"

    # 📄 citim ED corect
    ed = read_info_cfg(f"{base}/Info.cfg")
    ed_str = str(ed).zfill(2)

    image_path = f"{base}/{patient}_frame{ed_str}.nii.gz"
    mask_path = f"{base}/{patient}_frame{ed_str}_gt.nii.gz"

    # 📦 încărcăm volum + spacing
    volume, spacing = load_nifti_with_spacing(image_path)
    gt_mask_volume, _ = load_nifti_with_spacing(mask_path)

    print("Shape:", volume.shape)
    print("Spacing:", spacing)

    # 🤖 modele
    model_lv = UNet()
    model_lv.load_state_dict(torch.load("../../model_lv.pth"))
    model_lv.eval()

    model_rv = UNet()
    model_rv.load_state_dict(torch.load("../../model_rv.pth"))
    model_rv.eval()

    pred_lv = []
    pred_rv = []

    # 🔁 predicție pe fiecare slice
    for i in range(volume.shape[0]):

        img = volume[i]

        # 🔧 resize ca la training
        img_resized = cv2.resize(img, (256, 256))

        # normalizare
        img_resized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized) + 1e-8)

        tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            out_lv = model_lv(tensor)
            out_rv = model_rv(tensor)

        # 🔥 sigmoid IMPORTANT
        pred_lv_slice = torch.sigmoid(out_lv).squeeze().numpy()
        pred_rv_slice = torch.sigmoid(out_rv).squeeze().numpy()

        pred_lv.append((pred_lv_slice > 0.5).astype(float))
        pred_rv.append((pred_rv_slice > 0.5).astype(float))

    pred_lv = np.array(pred_lv)
    pred_rv = np.array(pred_rv)

    # 🧮 volume
    vol_lv = compute_volume(pred_lv, spacing)
    vol_rv = compute_volume(pred_rv, spacing)

    print("\n==============================")
    print("📊 REZULTATE")
    print("==============================")
    print(f"Volume LV: {vol_lv:.2f} mL")
    print(f"Volume RV: {vol_rv:.2f} mL")

    lrvr = vol_lv / (vol_rv + 1e-8)
    print(f"\nLRVR: {lrvr:.2f}")

    if lrvr > 1.3:
        print("⚠️ LV prea mare")
    elif lrvr < 0.8:
        print("⚠️ RV prea mare")
    else:
        print("✅ Normal")

    # 🖼️ alegem slice central
    idx = volume.shape[0] // 2

    img = volume[idx]

    # 🔧 resize și GT ca să fie pe aceeași dimensiune
    img_resized = cv2.resize(img, (256, 256))
    gt_slice = gt_mask_volume[idx]
    gt_slice = cv2.resize(gt_slice, (256, 256), interpolation=cv2.INTER_NEAREST)

    # 🎯 extragem GT separat
    gt_lv = (gt_slice == 3).astype(float)
    gt_rv = (gt_slice == 1).astype(float)

    # 🎯 prediction
    pred_lv_slice = pred_lv[idx]
    pred_rv_slice = pred_rv[idx]

    # =========================
    # 1️⃣ IMAGE
    # =========================
    plt.imshow(img_resized, cmap='gray')
    plt.title("Image")
    plt.axis('off')
    plt.show()

    # =========================
    # 2️⃣ GROUND TRUTH
    # =========================
    gt_overlay = np.zeros((256, 256, 3))

    gt_overlay[:, :, 1] = gt_lv  # verde
    gt_overlay[:, :, 0] = gt_rv  # roșu

    plt.imshow(gt_overlay)
    plt.title("Ground Truth (LV verde, RV roșu)")
    plt.axis('off')
    plt.show()

    # =========================
    # 3️⃣ PREDICTION
    # =========================
    pred_overlay = np.zeros((256, 256, 3))

    pred_overlay[:, :, 1] = pred_lv_slice  # verde
    pred_overlay[:, :, 0] = pred_rv_slice  # roșu

    plt.imshow(pred_overlay)
    plt.title("Prediction (LV verde, RV roșu)")
    plt.axis('off')
    plt.show()

    # =========================
    # 4️⃣ OVERLAY FINAL
    # =========================
    plt.imshow(img_resized, cmap='gray')
    plt.imshow(pred_overlay, alpha=0.4)
    plt.title("Overlay AI")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    test_acdc()