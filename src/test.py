import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.unet import UNet
from src.analysis.metrics import compute_area, compute_volume, interpret_results
from src.datasetloader import load_nifti_with_spacing


def test_full_pipeline():
    # 📁 alegem un pacient
    patient_id = "026"

    image_path = f"../data/raw/imagesTr/la_{patient_id}.nii"
    mask_path = f"../data/raw/labelsTr/la_{patient_id}.nii"

    # 📦 încărcăm volum 3D + spacing
    volume, spacing = load_nifti_with_spacing(image_path)
    gt_mask_volume, _ = load_nifti_with_spacing(mask_path)

    print("Shape volum:", volume.shape)
    print("Spacing:", spacing)

    # 🤖 model
    model = UNet()
    model.load_state_dict(torch.load("../model.pth"))
    model.eval()

    # 🧠 vom construi predicția 3D
    pred_volume = []

    # 🔁 trecem prin fiecare slice
    for i in range(volume.shape[0]):
        slice_img = volume[i]

        # normalizare simplă
        slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)

        # transformare în tensor
        tensor_img = torch.tensor(slice_img).unsqueeze(0).unsqueeze(0).float()

        # predicție
        with torch.no_grad():
            output = model(tensor_img)

        pred = output.squeeze().numpy()
        pred_binary = (pred > 0.5).astype(float)

        pred_volume.append(pred_binary)

    pred_volume = np.array(pred_volume)

    # 🧮 ===== CALCUL METRICS =====

    # alegem un slice pentru arie
    slice_idx = volume.shape[0] // 2
    pred_slice = pred_volume[slice_idx]

    area = compute_area(pred_slice, spacing)
    volume_ml = compute_volume(pred_volume, spacing)

    print("\n🔹 METRICS (PE PREDICȚIA AI):")
    interpret_results(area, volume_ml)

    # 🖼️ ===== AFIȘARE =====

    # imagine originală
    plt.imshow(volume[slice_idx], cmap='gray')
    plt.title("Image")
    plt.show()

    # ground truth
    plt.imshow(gt_mask_volume[slice_idx], cmap='gray')
    plt.title("Ground Truth")
    plt.show()

    # predicția AI
    plt.imshow(pred_slice, cmap='gray')
    plt.title("Prediction")
    plt.show()

    # overlay
    plt.imshow(volume[slice_idx], cmap='gray')
    plt.imshow(pred_slice, alpha=0.4)
    plt.title("Overlay AI")
    plt.show()


if __name__ == "__main__":
    test_full_pipeline()