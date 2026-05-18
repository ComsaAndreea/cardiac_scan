#######################
#cu labelsTs
######################

import os
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.unet import UNet
from src.LeftAtrium.datasetloader import load_nifti_with_spacing
from src.analysis.metrics import compute_area, compute_volume


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGES_TS_DIR = PROJECT_ROOT / "data" / "raw" / "LeftAtrium" / "imagesTr"
LABELS_TS_DIR = PROJECT_ROOT / "data" / "raw" / "LeftAtrium" / "labelsTr"

MODEL_PATH = PROJECT_ROOT / "model_leftatrium.pth"


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)


def find_test_cases():
    if not IMAGES_TS_DIR.exists():
        raise FileNotFoundError(f"Nu există folderul de imagini test: {IMAGES_TS_DIR}")

    if not LABELS_TS_DIR.exists():
        raise FileNotFoundError(
            f"Nu există folderul de măști test: {LABELS_TS_DIR}\n"
            "Pentru ce vrei tu acum — ground truth, arie GT, volum GT — este obligatoriu să existe labelsTs.\n"
            "În PDF-ul atașat apare imagesTs, dar nu apare labelsTs."
        )

    cases = []

    for image_file in sorted(IMAGES_TS_DIR.iterdir()):
        if not image_file.name.endswith(".nii") and not image_file.name.endswith(".nii.gz"):
            continue

        if image_file.name.startswith("._"):
            continue

        mask_file = LABELS_TS_DIR / image_file.name

        if mask_file.exists():
            cases.append((image_file, mask_file))

    if len(cases) == 0:
        raise FileNotFoundError(
            "Nu am găsit niciun pacient de test cu imagine + ground truth.\n"
            f"Am căutat imagini în: {IMAGES_TS_DIR}\n"
            f"Am căutat măști cu același nume în: {LABELS_TS_DIR}"
        )

    return cases


def predict_volume(model, volume):
    pred_slices = []

    model.eval()

    for i in range(volume.shape[0]):
        img = volume[i]
        img = normalize_image(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output)

        pred = prob.squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)

        pred_slices.append(pred_binary)

    return np.array(pred_slices)


def choose_representative_slice(gt_volume):
    """
    Alegem slice-ul unde ground truth-ul are aria cea mai mare.
    Este mai relevant decât slice-ul central, pentru că structura poate să nu fie maximă la mijloc.
    """
    areas = [np.sum(gt_volume[i] > 0) for i in range(gt_volume.shape[0])]
    return int(np.argmax(areas))


def show_left_atrium_case(volume, gt_volume, pred_volume, spacing, patient_name):
    slice_idx = choose_representative_slice(gt_volume)

    image_slice = volume[slice_idx]
    gt_slice = (gt_volume[slice_idx] > 0).astype(np.float32)
    pred_slice = pred_volume[slice_idx]

    area_gt = compute_area(gt_slice, spacing)
    area_pred = compute_area(pred_slice, spacing)

    volume_gt = compute_volume((gt_volume > 0).astype(np.float32), spacing)
    volume_pred = compute_volume(pred_volume, spacing)

    print("\n==============================")
    print("LEFT ATRIUM TEST")
    print("==============================")
    print(f"Pacient test random: {patient_name}")
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")
    print(f"Slice afișat: {slice_idx}")
    print("------------------------------")
    print(f"Arie Ground Truth: {area_gt:.2f} cm²")
    print(f"Arie Prediction:   {area_pred:.2f} cm²")
    print(f"Volum Ground Truth: {volume_gt:.2f} mL")
    print(f"Volum Prediction:   {volume_pred:.2f} mL")
    print("==============================\n")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(gt_slice, alpha=0.35, cmap="Greens")
    axes[3].imshow(pred_slice, alpha=0.35, cmap="Reds")
    axes[3].set_title("Overlay: GT green, Pred red")
    axes[3].axis("off")

    plt.suptitle(
        f"{patient_name}\n"
        f"GT area={area_gt:.2f} cm² | Pred area={area_pred:.2f} cm² | "
        f"GT volume={volume_gt:.2f} mL | Pred volume={volume_pred:.2f} mL"
    )

    plt.tight_layout()
    plt.show()


def test_left_atrium():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu există modelul: {MODEL_PATH}\n"
            "Antrenează mai întâi modelul Left Atrium."
        )

    cases = find_test_cases()
    image_path, mask_path = random.choice(cases)

    volume, spacing = load_nifti_with_spacing(str(image_path))
    gt_volume, _ = load_nifti_with_spacing(str(mask_path))

    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    pred_volume = predict_volume(model, volume)

    show_left_atrium_case(
        volume=volume,
        gt_volume=gt_volume,
        pred_volume=pred_volume,
        spacing=spacing,
        patient_name=image_path.name
    )


if __name__ == "__main__":
    test_left_atrium()

#############################
#fara labelsTs
#############################
#
# import random
# from pathlib import Path
#
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# from src.models.unet import UNet
# from src.LeftAtrium.datasetloader import load_nifti_with_spacing
# from src.analysis.metrics import compute_area, compute_volume
#
#
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
#
# IMAGES_TS_DIR = PROJECT_ROOT / "data" / "raw" / "LeftAtrium" / "imagesTs"
# MODEL_PATH = PROJECT_ROOT / "model_leftatrium.pth"
#
#
# def normalize_image(image):
#     return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
#
#
# def find_test_images():
#     if not IMAGES_TS_DIR.exists():
#         raise FileNotFoundError(f"Nu există folderul de imagini test: {IMAGES_TS_DIR}")
#
#     images = []
#
#     for image_file in sorted(IMAGES_TS_DIR.iterdir()):
#         if image_file.name.startswith("._"):
#             continue
#
#         if image_file.name.endswith(".nii") or image_file.name.endswith(".nii.gz"):
#             images.append(image_file)
#
#     if len(images) == 0:
#         raise FileNotFoundError(f"Nu am găsit imagini test în: {IMAGES_TS_DIR}")
#
#     return images
#
#
# def predict_volume(model, volume):
#     pred_slices = []
#
#     model.eval()
#
#     for i in range(volume.shape[0]):
#         img = volume[i]
#         img = normalize_image(img)
#
#         tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#
#         with torch.no_grad():
#             output = model(tensor)
#             prob = torch.sigmoid(output)
#
#         pred = prob.squeeze().cpu().numpy()
#         pred_binary = (pred > 0.5).astype(np.float32)
#
#         pred_slices.append(pred_binary)
#
#     return np.array(pred_slices)
#
#
# def choose_representative_slice(pred_volume):
#     areas = [np.sum(pred_volume[i] > 0) for i in range(pred_volume.shape[0])]
#     return int(np.argmax(areas))
#
#
# def show_left_atrium_prediction(volume, pred_volume, spacing, patient_name):
#     slice_idx = choose_representative_slice(pred_volume)
#
#     image_slice = volume[slice_idx]
#     pred_slice = pred_volume[slice_idx]
#
#     area_pred = compute_area(pred_slice, spacing)
#     volume_pred = compute_volume(pred_volume, spacing)
#
#     print("\n==============================")
#     print("LEFT ATRIUM TEST - PREDICTION ONLY")
#     print("==============================")
#     print(f"Pacient test random: {patient_name}")
#     print(f"Volume shape: {volume.shape}")
#     print(f"Spacing: {spacing}")
#     print(f"Slice afișat: {slice_idx}")
#     print("------------------------------")
#     print(f"Arie Prediction:   {area_pred:.2f} cm²")
#     print(f"Volum Prediction:  {volume_pred:.2f} mL")
#     print("==============================\n")
#
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#     axes[0].imshow(image_slice, cmap="gray")
#     axes[0].set_title("Original image")
#     axes[0].axis("off")
#
#     axes[1].imshow(pred_slice, cmap="gray")
#     axes[1].set_title("Prediction")
#     axes[1].axis("off")
#
#     axes[2].imshow(image_slice, cmap="gray")
#     axes[2].imshow(pred_slice, alpha=0.35, cmap="Reds")
#     axes[2].set_title("Overlay: Prediction red")
#     axes[2].axis("off")
#
#     plt.suptitle(
#         f"{patient_name}\n"
#         f"Pred area={area_pred:.2f} cm² | Pred volume={volume_pred:.2f} mL"
#     )
#
#     plt.tight_layout()
#     plt.show()
#
#
# def test_left_atrium():
#     if not MODEL_PATH.exists():
#         raise FileNotFoundError(
#             f"Nu există modelul: {MODEL_PATH}\n"
#             "Antrenează mai întâi modelul Left Atrium."
#         )
#
#     test_images = find_test_images()
#     image_path = random.choice(test_images)
#
#     volume, spacing = load_nifti_with_spacing(str(image_path))
#
#     model = UNet()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()
#
#     pred_volume = predict_volume(model, volume)
#
#     show_left_atrium_prediction(
#         volume=volume,
#         pred_volume=pred_volume,
#         spacing=spacing,
#         patient_name=image_path.name
#     )
#
#
# if __name__ == "__main__":
#     test_left_atrium()