import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.unet import UNet
from src.analysis.metrics import compute_area, compute_volume, interpret_results

def load_nifti_with_spacing(path):
    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing() # (x, y, z)
    return array, spacing


def show_image_and_mask(image, mask):
    slice_idx = image.shape[0] // 2

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    print("Mask labels:", np.unique(mask))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image[slice_idx], cmap="gray")
    ax[0].set_title("MRI")
    ax[0].axis("off")

    ax[1].imshow(mask[slice_idx], cmap="jet")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(image[slice_idx], cmap="gray")
    ax[2].imshow(mask[slice_idx], cmap="jet", alpha=0.4)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for i in range(1,10):
        image_path = f"C:\\Users\\Asus\\Documents\\Andreea\\Disertation\\data\\MnM2\\dataset\\00{i}\\00{i}_SA_ED.nii.gz"
        mask_path = f"C:\\Users\\Asus\\Documents\\Andreea\\Disertation\\data\\MnM2\\dataset\\00{i}\\00{i}_SA_ED_gt.nii.gz"
        volume, spacing = load_nifti_with_spacing(image_path)
        gt_mask_volume, _ = load_nifti_with_spacing(mask_path)
        print("Labels found:", np.unique(gt_mask_volume))
        show_image_and_mask(volume, gt_mask_volume)

