from pathlib import Path

import numpy as np
import SimpleITK as sitk


RV_LABEL = 1
LV_LABEL = 3


def load_nifti_with_spacing(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    return array, spacing


def find_mnm2_long_axis_pairs(root_dir):
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Nu există folderul MnM2: {root_dir}")

    pairs = []

    for patient_dir in sorted(root_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient = patient_dir.name

        for phase in ["ED", "ES"]:
            image_path = patient_dir / f"{patient}_LA_{phase}.nii.gz"
            label_path = patient_dir / f"{patient}_LA_{phase}_gt.nii.gz"

            if image_path.exists() and label_path.exists():
                pairs.append((patient, phase, image_path, label_path))

    if len(pairs) == 0:
        raise FileNotFoundError(
            f"Nu am găsit perechi long-axis ED/ES în: {root_dir}\n"
            "Mă aștept la fișiere de forma 001_LA_ED.nii.gz și 001_LA_ED_gt.nii.gz"
        )

    return pairs


def extract_slices_from_pair(image_path, label_path, target):
    image_volume, _ = load_nifti_with_spacing(image_path)
    label_volume, _ = load_nifti_with_spacing(label_path)

    if target == "LV":
        target_label = LV_LABEL
    elif target == "RV":
        target_label = RV_LABEL
    else:
        raise ValueError("target trebuie să fie 'LV' sau 'RV'")

    X = []
    Y = []

    for i in range(image_volume.shape[0]):
        image_slice = image_volume[i].astype(np.float32)
        label_slice = label_volume[i]

        mask_slice = (label_slice == target_label).astype(np.float32)

        if np.sum(mask_slice) == 0:
            continue

        X.append(image_slice)
        Y.append(mask_slice)

    return X, Y


def load_mnm_dataset(root_dir, target):
    pairs = find_mnm2_long_axis_pairs(root_dir)

    all_X = []
    all_Y = []

    for patient, phase, image_path, label_path in pairs:
        X, Y = extract_slices_from_pair(image_path, label_path, target)
        all_X.extend(X)
        all_Y.extend(Y)

    print(f"M&M2 Long-Axis {target}: {len(all_X)} slice-uri încărcate.")

    return all_X, all_Y