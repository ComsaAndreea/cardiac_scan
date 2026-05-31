from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_nifti_with_spacing(path):
    path = Path(path)

    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        return array, spacing

    except Exception as e:
        print(f"SimpleITK failed for {path}")
        print(f"Fallback to nibabel. Error: {e}")

        import nibabel as nib

        img = nib.load(str(path))
        data = np.asarray(img.get_fdata())
        spacing = img.header.get_zooms()

        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        elif data.ndim == 2:
            data = data[np.newaxis, :, :]

        spacing = (
            float(spacing[0]),
            float(spacing[1]),
            float(spacing[2]) if len(spacing) > 2 else 1.0
        )

        return data, spacing


def find_image_label_pairs(split_dir):
    split_dir = Path(split_dir)

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Nu există folderul: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Nu există folderul: {labels_dir}")

    pairs = []

    for image_path in sorted(images_dir.glob("*.nii.gz")):
        label_name = image_path.name.replace(".nii.gz", "_gt.nii.gz")
        label_path = labels_dir / label_name

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"Lipsește label pentru: {image_path.name}")

    if len(pairs) == 0:
        raise FileNotFoundError(f"Nu am găsit perechi image-label în {split_dir}")

    return pairs


def load_sax_dataset(split_dir):
    pairs = find_image_label_pairs(split_dir)

    X = []
    Y = []

    for image_path, label_path in pairs:
        volume, _ = load_nifti_with_spacing(image_path)
        mask_volume, _ = load_nifti_with_spacing(label_path)

        for i in range(volume.shape[0]):
            image_slice = volume[i].astype(np.float32)
            mask_slice = mask_volume[i].astype(np.int64)

            if np.sum(mask_slice > 0) == 0:
                continue

            X.append(image_slice)
            Y.append(mask_slice)

    print(f"Loaded {len(X)} SAX slices from {split_dir}")

    return X, Y