import os
import SimpleITK as sitk


def load_nifti(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return array


def load_nifti_with_spacing(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    return array, spacing


def read_info_cfg(path):
    ed = None

    with open(path, "r") as f:
        for line in f:
            if "ED" in line:
                ed = int(line.split(":")[1].strip())

    return ed


def load_acdc_dataset(root_dir, target="LV"):
    X = []
    Y = []

    patients = os.listdir(root_dir)

    for p in patients:
        patient_path = os.path.join(root_dir, p)

        if not os.path.isdir(patient_path):
            continue

        info_path = os.path.join(patient_path, "Info.cfg")
        if not os.path.exists(info_path):
            continue

        ed = read_info_cfg(info_path)
        if ed is None:
            continue

        ed_str = str(ed).zfill(2)

        image_path = os.path.join(patient_path, f"{p}_frame{ed_str}.nii.gz")
        mask_path = os.path.join(patient_path, f"{p}_frame{ed_str}_gt.nii.gz")

        if not os.path.exists(image_path):
            continue

        image = load_nifti(image_path)
        mask = load_nifti(mask_path)

        for i in range(image.shape[0]):

            img_slice = image[i]
            mask_slice = mask[i]

            if mask_slice.max() == 0:
                continue

            # 🎯 SELECTARE CLASĂ
            if target == "LV":
                mask_slice = (mask_slice == 3).astype(float)
            elif target == "RV":
                mask_slice = (mask_slice == 1).astype(float)

            # ignorăm dacă nu există acea structură
            if mask_slice.max() == 0:
                continue

            X.append(img_slice)
            Y.append(mask_slice)

    print(f"[{target}] Total samples: {len(X)}")
    return X, Y