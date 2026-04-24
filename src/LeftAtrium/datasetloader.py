import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

def load_nifti(path):
    """ Încarcă un fișier .nii / .nii.gz și îl transformă în numpy array """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return array

def load_nifti_with_spacing(path):
    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing() # (x, y, z)
    return array, spacing

def visualize_slice(volume, slice_idx=None):
    """ Afișează un slice dintr-un volum 3D """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    plt.imshow(volume[slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()

def load_and_show_example(image_path):
    """ Încarcă o imagine și afișează un slice """
    volume = load_nifti(image_path)
    print("Volume shape:", volume.shape)
    visualize_slice(volume)

def load_image_and_mask(image_path, mask_path):
    """ Încarcă imagine + mask """
    image = load_nifti(image_path)
    mask = load_nifti(mask_path)
    return image, mask

def show_image_and_mask(image, mask):
    """ Afișează imagine + mask + overlay """
    slice_idx = image.shape[0] // 2

    # imagine
    plt.imshow(image[slice_idx], cmap='gray')
    plt.title("Image")
    plt.axis('off')
    plt.show()

    # mask
    plt.imshow(mask[slice_idx], cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    plt.show()

    # overlay
    plt.imshow(image[slice_idx], cmap='gray')
    plt.imshow(mask[slice_idx], alpha=0.4)
    plt.title("Overlay")
    plt.axis('off')
    plt.show()

def create_dataset(images_dir, masks_dir):
    """ Creează dataset din toate volumele """
    X = []
    Y = []
    files = sorted(os.listdir(images_dir))
    for file in files:
        if not file.endswith(".nii"):
            continue
        image_path = os.path.join(images_dir, file)
        mask_path = os.path.join(masks_dir, file)
        # verifică dacă există mask
        if not os.path.exists(mask_path):
            continue
        image = load_nifti(image_path)
        mask = load_nifti(mask_path)
        # parcurgem toate slice-urile
        for i in range(image.shape[0]):
            img_slice = image[i]
            mask_slice = mask[i]
            # ignorăm slice-uri goale (fără inimă)
            if mask_slice.max() == 0:
                continue
            X.append(img_slice)
            Y.append(mask_slice)

    print(f"Total samples: {len(X)}")
    return X, Y