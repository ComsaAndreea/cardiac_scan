import numpy as np


def normalize_image(image):
    """
    Normalizează imaginea în intervalul [0, 1].
    """
    image = image.astype(np.float32)
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)


def pad_to_size(image, target_size=(512, 428), value=0):
    """
    Adaugă padding până la dimensiunea target_size.
    Nu face resize, deci nu deformează anatomia.
    """
    h, w = image.shape
    target_h, target_w = target_size

    if h > target_h or w > target_w:
        raise ValueError(
            f"Imaginea are dimensiunea {image.shape}, mai mare decât target_size={target_size}. "
            "Mărește target_size sau verifică datasetul."
        )

    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = np.pad(
        image,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=value
    )

    return padded