import numpy as np


def compute_area(mask_slice, spacing):
    """
    Calculează aria unui slice (cm^2)
    """
    pixel_area = spacing[0] * spacing[1]  # mm^2
    area_pixels = np.sum(mask_slice > 0)

    area_mm2 = area_pixels * pixel_area
    area_cm2 = area_mm2 / 100  # mm^2 → cm^2

    return area_cm2


def compute_volume(mask_volume, spacing):
    """
    Calculează volumul total (ml)
    """
    pixel_area = spacing[0] * spacing[1]
    slice_thickness = spacing[2]

    volume = 0

    for i in range(mask_volume.shape[0]):
        mask_slice = mask_volume[i]
        area_pixels = np.sum(mask_slice > 0)

        area_mm2 = area_pixels * pixel_area
        volume += area_mm2 * slice_thickness

    volume_ml = volume / 1000  # mm^3 → ml

    return volume_ml


def interpret_results(area_cm2, volume_ml, BSA=1.8):
    """
    Interpretare simplă medicală
    """
    LAVi = volume_ml / BSA

    print(f"Area: {area_cm2:.2f} cm^2")
    print(f"Volume: {volume_ml:.2f} mL")
    print(f"LAVi: {LAVi:.2f} mL/m^2")

    # interpretare aria
    if area_cm2 > 30:
        print("⚠️ Arie mare → posibilă presiune crescută (PAWP)")
    else:
        print("✅ Arie normală")

    # interpretare LAVi
    if LAVi > 38:
        print("⚠️ LAVi ridicat → risc mare")
    elif LAVi > 34:
        print("⚠️ LAVi borderline")
    else:
        print("✅ LAVi normal")