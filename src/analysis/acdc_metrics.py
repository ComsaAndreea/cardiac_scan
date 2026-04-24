import numpy as np


# =========================================================
# 🔥 UTIL: verificăm dacă un slice conține inimă
# =========================================================
def is_valid_slice(slice_mask, min_pixels=200):
    """
    Elimină slice-urile fără ventriculi
    """
    return np.sum(slice_mask > 0) > min_pixels


# =========================================================
# 🧮 VOLUME CALCUL (corect medical)
# =========================================================
def compute_volume(mask, label, spacing):
    """
    Volum în mL pentru LV sau RV
    """

    pixel_area = spacing[0] * spacing[1]  # mm²
    slice_thickness = spacing[2]          # mm

    volume_mm3 = 0

    for i in range(mask.shape[0]):
        slice_mask = mask[i]

        # 🔴 ignorăm slice-uri goale
        if not is_valid_slice(slice_mask):
            continue

        pixels = np.sum(slice_mask == label)

        if pixels == 0:
            continue

        area_mm2 = pixels * pixel_area
        volume_mm3 += area_mm2 * slice_thickness

    return volume_mm3 / 1000.0  # mL


# =========================================================
# 📏 AREA CALCUL (alegem slice-ul cel mai relevant)
# =========================================================
def compute_area(mask, label, spacing):
    """
    Arie LV / RV din slice-ul cu cea mai mare activitate
    """

    best_slice = None
    best_pixels = 0

    for i in range(mask.shape[0]):
        slice_mask = mask[i]
        pixels = np.sum(slice_mask == label)

        if pixels > best_pixels:
            best_pixels = pixels
            best_slice = slice_mask

    if best_slice is None:
        return 0

    pixel_area = spacing[0] * spacing[1]
    area_cm2 = best_pixels * pixel_area / 100.0

    return area_cm2


# =========================================================
# 📊 METRICS COMPLETE
# =========================================================
def compute_all_metrics(mask, spacing):
    """
    LV = 3
    RV = 1
    """

    lv_label = 3
    rv_label = 1

    volume_lv = compute_volume(mask, lv_label, spacing)
    volume_rv = compute_volume(mask, rv_label, spacing)

    lrvr = volume_lv / (volume_rv + 1e-8)

    area_lv = compute_area(mask, lv_label, spacing)
    area_rv = compute_area(mask, rv_label, spacing)

    area_ratio = area_lv / (area_rv + 1e-8)

    return volume_lv, volume_rv, lrvr, area_lv, area_rv, area_ratio


# =========================================================
# 🧠 INTERPRETARE MEDICALĂ
# =========================================================
def interpret_results(lrvr, area_ratio):
    print("\n==============================")
    print("📊 INTERPRETARE MEDICALĂ")
    print("==============================")

    # LRVR
    print(f"\n🔹 LV/RV Volume Ratio (LRVR): {lrvr:.2f}")

    if lrvr < 0.8:
        print("⚠️ LV prea mic → posibil disfuncție cardiacă")
    elif lrvr > 1.3:
        print("⚠️ LV prea mare → posibilă dilatare ventriculară stângă")
    else:
        print("✅ LRVR normal")

    # AREA RATIO
    print(f"\n🔹 LV/RV Area Ratio: {area_ratio:.2f}")

    if area_ratio > 1.3:
        print("⚠️ posibil RV subdimensionat / LV dominant")
    elif area_ratio < 0.6:
        print("⚠️ posibil RV dilatat")
    else:
        print("✅ Area ratio normal")

    print("\n==============================\n")