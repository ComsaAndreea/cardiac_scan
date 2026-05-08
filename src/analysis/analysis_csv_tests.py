import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# LEFT ATRIUM ANALYSIS
# =========================================================
def analyze_left_atrium():

    df = pd.read_csv("../../experiments/left_atrium_results.csv")

    print("\n==============================")
    print("LEFT ATRIUM ANALYSIS")
    print("==============================")

    # =============================
    # 📊 STATISTICI GENERALE
    # =============================
    print(f"Dice mediu: {df['dice'].mean():.4f}")
    print(f"Dice min: {df['dice'].min():.4f}")
    print(f"Dice max: {df['dice'].max():.4f}")

    print(f"\nEroare volum medie: {df['vol_error'].mean():.2f} mL")
    print(f"Eroare arie medie: {df['area_error'].mean():.2f} cm²")

    # =============================
    # 📊 HISTOGRAMĂ DICE
    # =============================
    plt.figure()
    plt.hist(df["dice"])

    plt.title(
        "Dice Score Distribution (Left Atrium)\n"
        "→ Measures overlap between predicted segmentation and ground truth"
    )
    plt.xlabel("Dice Score (0 = no overlap, 1 = perfect overlap)")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/la_dice_hist.png")
    plt.close()

    # =============================
    # 📊 HISTOGRAMĂ EROARE VOLUM
    # =============================
    plt.figure()
    plt.hist(df["vol_error"])

    plt.title(
        "Volume Error Distribution (Left Atrium)\n"
        "→ Difference between predicted and true volume (mL)"
    )
    plt.xlabel("Volume Error (mL)")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/la_volume_error_hist.png")
    plt.close()

    # =============================
    # 📊 BOXPLOT DICE
    # =============================
    plt.figure()
    plt.boxplot(df["dice"])

    plt.title(
        "Dice Score Variability (Left Atrium)\n"
        "→ Shows median, spread and outliers"
    )
    plt.ylabel("Dice Score")

    plt.savefig("experiments/la_dice_boxplot.png")
    plt.close()

    # =============================
    # 🧠 WORST CASES (cele mai slabe rezultate)
    # =============================
    worst = df.sort_values("dice").head(5)
    worst.to_csv("experiments/la_worst_cases.csv", index=False)

    print("✔ Worst Left Atrium cases saved!")


# =========================================================
# ACDC (LEFT + RIGHT VENTRICLE) ANALYSIS
# =========================================================
def analyze_acdc():

    df = pd.read_csv("../../experiments/lrv_results_unet.csv")

    print("\n==============================")
    print("ACDC ANALYSIS (LV / RV)")
    print("==============================")

    # =============================
    # 📊 STATISTICI
    # =============================
    print(f"Dice LV mediu: {df['dice_lv'].mean():.4f}")
    print(f"Dice RV mediu: {df['dice_rv'].mean():.4f}")

    print(f"\nLRVR error mediu: {df['lrvr_error'].mean():.2f}")
    print(f"Area ratio mediu: {df['area_ratio'].mean():.2f}")
    print(f"Diameter ratio mediu: {df['diameter_ratio'].mean():.2f}")

    # =============================
    # 📊 HISTOGRAMĂ DICE LV
    # =============================
    plt.figure()
    plt.hist(df["dice_lv"])

    plt.title(
        "Dice Score Distribution - Left Ventricle\n"
        "→ Evaluates segmentation accuracy for LV"
    )
    plt.xlabel("Dice Score")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/acdc_dice_lv.png")
    plt.close()

    # =============================
    # 📊 HISTOGRAMĂ DICE RV
    # =============================
    plt.figure()
    plt.hist(df["dice_rv"])

    plt.title(
        "Dice Score Distribution - Right Ventricle\n"
        "→ Evaluates segmentation accuracy for RV"
    )
    plt.xlabel("Dice Score")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/acdc_dice_rv.png")
    plt.close()

    # =============================
    # 📊 HISTOGRAMĂ LRVR ERROR
    # =============================
    plt.figure()
    plt.hist(df["lrvr_error"])

    plt.title(
        "LRVR Error Distribution\n"
        "→ Error in Left-to-Right Ventricular Volume Ratio"
    )
    plt.xlabel("LRVR Error")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/acdc_lrvr_error.png")
    plt.close()

    # =============================
    # 📊 AREA RATIO DISTRIBUTION
    # =============================
    plt.figure()
    plt.hist(df["area_ratio"])

    plt.title(
        "LV/RV Area Ratio Distribution\n"
        "→ Used to detect possible aortic coarctation"
    )
    plt.xlabel("Area Ratio (LV/RV)")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/acdc_area_ratio.png")
    plt.close()

    # =============================
    # 📊 DIAMETER RATIO DISTRIBUTION
    # =============================
    plt.figure()
    plt.hist(df["diameter_ratio"])

    plt.title(
        "RV/LV Diameter Ratio Distribution\n"
        "→ Used to detect RV dilation (e.g. pulmonary embolism)"
    )
    plt.xlabel("Diameter Ratio (RV/LV)")
    plt.ylabel("Number of Patients")

    plt.savefig("experiments/acdc_diameter_ratio.png")
    plt.close()

    # =============================
    # 🧠 WORST CASES
    # =============================
    worst_lv = df.sort_values("dice_lv").head(5)
    worst_lv.to_csv("experiments/acdc_worst_lv.csv", index=False)

    worst_rv = df.sort_values("dice_rv").head(5)
    worst_rv.to_csv("experiments/acdc_worst_rv.csv", index=False)

    print("✔ Worst ACDC cases saved!")


# =========================================================
# MAIN
# =========================================================
def main():

    os.makedirs("experiments", exist_ok=True)

    analyze_left_atrium()
    analyze_acdc()

    print("\n==============================")
    print("✔ ANALYSIS COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()