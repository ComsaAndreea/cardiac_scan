import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = PROJECT_ROOT /"src"/"SAX_LRV"/ "experiments"/ "test_results_all_models.csv"

PLOTS_DIR = PROJECT_ROOT /"src"/"SAX_LRV"/ "experiments" / "SAX_LRV" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_LABELS = {
    "unet": "U-Net",
    "attention_unet": "Attention U-Net",
    "unetpp": "U-Net++",
}


def load_results():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Nu există fișierul: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    df["model_label"] = df["model"].map(MODEL_LABELS)

    return df


def save_bar_plot(df, metrics, title, ylabel, filename):
    summary = df.groupby("model_label")[metrics].mean()

    ax = summary.plot(kind="bar", figsize=(9, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def save_boxplot(df, metric, title, ylabel, filename):
    labels = []
    data = []

    for model_key, model_label in MODEL_LABELS.items():
        values = df[df["model"] == model_key][metric].dropna()
        labels.append(model_label)
        data.append(values)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def save_mean_line_plot(df, metrics, title, ylabel, filename):
    summary = df.groupby("model_label")[metrics].mean()

    plt.figure(figsize=(8, 5))

    for metric in metrics:
        plt.plot(summary.index, summary[metric], marker="o", label=metric)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def main():
    df = load_results()

    # 1. Dice comparison
    save_bar_plot(
        df,
        metrics=["rv_dice", "myo_dice", "lv_dice"],
        title="Mean Dice Score by Model",
        ylabel="Dice Score",
        filename="01_mean_dice_by_model.png"
    )

    # 2. Precision comparison
    save_bar_plot(
        df,
        metrics=["rv_precision", "myo_precision", "lv_precision"],
        title="Mean Precision by Model",
        ylabel="Precision",
        filename="02_mean_precision_by_model.png"
    )

    # 3. Recall comparison
    save_bar_plot(
        df,
        metrics=["rv_recall", "myo_recall", "lv_recall"],
        title="Mean Recall by Model",
        ylabel="Recall",
        filename="03_mean_recall_by_model.png"
    )

    # 4. F1 comparison
    save_bar_plot(
        df,
        metrics=["rv_f1", "myo_f1", "lv_f1"],
        title="Mean F1-Score by Model",
        ylabel="F1-Score",
        filename="04_mean_f1_by_model.png"
    )

    # 5. Volume absolute error
    save_bar_plot(
        df,
        metrics=[
            "rv_volume_abs_error_ml",
            "myo_volume_abs_error_ml",
            "lv_volume_abs_error_ml"
        ],
        title="Mean Absolute Volume Error by Model",
        ylabel="Volume Error (mL)",
        filename="05_mean_volume_error_by_model.png"
    )

    # 6. Area absolute error
    save_bar_plot(
        df,
        metrics=[
            "rv_area_abs_error_cm2",
            "myo_area_abs_error_cm2",
            "lv_area_abs_error_cm2"
        ],
        title="Mean Absolute Area Error by Model",
        ylabel="Area Error (cm²)",
        filename="06_mean_area_error_by_model.png"
    )

    # 7. LV/RV volume ratio error
    save_boxplot(
        df,
        metric="lv_rv_volume_ratio_abs_error",
        title="LV/RV Volume Ratio Absolute Error",
        ylabel="Absolute Error",
        filename="07_lv_rv_volume_ratio_error_boxplot.png"
    )

    # 8. LV/RV area ratio error
    save_boxplot(
        df,
        metric="lv_rv_area_ratio_abs_error",
        title="LV/RV Area Ratio Absolute Error",
        ylabel="Absolute Error",
        filename="08_lv_rv_area_ratio_error_boxplot.png"
    )

    # 9. Dice distribution per class
    save_boxplot(
        df,
        metric="lv_dice",
        title="LV Dice Score Distribution",
        ylabel="Dice Score",
        filename="09_lv_dice_boxplot.png"
    )

    save_boxplot(
        df,
        metric="rv_dice",
        title="RV Dice Score Distribution",
        ylabel="Dice Score",
        filename="10_rv_dice_boxplot.png"
    )

    save_boxplot(
        df,
        metric="myo_dice",
        title="Myocardium Dice Score Distribution",
        ylabel="Dice Score",
        filename="11_myo_dice_boxplot.png"
    )

    print("\nPlots generated successfully.")
    print(f"Saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()