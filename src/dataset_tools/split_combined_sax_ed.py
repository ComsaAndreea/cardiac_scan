import csv
import math
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "data" / "CombinedSAX_ED"
INPUT_IMAGES = INPUT_ROOT / "images"
INPUT_LABELS = INPUT_ROOT / "labels"
INPUT_METADATA = INPUT_ROOT / "metadata.csv"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "CombinedSAX_ED_split"
TRAIN_IMAGES = OUTPUT_ROOT / "training" / "images"
TRAIN_LABELS = OUTPUT_ROOT / "training" / "labels"
TEST_IMAGES = OUTPUT_ROOT / "testing" / "images"
TEST_LABELS = OUTPUT_ROOT / "testing" / "labels"

OUTPUT_METADATA = OUTPUT_ROOT / "metadata.csv"
SPLIT_REPORT = OUTPUT_ROOT / "split_report.txt"

TEST_RATIO = 0.30
RANDOM_SEED = 42


DISEASE_LEGEND = {
    "NOR": "Normal control",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "MINF": "Myocardial infarction",
    "RV": "Abnormal right ventricle",
    "LV": "Abnormal left ventricle",
    "ARR": "Arrhythmia",
    "ARV": "Arrhythmogenic right ventricular cardiomyopathy",
    "IHD": "Ischemic heart disease",
    "DC": "Dilated cardiomyopathy",
    "UNK": "Unknown / not available",
}


def ensure_dirs():
    for folder in [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]:
        folder.mkdir(parents=True, exist_ok=True)


def read_metadata():
    if not INPUT_METADATA.exists():
        raise FileNotFoundError(f"Nu există metadata.csv: {INPUT_METADATA}")

    with open(INPUT_METADATA, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def patient_key(row):
    return f"{row['source']}_{row['patient']}"


def stratified_patient_split(rows):
    """
    Split stratificat pe pathology, la nivel de pacient.
    Un pacient nu ajunge și în training și în testing.
    """
    patients = {}

    for row in rows:
        key = patient_key(row)

        if key not in patients:
            patients[key] = {
                "source": row["source"],
                "patient": row["patient"],
                "pathology": row.get("pathology", "UNK"),
                "rows": []
            }

        patients[key]["rows"].append(row)

    by_pathology = defaultdict(list)

    for key, item in patients.items():
        pathology = item["pathology"] or "UNK"
        by_pathology[pathology].append(key)

    random.seed(RANDOM_SEED)

    train_patients = set()
    test_patients = set()

    for pathology, keys in by_pathology.items():
        keys = sorted(keys)
        random.shuffle(keys)

        n = len(keys)

        if n == 1:
            n_test = 0
        else:
            n_test = max(1, int(round(n * TEST_RATIO)))

        test_keys = set(keys[:n_test])
        train_keys = set(keys[n_test:])

        test_patients.update(test_keys)
        train_patients.update(train_keys)

    return patients, train_patients, test_patients


def copy_sample(row, split):
    image_name = row["image"]
    label_name = row["label"]

    src_image = INPUT_IMAGES / image_name
    src_label = INPUT_LABELS / label_name

    if not src_image.exists():
        raise FileNotFoundError(f"Lipsește imaginea: {src_image}")

    if not src_label.exists():
        raise FileNotFoundError(f"Lipsește label-ul: {src_label}")

    if split == "training":
        dst_image = TRAIN_IMAGES / image_name
        dst_label = TRAIN_LABELS / label_name
    elif split == "testing":
        dst_image = TEST_IMAGES / image_name
        dst_label = TEST_LABELS / label_name
    else:
        raise ValueError(f"Split invalid: {split}")

    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_label, dst_label)


def write_metadata(rows):
    if len(rows) == 0:
        raise RuntimeError("Nu există rânduri pentru metadata.")

    fieldnames = list(rows[0].keys())

    if "final_split" not in fieldnames:
        fieldnames.append("final_split")

    with open(OUTPUT_METADATA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_patients_by_split(rows):
    split_patients = defaultdict(set)

    for row in rows:
        split = row["final_split"]
        split_patients[split].add(patient_key(row))

    return {split: len(patients) for split, patients in split_patients.items()}


def count_samples_by_split(rows):
    return Counter(row["final_split"] for row in rows)


def count_pathology_by_split(rows):
    result = defaultdict(Counter)

    seen = set()

    for row in rows:
        key = (row["final_split"], patient_key(row))

        if key in seen:
            continue

        seen.add(key)

        pathology = row.get("pathology", "UNK") or "UNK"
        result[row["final_split"]][pathology] += 1

    return result


def write_report(rows):
    patient_counts = count_patients_by_split(rows)
    sample_counts = count_samples_by_split(rows)
    pathology_counts = count_pathology_by_split(rows)

    total_patients = sum(patient_counts.values())
    total_samples = sum(sample_counts.values())

    with open(SPLIT_REPORT, "w", encoding="utf-8") as f:
        f.write("COMBINED SAX ED SPLIT REPORT\n")
        f.write("============================\n\n")

        f.write(f"Input folder: {INPUT_ROOT}\n")
        f.write(f"Output folder: {OUTPUT_ROOT}\n")
        f.write(f"Test ratio requested: {TEST_RATIO:.2f}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("TOTALS\n")
        f.write("------\n")
        f.write(f"Total patients: {total_patients}\n")
        f.write(f"Total samples: {total_samples}\n\n")

        f.write("PATIENTS BY SPLIT\n")
        f.write("-----------------\n")
        for split in ["training", "testing"]:
            count = patient_counts.get(split, 0)
            percent = count / total_patients * 100 if total_patients > 0 else 0
            f.write(f"{split}: {count} patients ({percent:.2f}%)\n")

        f.write("\nSAMPLES BY SPLIT\n")
        f.write("----------------\n")
        for split in ["training", "testing"]:
            count = sample_counts.get(split, 0)
            percent = count / total_samples * 100 if total_samples > 0 else 0
            f.write(f"{split}: {count} samples ({percent:.2f}%)\n")

        f.write("\nPATHOLOGY DISTRIBUTION BY SPLIT\n")
        f.write("-------------------------------\n")
        for split in ["training", "testing"]:
            f.write(f"\n{split.upper()}:\n")
            counter = pathology_counts.get(split, Counter())

            total = sum(counter.values())

            for pathology, count in sorted(counter.items()):
                percent = count / total * 100 if total > 0 else 0
                meaning = DISEASE_LEGEND.get(pathology, "Not defined in legend")
                f.write(f"  {pathology}: {count} patients ({percent:.2f}%) - {meaning}\n")

        f.write("\nDISEASE LEGEND\n")
        f.write("--------------\n")
        for key, value in sorted(DISEASE_LEGEND.items()):
            f.write(f"{key}: {value}\n")


def main():
    ensure_dirs()

    rows = read_metadata()
    patients, train_patients, test_patients = stratified_patient_split(rows)

    output_rows = []

    for row in rows:
        key = patient_key(row)

        if key in test_patients:
            final_split = "testing"
        else:
            final_split = "training"

        new_row = dict(row)
        new_row["final_split"] = final_split

        copy_sample(row, final_split)
        output_rows.append(new_row)

    write_metadata(output_rows)
    write_report(output_rows)

    print("\nDone.")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Metadata: {OUTPUT_METADATA}")
    print(f"Report: {SPLIT_REPORT}")

    print("\nPatient counts:")
    print(count_patients_by_split(output_rows))

    print("\nSample counts:")
    print(count_samples_by_split(output_rows))

    print("\nPathology by split:")
    pathology_counts = count_pathology_by_split(output_rows)
    for split, counter in pathology_counts.items():
        print(split, dict(counter))


if __name__ == "__main__":
    main()