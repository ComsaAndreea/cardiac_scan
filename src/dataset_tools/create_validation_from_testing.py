import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "CombinedSAX_ED_split"

TEST_IMAGES = DATA_ROOT / "testing" / "images"
TEST_LABELS = DATA_ROOT / "testing" / "labels"

VAL_IMAGES = DATA_ROOT / "validation" / "images"
VAL_LABELS = DATA_ROOT / "validation" / "labels"

METADATA_PATH = DATA_ROOT / "metadata.csv"
REPORT_PATH = DATA_ROOT / "validation_split_report.txt"

VALIDATION_RATIO_FROM_TESTING = 0.50
RANDOM_SEED = 42


DISEASE_LEGEND = {
    "NOR": "Normal control",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "MINF": "Myocardial infarction",
    "RV": "Abnormal right ventricle",
    "LV": "Abnormal left ventricle",
    "UNK": "Unknown / not available",
}


def ensure_dirs():
    VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    VAL_LABELS.mkdir(parents=True, exist_ok=True)


def patient_key(row):
    return f"{row['source']}_{row['patient']}"


def read_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Nu există metadata.csv: {METADATA_PATH}")

    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_metadata(rows):
    fieldnames = list(rows[0].keys())

    if "final_split" not in fieldnames:
        fieldnames.append("final_split")

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def move_sample_to_validation(row):
    image_name = row["image"]
    label_name = row["label"]

    src_image = TEST_IMAGES / image_name
    src_label = TEST_LABELS / label_name

    dst_image = VAL_IMAGES / image_name
    dst_label = VAL_LABELS / label_name

    if not src_image.exists():
        raise FileNotFoundError(f"Lipsește imaginea în testing: {src_image}")

    if not src_label.exists():
        raise FileNotFoundError(f"Lipsește label-ul în testing: {src_label}")

    shutil.move(str(src_image), str(dst_image))
    shutil.move(str(src_label), str(dst_label))


def select_validation_patients(test_rows):
    patients = {}

    for row in test_rows:
        key = patient_key(row)

        if key not in patients:
            patients[key] = {
                "pathology": row.get("pathology", "UNK") or "UNK",
                "rows": [],
            }

        patients[key]["rows"].append(row)

    by_pathology = defaultdict(list)

    for key, item in patients.items():
        by_pathology[item["pathology"]].append(key)

    random.seed(RANDOM_SEED)

    validation_patients = set()

    for pathology, keys in by_pathology.items():
        keys = sorted(keys)
        random.shuffle(keys)

        n = len(keys)

        if n <= 1:
            n_val = 0
        else:
            n_val = max(1, round(n * VALIDATION_RATIO_FROM_TESTING))

        validation_patients.update(keys[:n_val])

    return validation_patients


def write_report(rows, moved_patients):
    split_patient_counter = defaultdict(set)
    split_sample_counter = Counter()
    split_pathology_counter = defaultdict(Counter)

    seen = set()

    for row in rows:
        split = row.get("final_split", "unknown")
        key = patient_key(row)
        pathology = row.get("pathology", "UNK") or "UNK"

        split_patient_counter[split].add(key)
        split_sample_counter[split] += 1

        if (split, key) not in seen:
            split_pathology_counter[split][pathology] += 1
            seen.add((split, key))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("VALIDATION SPLIT FROM EXISTING TESTING SET\n")
        f.write("=========================================\n\n")

        f.write(f"Data root: {DATA_ROOT}\n")
        f.write(f"Validation ratio from testing: {VALIDATION_RATIO_FROM_TESTING:.2f}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Moved patients to validation: {len(moved_patients)}\n\n")

        f.write("Patients by split:\n")
        for split in ["training", "validation", "testing"]:
            count = len(split_patient_counter[split])
            f.write(f"{split}: {count} patients\n")

        f.write("\nSamples by split:\n")
        for split in ["training", "validation", "testing"]:
            count = split_sample_counter[split]
            f.write(f"{split}: {count} samples\n")

        f.write("\nPathology distribution by split:\n")
        for split in ["training", "validation", "testing"]:
            f.write(f"\n{split.upper()}:\n")
            for pathology, count in sorted(split_pathology_counter[split].items()):
                meaning = DISEASE_LEGEND.get(pathology, "Not defined in legend")
                f.write(f"  {pathology}: {count} patients - {meaning}\n")

        f.write("\nDisease legend:\n")
        for key, value in sorted(DISEASE_LEGEND.items()):
            f.write(f"{key}: {value}\n")


def main():
    ensure_dirs()

    rows = read_metadata()

    # Dacă scriptul a mai fost rulat, nu îl rulăm din nou accidental.
    existing_val_rows = [r for r in rows if r.get("final_split") == "validation"]
    if len(existing_val_rows) > 0:
        raise RuntimeError(
            "Există deja rânduri cu final_split=validation în metadata.csv. "
            "Pare că validation split-ul a fost deja creat."
        )

    test_rows = [r for r in rows if r.get("final_split") == "testing"]

    if len(test_rows) == 0:
        raise RuntimeError("Nu există rânduri cu final_split=testing în metadata.csv.")

    validation_patients = select_validation_patients(test_rows)

    updated_rows = []

    for row in rows:
        key = patient_key(row)

        if row.get("final_split") == "testing" and key in validation_patients:
            move_sample_to_validation(row)
            new_row = dict(row)
            new_row["final_split"] = "validation"
            updated_rows.append(new_row)
        else:
            updated_rows.append(row)

    write_metadata(updated_rows)
    write_report(updated_rows, validation_patients)

    print("\nDone.")
    print(f"Validation patients: {len(validation_patients)}")
    print(f"Metadata updated: {METADATA_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()