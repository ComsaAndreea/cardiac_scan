import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACDC_ROOT = PROJECT_ROOT.parent / "data" / "ResourcesACDC"
MNM2_ROOT = PROJECT_ROOT.parent / "data" / "MnM2" / "dataset"
MNM2_INFO_CSV = PROJECT_ROOT.parent / "data" / "MnM2" / "dataset_information.csv"

OUTPUT_ROOT = PROJECT_ROOT.parent / "data" / "CombinedSAX_ED"
IMAGES_OUT = OUTPUT_ROOT / "images"
LABELS_OUT = OUTPUT_ROOT / "labels"
METADATA_PATH = OUTPUT_ROOT / "metadata.csv"


def ensure_dirs():
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.mkdir(parents=True, exist_ok=True)


def clean_text(value):
    if value is None:
        return "UNK"

    value = str(value).strip()

    if value == "":
        return "UNK"

    value = value.replace(" ", "")
    value = value.replace("-", "")
    value = value.replace("_", "")

    return value.upper()


def copy_file(src, dst):
    if not src.exists():
        print(f"Missing: {src}")
        return False

    shutil.copy2(src, dst)
    return True


def read_acdc_info_cfg(info_path):
    data = {}

    if not info_path.exists():
        return data

    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
            elif "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()

    return data


def get_acdc_ed_frame(info_data):
    for key in ["ED", "ed", "End-diastole", "EndDiastole"]:
        if key in info_data:
            return str(info_data[key]).zfill(2)

    return None


def get_acdc_pathology(info_data):
    for key in ["Group", "group", "Diagnosis", "diagnosis"]:
        if key in info_data:
            return clean_text(info_data[key])

    return "UNK"


def load_mnm2_info():
    """
    Încearcă să citească dataset_information.csv indiferent de denumirea exactă a coloanelor.
    Returnează:
      {
        "001": {
            "pathology": "...",
            "vendor": "...",
            "scanner": "...",
            "field_strength": "..."
        }
      }
    """
    info = {}

    if not MNM2_INFO_CSV.exists():
        print(f"MNM2 info CSV missing: {MNM2_INFO_CSV}")
        return info

    with open(MNM2_INFO_CSV, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            normalized = {k.lower().strip(): v for k, v in row.items() if k is not None}

            patient = None

            for key in ["subject", "subject_code", "code", "patient", "patient_id", "id", "external_code"]:
                if key in normalized and normalized[key].strip() != "":
                    patient = normalized[key].strip()
                    break

            if patient is None:
                # fallback: prima coloană
                first_value = list(row.values())[0]
                patient = str(first_value).strip()

            patient = patient.replace("sub-", "")
            patient = patient.zfill(3)

            pathology = "UNK"
            for key in ["pathology", "disease", "diagnosis", "group", "disease_category"]:
                if key in normalized and normalized[key].strip() != "":
                    pathology = clean_text(normalized[key])
                    break

            vendor = "UNK"
            for key in ["vendor", "manufacturer"]:
                if key in normalized and normalized[key].strip() != "":
                    vendor = clean_text(normalized[key])
                    break

            scanner = "UNK"
            for key in ["scanner", "scanner_model", "model"]:
                if key in normalized and normalized[key].strip() != "":
                    scanner = clean_text(normalized[key])
                    break

            field_strength = "UNK"
            for key in ["field_strength", "fieldstrength", "tesla"]:
                if key in normalized and normalized[key].strip() != "":
                    field_strength = clean_text(normalized[key])
                    break

            info[patient] = {
                "pathology": pathology,
                "vendor": vendor,
                "scanner": scanner,
                "field_strength": field_strength,
            }

    return info


def get_mnm2_original_split(patient):
    """
    Conform README M&Ms-2:
      001-160 training
      161-200 validation
      201-360 testing
    """
    try:
        patient_num = int(patient)
    except ValueError:
        return "unknown"

    if 1 <= patient_num <= 160:
        return "training"

    if 161 <= patient_num <= 200:
        return "validation"

    if 201 <= patient_num <= 360:
        return "testing"

    return "unknown"


def collect_acdc_rows():
    rows = []

    for original_split in ["training", "testing"]:
        split_dir = ACDC_ROOT / original_split

        if not split_dir.exists():
            print(f"ACDC split missing: {split_dir}")
            continue

        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            patient = patient_dir.name
            info_path = patient_dir / "Info.cfg"
            info_data = read_acdc_info_cfg(info_path)

            ed_frame = get_acdc_ed_frame(info_data)
            pathology = get_acdc_pathology(info_data)

            if ed_frame is None:
                print(f"Missing ED frame in Info.cfg for {patient}")
                continue

            image_path = patient_dir / f"{patient}_frame{ed_frame}.nii.gz"
            gt_path = patient_dir / f"{patient}_frame{ed_frame}_gt.nii.gz"

            if not image_path.exists() or not gt_path.exists():
                print(f"Missing ACDC ED files for {patient}: frame {ed_frame}")
                continue

            new_name = f"acdc_{patient}_SAX_ED_{pathology}.nii.gz"
            new_gt_name = f"acdc_{patient}_SAX_ED_{pathology}_gt.nii.gz"

            ok_img = copy_file(image_path, IMAGES_OUT / new_name)
            ok_gt = copy_file(gt_path, LABELS_OUT / new_gt_name)

            if ok_img and ok_gt:
                rows.append({
                    "source": "ACDC",
                    "original_split": original_split,
                    "patient": patient,
                    "phase": "ED",
                    "frame": ed_frame,
                    "view": "SAX",
                    "pathology": pathology,
                    "vendor": "UNK",
                    "scanner": "UNK",
                    "field_strength": "UNK",
                    "image": new_name,
                    "label": new_gt_name
                })

    return rows


def collect_mnm2_rows():
    rows = []
    mnm2_info = load_mnm2_info()

    if not MNM2_ROOT.exists():
        print(f"MNM2 root missing: {MNM2_ROOT}")
        return rows

    for patient_dir in sorted(MNM2_ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient = patient_dir.name.zfill(3)

        image_path = patient_dir / f"{patient}_SA_ED.nii.gz"
        gt_path = patient_dir / f"{patient}_SA_ED_gt.nii.gz"

        if not image_path.exists() or not gt_path.exists():
            print(f"Missing MNM2 ED SAX files for {patient}")
            continue

        patient_info = mnm2_info.get(patient, {})

        pathology = patient_info.get("pathology", "UNK")
        vendor = patient_info.get("vendor", "UNK")
        scanner = patient_info.get("scanner", "UNK")
        field_strength = patient_info.get("field_strength", "UNK")
        original_split = get_mnm2_original_split(patient)

        new_name = f"mnm2_{patient}_SAX_ED_{pathology}.nii.gz"
        new_gt_name = f"mnm2_{patient}_SAX_ED_{pathology}_gt.nii.gz"

        ok_img = copy_file(image_path, IMAGES_OUT / new_name)
        ok_gt = copy_file(gt_path, LABELS_OUT / new_gt_name)

        if ok_img and ok_gt:
            rows.append({
                "source": "MNM2",
                "original_split": original_split,
                "patient": patient,
                "phase": "ED",
                "frame": "ED",
                "view": "SAX",
                "pathology": pathology,
                "vendor": vendor,
                "scanner": scanner,
                "field_strength": field_strength,
                "image": new_name,
                "label": new_gt_name
            })

    return rows


def save_metadata(rows):
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source",
            "original_split",
            "patient",
            "phase",
            "frame",
            "view",
            "pathology",
            "vendor",
            "scanner",
            "field_strength",
            "image",
            "label"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()

    rows = []

    print("Collecting ACDC ED SAX...")
    acdc_rows = collect_acdc_rows()
    rows.extend(acdc_rows)

    print("Collecting MNM2 ED SAX...")
    mnm2_rows = collect_mnm2_rows()
    rows.extend(mnm2_rows)

    save_metadata(rows)

    print("\nDone.")
    print(f"ACDC samples: {len(acdc_rows)}")
    print(f"MNM2 samples: {len(mnm2_rows)}")
    print(f"Total samples: {len(rows)}")
    print(f"Images folder: {IMAGES_OUT}")
    print(f"Labels folder: {LABELS_OUT}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()