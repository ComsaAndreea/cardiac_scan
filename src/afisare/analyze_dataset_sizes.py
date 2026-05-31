import csv
from pathlib import Path
from collections import defaultdict, Counter

import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMBINED_ROOT = PROJECT_ROOT / "data" / "CombinedSAX"
IMAGES_DIR = COMBINED_ROOT / "images"
LABELS_DIR = COMBINED_ROOT / "labels"
METADATA_PATH = COMBINED_ROOT / "metadata.csv"

OUTPUT_CSV = COMBINED_ROOT / "combined_sax_size_spacing_analysis.csv"


def read_nifti_info(path):
    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()

        if len(array.shape) == 3:
            slices, height, width = array.shape
        elif len(array.shape) == 2:
            slices = 1
            height, width = array.shape
        else:
            raise ValueError(f"Shape neașteptat pentru {path}: {array.shape}")

        return {
            "slices": slices,
            "height": height,
            "width": width,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2] if len(spacing) > 2 else 1.0,
            "shape": array.shape,
            "reader": "SimpleITK",
        }

    except Exception as e:
        print(f"SimpleITK failed for {path}")
        print(f"Fallback to nibabel. Error: {e}")

        import nibabel as nib

        img = nib.load(str(path))
        data = img.get_fdata()
        zooms = img.header.get_zooms()

        if len(data.shape) == 3:
            height, width, slices = data.shape
        elif len(data.shape) == 2:
            height, width = data.shape
            slices = 1
        else:
            raise ValueError(f"Shape neașteptat pentru {path}: {data.shape}")

        return {
            "slices": slices,
            "height": height,
            "width": width,
            "spacing_x": zooms[0],
            "spacing_y": zooms[1],
            "spacing_z": zooms[2] if len(zooms) > 2 else 1.0,
            "shape": data.shape,
            "reader": "nibabel",
        }

def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Nu există metadata.csv: {METADATA_PATH}")

    rows = []

    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def summarize(values):
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main():
    rows = load_metadata()

    analysis_rows = []
    grouped = defaultdict(list)

    for row in rows:
        source = row["source"]
        image_path = IMAGES_DIR / row["image"]
        label_path = LABELS_DIR / row["label"]

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        if not label_path.exists():
            print(f"Missing label: {label_path}")
            continue

        image_info = read_nifti_info(image_path)
        label_info = read_nifti_info(label_path)

        analysis_row = {
            "source": source,
            "patient": row["patient"],
            "phase": row["phase"],
            "view": row["view"],
            "image": row["image"],
            "label": row["label"],

            "image_slices": image_info["slices"],
            "image_height": image_info["height"],
            "image_width": image_info["width"],

            "label_slices": label_info["slices"],
            "label_height": label_info["height"],
            "label_width": label_info["width"],

            "spacing_x": image_info["spacing_x"],
            "spacing_y": image_info["spacing_y"],
            "spacing_z": image_info["spacing_z"],
        }

        analysis_rows.append(analysis_row)
        grouped[source].append(analysis_row)

    if len(analysis_rows) == 0:
        raise RuntimeError("Nu am putut analiza niciun fișier.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(analysis_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)

    print("\n==============================")
    print("COMBINED SAX DATASET ANALYSIS")
    print("==============================")
    print(f"Total samples: {len(analysis_rows)}")
    print(f"Output CSV: {OUTPUT_CSV}")

    for source, source_rows in grouped.items():
        print("\n------------------------------")
        print(f"Source: {source}")
        print("------------------------------")

        heights = [r["image_height"] for r in source_rows]
        widths = [r["image_width"] for r in source_rows]
        slices = [r["image_slices"] for r in source_rows]

        spacing_x = [r["spacing_x"] for r in source_rows]
        spacing_y = [r["spacing_y"] for r in source_rows]
        spacing_z = [r["spacing_z"] for r in source_rows]

        print(f"Samples: {len(source_rows)}")

        print(f"Height min/max/mean: {min(heights)} / {max(heights)} / {sum(heights)/len(heights):.2f}")
        print(f"Width  min/max/mean: {min(widths)} / {max(widths)} / {sum(widths)/len(widths):.2f}")
        print(f"Slices min/max/mean: {min(slices)} / {max(slices)} / {sum(slices)/len(slices):.2f}")

        print(f"Spacing X min/max/mean: {min(spacing_x):.4f} / {max(spacing_x):.4f} / {sum(spacing_x)/len(spacing_x):.4f}")
        print(f"Spacing Y min/max/mean: {min(spacing_y):.4f} / {max(spacing_y):.4f} / {sum(spacing_y)/len(spacing_y):.4f}")
        print(f"Spacing Z min/max/mean: {min(spacing_z):.4f} / {max(spacing_z):.4f} / {sum(spacing_z)/len(spacing_z):.4f}")

        shape_counter = Counter((r["image_height"], r["image_width"]) for r in source_rows)

        print("\nMost common HxW:")
        for shape, count in shape_counter.most_common(10):
            print(f"{shape}: {count}")

    all_heights = [r["image_height"] for r in analysis_rows]
    all_widths = [r["image_width"] for r in analysis_rows]

    target_h = max(all_heights)
    target_w = max(all_widths)

    print("\n==============================")
    print("RECOMMENDED TARGET SIZE")
    print("==============================")
    print(f"Minimum safe target: ({target_h}, {target_w})")

    square = max(target_h, target_w)
    print(f"Square safe target: ({square}, {square})")

    print("\nMultiples of 32 suggestion:")
    target_h_32 = ((target_h + 31) // 32) * 32
    target_w_32 = ((target_w + 31) // 32) * 32
    print(f"Nearest safe multiple of 32: ({target_h_32}, {target_w_32})")


if __name__ == "__main__":
    main()