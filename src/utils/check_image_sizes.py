from pathlib import Path
import SimpleITK as sitk
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACDC_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "LRVentricle" / "training"
ACDC_TEST_DIR = PROJECT_ROOT / "data" / "raw" / "LRVentricle" / "testing"


def read_shape(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    # Pentru fișiere 3D: shape = (slices, height, width)
    if len(array.shape) == 3:
        return array.shape[1], array.shape[2]

    # Pentru fișiere 2D: shape = (height, width)
    if len(array.shape) == 2:
        return array.shape[0], array.shape[1]

    return array.shape


def scan_folder(folder):
    results = []

    for file in folder.rglob("*.nii.gz"):
        if "_4d" in file.name:
            continue

        shape = read_shape(file)
        results.append((file, shape))

    return results


def main():
    all_results = []

    if ACDC_TRAIN_DIR.exists():
        all_results.extend(scan_folder(ACDC_TRAIN_DIR))

    if ACDC_TEST_DIR.exists():
        all_results.extend(scan_folder(ACDC_TEST_DIR))

    if len(all_results) == 0:
        print("Nu am găsit fișiere .nii.gz.")
        return

    max_h = max(shape[0] for _, shape in all_results)
    max_w = max(shape[1] for _, shape in all_results)

    print("\n==============================")
    print("ACDC IMAGE SIZE CHECK")
    print("==============================")
    print(f"Număr fișiere verificate: {len(all_results)}")
    print(f"Max height: {max_h}")
    print(f"Max width:  {max_w}")
    print(f"Target size minim recomandat: ({max_h}, {max_w})")

    square = max(max_h, max_w)
    print(f"Target size pătrat safe: ({square}, {square})")
    print("==============================\n")

    print("Cele mai mari imagini:")
    largest = sorted(
        all_results,
        key=lambda x: x[1][0] * x[1][1],
        reverse=True
    )[:10]

    for file, shape in largest:
        print(f"{shape} -> {file}")

    print("\nDistribuție dimensiuni:")
    counter = Counter(shape for _, shape in all_results)

    for shape, count in counter.most_common():
        print(f"{shape}: {count} fișiere")


if __name__ == "__main__":
    main()