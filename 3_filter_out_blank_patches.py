# filter_blank_patches.py

from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

INPUT_PATCH_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/patches_from_downsample_4.0")
OUTPUT_PATCH_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/clean_patches_from_downsample_4.0")

INPUT_METADATA = INPUT_PATCH_DIR / "patch_metadata.csv"

# Stronger blank/background filtering
MIN_TISSUE_PERCENT = 20

OUTPUT_PATCH_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# TISSUE DETECTION
# =========================================================

def calculate_tissue_percent(img_rgb):
    """
    Detect non-background H&E-like tissue.
    Removes white, gray, and black background regions.
    """

    img = np.array(img_rgb)

    # RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)

    # Tissue usually has color saturation.
    # Blank background has very low saturation.
    tissue_mask = (
        (s > 20) &      # enough color
        (v > 40) &      # not black
        (v < 245)       # not pure white
    )

    tissue_percent = tissue_mask.mean() * 100

    return tissue_percent


def is_good_patch(patch_path):
    img = Image.open(patch_path).convert("RGB")
    tissue_percent = calculate_tissue_percent(img)
    return tissue_percent >= MIN_TISSUE_PERCENT, tissue_percent


# =========================================================
# MAIN
# =========================================================

def main():

    metadata = pd.read_csv(INPUT_METADATA)

    clean_records = []
    removed_records = []

    total = len(metadata)
    kept = 0
    removed = 0

    for idx, row in metadata.iterrows():

        patch_path = Path(row["patch_path"])

        if not patch_path.exists():
            removed_records.append({
                **row.to_dict(),
                "filter_status": "missing_file",
                "new_tissue_percent": 0
            })
            removed += 1
            continue

        keep, new_tissue_percent = is_good_patch(patch_path)

        if keep:
            rel_path = patch_path.relative_to(INPUT_PATCH_DIR)
            new_patch_path = OUTPUT_PATCH_DIR / rel_path
            new_patch_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(patch_path, new_patch_path)

            record = row.to_dict()
            record["original_patch_path"] = str(patch_path)
            record["patch_path"] = str(new_patch_path)
            record["new_tissue_percent"] = new_tissue_percent
            record["filter_status"] = "kept"

            clean_records.append(record)
            kept += 1

        else:
            record = row.to_dict()
            record["filter_status"] = "removed_background"
            record["new_tissue_percent"] = new_tissue_percent

            removed_records.append(record)
            removed += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total} | kept={kept} | removed={removed}")

    clean_df = pd.DataFrame(clean_records)
    removed_df = pd.DataFrame(removed_records)

    clean_df.to_csv(OUTPUT_PATCH_DIR / "clean_patch_metadata.csv", index=False)
    removed_df.to_csv(OUTPUT_PATCH_DIR / "removed_patch_metadata.csv", index=False)

    summary = pd.DataFrame([{
        "total_patches": total,
        "kept_patches": kept,
        "removed_patches": removed,
        "min_tissue_percent": MIN_TISSUE_PERCENT
    }])

    summary.to_csv(OUTPUT_PATCH_DIR / "filter_summary.csv", index=False)

    print("\nDONE")
    print(f"Total patches: {total}")
    print(f"Kept patches: {kept}")
    print(f"Removed patches: {removed}")
    print(f"Clean metadata: {OUTPUT_PATCH_DIR / 'clean_patch_metadata.csv'}")


if __name__ == "__main__":
    main()
