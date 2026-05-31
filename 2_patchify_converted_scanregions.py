# patchify_converted_scanregions.py

from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tifffile


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/converted_scanregions_new_downsample_4.0")
OUTPUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/patches_from_downsample_4.0")

PATCH_SIZE = 256
STRIDE = 256

# Remove mostly white/background patches
MIN_TISSUE_PERCENT = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def tissue_percentage(patch_rgb):
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)

    # remove white background and black borders
    tissue_mask = (gray > 20) & (gray < 235)

    return tissue_mask.mean() * 100


def read_tiff_rgb(tif_path):
    img = tifffile.imread(str(tif_path))

    img = np.squeeze(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.ndim == 3:
        if img.shape[0] in [3, 4]:
            img = np.moveaxis(img, 0, -1)

        if img.shape[-1] > 3:
            img = img[..., :3]

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.astype(np.uint8)

    return img


def parse_scanregion_name(tif_path):
    """
    Expected example:
    2026_04_20__16__20x__ScanRegion0.tif
    or
    2026_04_20__16__20x__ScanRegion0__ds4.tif
    """

    stem = tif_path.stem

    mag_match = re.search(r"__(10x|20x)__", stem)
    region_match = re.search(r"(ScanRegion\d+)", stem)
    ds_match = re.search(r"__ds(\d+)", stem)

    magnification = mag_match.group(1) if mag_match else tif_path.parent.name
    scanregion = region_match.group(1) if region_match else "ScanRegion_unknown"
    downsample = int(ds_match.group(1)) if ds_match else 4

    czi_base = stem.split(f"__{magnification}__")[0]

    return czi_base, magnification, scanregion, downsample


# =========================================================
# PATCHIFY ONE TIFF
# =========================================================

def patchify_one_tiff(tif_path):
    czi_base, magnification, scanregion, downsample = parse_scanregion_name(tif_path)

    print("\n" + "=" * 80)
    print(f"Processing: {tif_path.name}")
    print(f"CZI base: {czi_base}")
    print(f"Magnification: {magnification}")
    print(f"ScanRegion: {scanregion}")
    print(f"Downsample: {downsample}")

    img = read_tiff_rgb(tif_path)

    h, w, _ = img.shape

    print(f"Image size: {w} x {h}")

    out_dir = OUTPUT_DIR / magnification / czi_base / scanregion
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    total = 0
    saved = 0
    removed = 0

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            total += 1

            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            tissue_pct = tissue_percentage(patch)

            if tissue_pct < MIN_TISSUE_PERCENT:
                removed += 1
                continue

            patch_filename = (
                f"{czi_base}"
                f"__{magnification}"
                f"__{scanregion}"
                f"__ds{downsample}"
                f"__x_{x}"
                f"__y_{y}"
                f"__ps_{PATCH_SIZE}.png"
            )

            patch_path = out_dir / patch_filename

            Image.fromarray(patch).save(patch_path)

            records.append({
                "patch_filename": patch_filename,
                "czi_base": czi_base,
                "magnification": magnification,
                "scanregion": scanregion,
                "downsample": downsample,

                # coordinates in the exported TIFF
                "x": x,
                "y": y,

                # approximate original-level coordinates
                "x_original_estimated": x * downsample,
                "y_original_estimated": y * downsample,

                "patch_size": PATCH_SIZE,
                "stride": STRIDE,
                "scanregion_width": w,
                "scanregion_height": h,
                "source_tiff": str(tif_path),
                "patch_path": str(patch_path),
                "tissue_percent": tissue_pct
            })

            saved += 1

    summary = {
        "source_tiff": str(tif_path),
        "czi_base": czi_base,
        "magnification": magnification,
        "scanregion": scanregion,
        "downsample": downsample,
        "scanregion_width": w,
        "scanregion_height": h,
        "total_patches_checked": total,
        "saved_tissue_patches": saved,
        "removed_background_patches": removed
    }

    print(f"Total checked: {total}")
    print(f"Saved: {saved}")
    print(f"Removed background: {removed}")

    return records, summary


# =========================================================
# MAIN
# =========================================================

def main():
    all_records = []
    all_summaries = []

    tif_files = sorted(list(INPUT_DIR.rglob("*.tif")) + list(INPUT_DIR.rglob("*.tiff")))

    print(f"Found {len(tif_files)} TIFF files")

    for tif_path in tif_files:
        try:
            records, summary = patchify_one_tiff(tif_path)

            all_records.extend(records)
            all_summaries.append(summary)

            pd.DataFrame(all_records).to_csv(
                OUTPUT_DIR / "patch_metadata_running.csv",
                index=False
            )

            pd.DataFrame(all_summaries).to_csv(
                OUTPUT_DIR / "scanregion_patch_summary_running.csv",
                index=False
            )

        except Exception as e:
            print(f"FAILED: {tif_path}")
            print(f"ERROR: {e}")

            all_summaries.append({
                "source_tiff": str(tif_path),
                "status": "failed",
                "error": str(e)
            })

    metadata = pd.DataFrame(all_records)
    summary = pd.DataFrame(all_summaries)

    metadata.to_csv(OUTPUT_DIR / "patch_metadata.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "scanregion_patch_summary.csv", index=False)

    print("\nDONE")
    print(f"Total saved patches: {len(metadata)}")
    print(f"Metadata: {OUTPUT_DIR / 'patch_metadata.csv'}")
    print(f"Summary: {OUTPUT_DIR / 'scanregion_patch_summary.csv'}")


if __name__ == "__main__":
    main()
