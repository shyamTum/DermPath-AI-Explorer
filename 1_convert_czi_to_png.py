# patchify_single_czi_regionwise.py

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from aicspylibczi import CziFile


CZI_PATH = Path("/home/ghoshlab/Desktop/Shyam/imaging/2026-04-20-Axioscan/10x/2026_04_20__02.czi")
MAGNIFICATION = "10x"
OUTPUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/patches_single_test")

PATCH_SIZE = 256
STRIDE = 256
MIN_TISSUE_PERCENT = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr.astype(np.uint8)


def convert_to_rgb(arr):
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = normalize_to_uint8(arr)
        return np.stack([arr] * 3, axis=-1)

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[0] in [1, 3, 4]:
            arr = np.moveaxis(arr, 0, -1)

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if arr.shape[-1] > 3:
            arr = arr[..., :3]

        arr = normalize_to_uint8(arr)

    return arr


def tissue_percentage(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Tissue should be neither too white nor completely black
    tissue_mask = (gray > 20) & (gray < 220)

    return tissue_mask.mean() * 100


def main():
    image_name = CZI_PATH.stem
    out_dir = OUTPUT_DIR / MAGNIFICATION / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    czi = CziFile(CZI_PATH)

    bbox = czi.get_mosaic_bounding_box()
    x0, y0, w, h = bbox.x, bbox.y, bbox.w, bbox.h

    print(f"CZI: {CZI_PATH.name}")
    print(f"Dims: {czi.dims}")
    print(f"Is mosaic: {czi.is_mosaic()}")
    print(f"Bounding box x={x0}, y={y0}, w={w}, h={h}")

    records = []
    total = 0
    saved = 0
    removed = 0

    for y in range(y0, y0 + h - PATCH_SIZE + 1, STRIDE):
        for x in range(x0, x0 + w - PATCH_SIZE + 1, STRIDE):

            total += 1

            try:
                arr = czi.read_mosaic(
                    C=0,
                    region=(x, y, PATCH_SIZE, PATCH_SIZE)
                )
            except Exception:
                continue

            patch = convert_to_rgb(arr)

            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                continue

            tissue_pct = tissue_percentage(patch)

            if tissue_pct < MIN_TISSUE_PERCENT:
                removed += 1
                continue

            # Store local coordinates relative to bounding box
            local_x = x - x0
            local_y = y - y0

            patch_filename = (
                f"{image_name}"
                f"__mag_{MAGNIFICATION}"
                f"__x_{local_x}"
                f"__y_{local_y}"
                f"__absx_{x}"
                f"__absy_{y}"
                f"__ps_{PATCH_SIZE}.png"
            )

            patch_path = out_dir / patch_filename
            Image.fromarray(patch).save(patch_path)

            records.append({
                "patch_filename": patch_filename,
                "original_czi": image_name,
                "magnification": MAGNIFICATION,
                "x": local_x,
                "y": local_y,
                "abs_x": x,
                "abs_y": y,
                "bbox_x0": x0,
                "bbox_y0": y0,
                "patch_size": PATCH_SIZE,
                "stride": STRIDE,
                "image_width": w,
                "image_height": h,
                "patch_path": str(patch_path),
                "tissue_percent": tissue_pct
            })

            saved += 1

        print(f"Processed row y={y - y0}/{h}, saved={saved}, removed={removed}")

    metadata = pd.DataFrame(records)
    metadata_path = OUTPUT_DIR / f"{image_name}__patch_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    summary = pd.DataFrame([{
        "original_czi": image_name,
        "magnification": MAGNIFICATION,
        "bbox_x0": x0,
        "bbox_y0": y0,
        "image_width": w,
        "image_height": h,
        "total_patches_checked": total,
        "saved_tissue_patches": saved,
        "removed_background_patches": removed,
        "metadata_path": str(metadata_path)
    }])

    summary_path = OUTPUT_DIR / f"{image_name}__summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nDONE")
    print(f"Total checked: {total}")
    print(f"Saved: {saved}")
    print(f"Removed background: {removed}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
