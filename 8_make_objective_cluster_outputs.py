from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Allow large TIFF reading
Image.MAX_IMAGE_PIXELS = None


# =========================================================
# PATHS
# =========================================================

CLUSTER_CSV = Path("/home/ghoshlab/Desktop/Shyam/imaging/umap_clustering_downsample_4.0/patch_umap_clusters.csv")

SCANREGION_TIFF_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/converted_scanregions_new_downsample_4.0")

OUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/objective_cluster_outputs_downsample_4.0_preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_INDEX_DIR = OUT_DIR / "cluster_patch_indices"
SOURCE_BREAKDOWN_DIR = OUT_DIR / "cluster_source_breakdowns"
OVERLAY_DIR = OUT_DIR / "scanregion_cluster_overlay_previews"

CLUSTER_INDEX_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_BREAKDOWN_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIG
# =========================================================

MAX_DISPLAY_WIDTH = 3000
OVERLAY_ALPHA = 0.38


# =========================================================
# DYNAMIC COLORS
# =========================================================

def get_dynamic_cluster_colors(cluster_ids):
    cluster_ids = sorted([int(c) for c in cluster_ids])
    cmap = plt.get_cmap("tab20")

    colors = {}

    for i, cluster_id in enumerate(cluster_ids):
        rgb = cmap(i % 20)[:3]
        colors[cluster_id] = tuple(int(v * 255) for v in rgb)

    return colors


# =========================================================
# HELPERS
# =========================================================

def find_source_tiff(row):
    czi_base = row["czi_base"]
    mag = row["magnification"]
    scanregion = row["scanregion"]
    ds = int(row["downsample"])

    pattern = f"{czi_base}__{mag}__{scanregion}*ds{ds}*.tif"
    candidates = list((SCANREGION_TIFF_DIR / mag).glob(pattern))

    if len(candidates) == 0:
        pattern = f"{czi_base}__{mag}__{scanregion}*.tif"
        candidates = list((SCANREGION_TIFF_DIR / mag).glob(pattern))

    return candidates[0] if candidates else None


def make_cluster_tables(df):
    summary_rows = []

    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id].copy()

        patch_index_path = CLUSTER_INDEX_DIR / f"cluster_{cluster_id}_patch_index.csv"
        sub.to_csv(patch_index_path, index=False)

        source_breakdown = (
            sub.groupby(["czi_base", "magnification", "scanregion"])
            .size()
            .reset_index(name="num_patches")
            .sort_values("num_patches", ascending=False)
        )

        source_breakdown_path = SOURCE_BREAKDOWN_DIR / f"cluster_{cluster_id}_source_breakdown.csv"
        source_breakdown.to_csv(source_breakdown_path, index=False)

        summary_rows.append({
            "cluster": int(cluster_id),
            "num_patches": len(sub),
            "num_czi_images": sub["czi_base"].nunique(),
            "num_scanregions": sub[["czi_base", "magnification", "scanregion"]].drop_duplicates().shape[0],
            "patch_index_csv": str(patch_index_path),
            "source_breakdown_csv": str(source_breakdown_path)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "cluster_objective_summary.csv", index=False)

    return summary_df


def resize_for_display(img, max_width=3000):
    w, h = img.size

    if w <= max_width:
        return img, 1.0

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_small = img.resize((new_w, new_h), Image.BILINEAR)

    return img_small, scale


def create_overlay_preview(df_scan, source_tiff, cluster_colors):

    base = Image.open(source_tiff).convert("RGB")
    original_w, original_h = base.size

    base_small, scale = resize_for_display(base, MAX_DISPLAY_WIDTH)
    np_base = np.array(base_small).astype(np.uint8)

    h_img, w_img, _ = np_base.shape

    # Full overlay RGB image
    color_layer = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # Full alpha mask
    alpha_layer = np.zeros((h_img, w_img), dtype=np.float32)

    for _, row in df_scan.iterrows():

        x = int(row["x"] * scale)
        y = int(row["y"] * scale)
        ps = max(1, int(row["patch_size"] * scale))
        cluster = int(row["cluster"])

        x2 = min(x + ps, w_img)
        y2 = min(y + ps, h_img)

        if x >= w_img or y >= h_img or x2 <= x or y2 <= y:
            continue

        patch = np_base[y:y2, x:x2]

        # -----------------------------
        # Tissue mask inside this patch
        # -----------------------------
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        tissue_mask = (
            (s > 18) &      # colored tissue
            (v > 35) &      # not black
            (v < 245)       # not white background
        ).astype(np.float32)

        if tissue_mask.mean() < 0.15:
            continue

        # Smooth mask edges
        tissue_mask = cv2.GaussianBlur(tissue_mask, (0, 0), sigmaX=2)

        color = np.array(cluster_colors.get(cluster, (255, 255, 255)), dtype=np.uint8)

        color_layer[y:y2, x:x2, :] = color

        # Keep strongest alpha if overlapping
        alpha_patch = tissue_mask * OVERLAY_ALPHA
        alpha_layer[y:y2, x:x2] = np.maximum(
            alpha_layer[y:y2, x:x2],
            alpha_patch
        )

    # -----------------------------
    # Blend overlay with original
    # -----------------------------
    alpha_3d = alpha_layer[..., None]

    blended = (
        np_base.astype(np.float32) * (1 - alpha_3d)
        + color_layer.astype(np.float32) * alpha_3d
    )

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    result = Image.fromarray(blended)

    # -----------------------------
    # Crop to tissue bounding region
    # -----------------------------
    gray = cv2.cvtColor(np_base, cv2.COLOR_RGB2GRAY)
    tissue_global = gray < 245

    ys, xs = np.where(tissue_global)

    if len(xs) > 0 and len(ys) > 0:
        margin = 80
        crop_x1 = max(xs.min() - margin, 0)
        crop_y1 = max(ys.min() - margin, 0)
        crop_x2 = min(xs.max() + margin, w_img)
        crop_y2 = min(ys.max() + margin, h_img)

        result = result.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return (
        result,
        original_w,
        original_h,
        result.size[0],
        result.size[1],
        scale
    )


def make_scanregion_overlays(df, cluster_colors):
    overlay_rows = []

    group_cols = ["czi_base", "magnification", "scanregion", "downsample"]

    for (czi_base, mag, scanregion, ds), df_scan in df.groupby(group_cols):

        source_tiff = find_source_tiff(df_scan.iloc[0])

        if source_tiff is None:
            print(f"Missing TIFF: {czi_base} {mag} {scanregion}")
            continue

        print(f"Creating preview overlay: {czi_base} | {mag} | {scanregion}")

        try:
            overlay_img, ow, oh, dw, dh, scale = create_overlay_preview(
                df_scan=df_scan,
                source_tiff=source_tiff,
                cluster_colors=cluster_colors
            )

            out_subdir = OVERLAY_DIR / mag / czi_base
            out_subdir.mkdir(parents=True, exist_ok=True)

            out_name = f"{czi_base}__{mag}__{scanregion}__ds{int(ds)}__cluster_overlay_preview.png"
            out_path = out_subdir / out_name

            overlay_img.save(out_path)

            overlay_rows.append({
                "czi_base": czi_base,
                "magnification": mag,
                "scanregion": scanregion,
                "downsample": int(ds),
                "num_patches": len(df_scan),
                "source_tiff": str(source_tiff),
                "overlay_path": str(out_path),
                "original_width": ow,
                "original_height": oh,
                "display_width": dw,
                "display_height": dh,
                "display_scale": scale,
                "status": "created",
                "error": ""
            })

        except Exception as e:
            print(f"FAILED overlay: {czi_base} {mag} {scanregion}")
            print(f"ERROR: {e}")

            overlay_rows.append({
                "czi_base": czi_base,
                "magnification": mag,
                "scanregion": scanregion,
                "downsample": int(ds),
                "num_patches": len(df_scan),
                "source_tiff": str(source_tiff),
                "overlay_path": "",
                "original_width": "",
                "original_height": "",
                "display_width": "",
                "display_height": "",
                "display_scale": "",
                "status": "failed",
                "error": str(e)
            })

    overlay_df = pd.DataFrame(overlay_rows)
    overlay_df.to_csv(OUT_DIR / "scanregion_overlay_index.csv", index=False)

    return overlay_df


def save_color_legend(cluster_colors):
    rows = []

    for cluster_id, rgb in cluster_colors.items():
        rows.append({
            "cluster": cluster_id,
            "R": rgb[0],
            "G": rgb[1],
            "B": rgb[2],
            "note": "Color is visualization-only; cluster labels are unsupervised visual groups."
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "cluster_color_legend.csv", index=False)


# =========================================================
# MAIN
# =========================================================

def main():
    df = pd.read_csv(CLUSTER_CSV)
    df["cluster"] = df["cluster"].astype(int)

    cluster_ids = sorted(df["cluster"].unique())
    cluster_colors = get_dynamic_cluster_colors(cluster_ids)

    print(f"Loaded patches: {len(df)}")
    print(f"Detected clusters: {cluster_ids}")
    print("Cluster colors are generated dynamically and are visualization-only.")

    save_color_legend(cluster_colors)

    make_cluster_tables(df)

    make_scanregion_overlays(df, cluster_colors)

    print("\nDONE")
    print(f"Output folder: {OUT_DIR}")
    print(f"Cluster summary: {OUT_DIR / 'cluster_objective_summary.csv'}")
    print(f"Overlay index: {OUT_DIR / 'scanregion_overlay_index.csv'}")
    print(f"Color legend: {OUT_DIR / 'cluster_color_legend.csv'}")


if __name__ == "__main__":
    main()
