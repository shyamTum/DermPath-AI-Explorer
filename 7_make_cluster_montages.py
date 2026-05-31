from pathlib import Path
import math
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# CONFIG
# =========================================================

CLUSTER_CSV = Path("/home/ghoshlab/Desktop/Shyam/imaging/umap_clustering_downsample_4.0/patch_umap_clusters.csv")

OUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/cluster_montages_downsample_4.0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE_DISPLAY = 128
PATCHES_PER_CLUSTER = 49   # 7 x 7 grid
GRID_COLS = 7
RANDOM_SEED = 42


# =========================================================
# HELPERS
# =========================================================

def make_montage(df_cluster, cluster_id):
    random.seed(RANDOM_SEED)

    if len(df_cluster) > PATCHES_PER_CLUSTER:
        df_sample = df_cluster.sample(PATCHES_PER_CLUSTER, random_state=RANDOM_SEED)
    else:
        df_sample = df_cluster

    rows = math.ceil(len(df_sample) / GRID_COLS)

    title_height = 45
    tile_w = PATCH_SIZE_DISPLAY
    tile_h = PATCH_SIZE_DISPLAY

    montage_w = GRID_COLS * tile_w
    montage_h = rows * tile_h + title_height

    montage = Image.new("RGB", (montage_w, montage_h), "white")
    draw = ImageDraw.Draw(montage)

    title = f"Cluster {cluster_id} | patches shown: {len(df_sample)} | total patches: {len(df_cluster)}"
    draw.text((10, 10), title, fill="black")

    for i, (_, row) in enumerate(df_sample.iterrows()):
        patch_path = Path(row["patch_path"])

        if not patch_path.exists():
            continue

        img = Image.open(patch_path).convert("RGB")
        img = img.resize((PATCH_SIZE_DISPLAY, PATCH_SIZE_DISPLAY))

        x = (i % GRID_COLS) * tile_w
        y = (i // GRID_COLS) * tile_h + title_height

        montage.paste(img, (x, y))

    out_path = OUT_DIR / f"cluster_{cluster_id}_montage.png"
    montage.save(out_path)

    return out_path


# =========================================================
# MAIN
# =========================================================

def main():
    df = pd.read_csv(CLUSTER_CSV)

    print(f"Total patches: {len(df)}")
    print(f"Clusters found: {sorted(df['cluster'].unique())}")

    summary_rows = []

    for cluster_id in sorted(df["cluster"].unique()):
        df_cluster = df[df["cluster"] == cluster_id]

        out_path = make_montage(df_cluster, cluster_id)

        summary_rows.append({
            "cluster": cluster_id,
            "num_patches": len(df_cluster),
            "montage_path": str(out_path)
        })

        print(f"Saved cluster {cluster_id}: {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "cluster_montage_summary.csv", index=False)

    print("\nDONE")
    print(f"Montages saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
