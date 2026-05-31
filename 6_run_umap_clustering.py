from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

EMB_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/dinov2_embeddings_downsample_4.0_full")

EMB_PATH = EMB_DIR / "dinov2_embeddings_merged.npy"
META_PATH = EMB_DIR / "dinov2_patch_metadata_merged.csv"

OUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/umap_clustering_downsample_4.0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 8
RANDOM_STATE = 42


# =========================================================
# MAIN
# =========================================================

def main():
    print("Loading embeddings...")
    X = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    print("Embeddings:", X.shape)
    print("Metadata:", meta.shape)

    print("Scaling embeddings...")
    X_scaled = StandardScaler().fit_transform(X)

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE
    )

    umap_xy = reducer.fit_transform(X_scaled)

    meta["umap_x"] = umap_xy[:, 0]
    meta["umap_y"] = umap_xy[:, 1]

    print("Running KMeans...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init="auto"
    )

    clusters = kmeans.fit_predict(X_scaled)
    meta["cluster"] = clusters

    try:
        sil = silhouette_score(X_scaled, clusters, metric="cosine")
    except Exception:
        sil = None

    print(f"Silhouette score: {sil}")

    meta.to_csv(OUT_DIR / "patch_umap_clusters.csv", index=False)

    np.save(OUT_DIR / "umap_xy.npy", umap_xy)
    np.save(OUT_DIR / "cluster_labels.npy", clusters)

    # =====================================================
    # PLOT 1: BY CLUSTER
    # =====================================================

    plt.figure(figsize=(10, 8))
    plt.scatter(
        meta["umap_x"],
        meta["umap_y"],
        c=meta["cluster"],
        s=3,
        alpha=0.7
    )
    plt.title("DINOv2 Patch UMAP Colored by KMeans Cluster")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "umap_by_cluster.png", dpi=300)
    plt.close()

    # =====================================================
    # PLOT 2: BY MAGNIFICATION
    # =====================================================

    plt.figure(figsize=(10, 8))

    for mag in sorted(meta["magnification"].dropna().unique()):
        sub = meta[meta["magnification"] == mag]
        plt.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=3,
            alpha=0.7,
            label=mag
        )

    plt.title("DINOv2 Patch UMAP Colored by Magnification")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "umap_by_magnification.png", dpi=300)
    plt.close()

    # =====================================================
    # CLUSTER SUMMARY
    # =====================================================

    cluster_summary = (
        meta
        .groupby(["cluster", "magnification"])
        .size()
        .reset_index(name="num_patches")
    )

    cluster_summary.to_csv(OUT_DIR / "cluster_summary.csv", index=False)

    print("\nDONE")
    print(f"Saved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
