from pathlib import Path
import numpy as np
import pandas as pd


CHUNK_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/dinov2_embeddings_downsample_4.0_full")

OUT_EMB = CHUNK_DIR / "dinov2_embeddings_merged.npy"
OUT_META = CHUNK_DIR / "dinov2_patch_metadata_merged.csv"


def main():
    emb_files = sorted(CHUNK_DIR.glob("embeddings_chunk_*.npy"))
    meta_files = sorted(CHUNK_DIR.glob("metadata_chunk_*.csv"))

    print(f"Embedding chunks found: {len(emb_files)}")
    print(f"Metadata chunks found: {len(meta_files)}")

    all_embs = []
    all_meta = []

    for emb_file, meta_file in zip(emb_files, meta_files):
        print(f"Loading {emb_file.name}")

        emb = np.load(emb_file)
        meta = pd.read_csv(meta_file)

        if len(emb) != len(meta):
            raise ValueError(f"Mismatch in {emb_file.name}: emb={len(emb)}, meta={len(meta)}")

        all_embs.append(emb)
        all_meta.append(meta)

    merged_emb = np.concatenate(all_embs, axis=0)
    merged_meta = pd.concat(all_meta, ignore_index=True)

    np.save(OUT_EMB, merged_emb)
    merged_meta.to_csv(OUT_META, index=False)

    print("\nDONE")
    print(f"Merged embedding shape: {merged_emb.shape}")
    print(f"Saved embeddings: {OUT_EMB}")
    print(f"Saved metadata: {OUT_META}")


if __name__ == "__main__":
    main()
