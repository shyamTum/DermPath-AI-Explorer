# extract_dinov2_embeddings_resume.py

from pathlib import Path
import gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# =========================================================
# CONFIG
# =========================================================

CLEAN_PATCH_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/clean_patches_from_downsample_4.0")
METADATA_CSV = CLEAN_PATCH_DIR / "clean_patch_metadata.csv"

OUTPUT_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging/dinov2_embeddings_downsample_4.0_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "dinov2_vits14"

BATCH_SIZE = 2
NUM_WORKERS = 0

CHUNK_SIZE = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(4)


# =========================================================
# DATASET
# =========================================================

class PatchDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["patch_path"]).convert("RGB")
        return self.transform(img), idx


# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Using device: {DEVICE}")

    metadata = pd.read_csv(METADATA_CSV).reset_index(drop=True)
    metadata["global_index"] = metadata.index

    total_patches = len(metadata)
    num_chunks = int(np.ceil(total_patches / CHUNK_SIZE))

    metadata.to_csv(OUTPUT_DIR / "all_patch_metadata_with_global_index.csv", index=False)

    print(f"Total patches: {total_patches}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total chunks: {num_chunks}")

    print(f"Loading model: {MODEL_NAME}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()

    for chunk_id in range(num_chunks):

        start = chunk_id * CHUNK_SIZE
        end = min((chunk_id + 1) * CHUNK_SIZE, total_patches)

        emb_path = OUTPUT_DIR / f"embeddings_chunk_{chunk_id:04d}.npy"
        meta_path = OUTPUT_DIR / f"metadata_chunk_{chunk_id:04d}.csv"

        if emb_path.exists() and meta_path.exists():
            print(f"Skipping existing chunk {chunk_id}: {start}-{end}")
            continue

        print("\n" + "=" * 80)
        print(f"Processing chunk {chunk_id}/{num_chunks - 1}: rows {start} to {end - 1}")

        chunk_df = metadata.iloc[start:end].reset_index(drop=True)

        dataset = PatchDataset(chunk_df)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        chunk_embeddings = []
        chunk_indices = []

        with torch.no_grad():
            for imgs, indices in tqdm(dataloader, desc=f"Chunk {chunk_id}"):

                imgs = imgs.to(DEVICE)

                feats = model(imgs)

                feats = feats.detach().cpu().numpy()

                chunk_embeddings.append(feats)
                chunk_indices.extend(indices.numpy().tolist())

        chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)

        ordered_chunk_metadata = chunk_df.iloc[chunk_indices].reset_index(drop=True)

        np.save(emb_path, chunk_embeddings)
        ordered_chunk_metadata.to_csv(meta_path, index=False)

        print(f"Saved: {emb_path}")
        print(f"Shape: {chunk_embeddings.shape}")

        del dataset, dataloader, chunk_embeddings, chunk_indices
        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\nDONE")
    print(f"All chunks saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
