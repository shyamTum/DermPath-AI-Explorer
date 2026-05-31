from pathlib import Path
import base64
import io
import hashlib
import requests

import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/imaging")

CLUSTER_CSV = BASE_DIR / "umap_clustering_downsample_4.0" / "patch_umap_clusters.csv"
EMB_PATH = BASE_DIR / "dinov2_embeddings_downsample_4.0_full" / "dinov2_embeddings_merged.npy"

MONTAGE_DIR = BASE_DIR / "cluster_montages_downsample_4.0"

OBJECTIVE_DIR = BASE_DIR / "objective_cluster_outputs_downsample_4.0_preview"
OVERLAY_INDEX_CSV = OBJECTIVE_DIR / "scanregion_overlay_index.csv"
CLUSTER_SUMMARY_CSV = OBJECTIVE_DIR / "cluster_objective_summary.csv"

UMAP_CLUSTER_PNG = BASE_DIR / "umap_clustering_downsample_4.0" / "umap_by_cluster.png"
UMAP_MAG_PNG = BASE_DIR / "umap_clustering_downsample_4.0" / "umap_by_magnification.png"


# =========================================================
# OLLAMA CONFIG
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CLUSTER_CSV)
embeddings = np.load(EMB_PATH)

overlay_df = pd.read_csv(OVERLAY_INDEX_CSV)
cluster_summary = pd.read_csv(CLUSTER_SUMMARY_CSV)

df["cluster"] = df["cluster"].astype(int)
cluster_summary["cluster"] = cluster_summary["cluster"].astype(int)

clusters = [int(x) for x in sorted(df["cluster"].unique())]
magnifications = [str(x) for x in sorted(df["magnification"].dropna().unique())]


# =========================================================
# DATASET SUMMARY
# =========================================================

def dataset_overview_text():
    rows = []

    total_czi = int(df["czi_base"].nunique())
    total_scanregions = int(
        df[["czi_base", "magnification", "scanregion"]]
        .drop_duplicates()
        .shape[0]
    )
    total_patches = int(len(df))
    total_clusters = int(df["cluster"].nunique())

    rows.append("# Dataset + Pipeline Overview\n")
    rows.append("### What this prototype does\n")
    rows.append(
        "CZI images → QuPath ScanRegions → 256×256 patches → background filtering → "
        "DINOv2 feature extraction → UMAP visualization → KMeans morphology clustering → "
        "similar patch retrieval → local VLM morphology description.\n"
    )

    rows.append("### Overall dataset\n")
    rows.append(f"- Total CZI images represented: **{total_czi}**")
    rows.append(f"- Total ScanRegions represented: **{total_scanregions}**")
    rows.append(f"- Total clean patches analyzed: **{total_patches}**")
    rows.append(f"- Global morphology clusters: **{total_clusters}**\n")

    rows.append("### By magnification\n")

    for mag in magnifications:
        sub = df[df["magnification"].astype(str) == str(mag)]
        czi_count = int(sub["czi_base"].nunique())
        scan_count = int(
            sub[["czi_base", "magnification", "scanregion"]]
            .drop_duplicates()
            .shape[0]
        )
        patch_count = int(len(sub))
        avg_scan = round(scan_count / max(czi_count, 1), 2)

        rows.append(f"#### {mag}")
        rows.append(f"- CZI images: **{czi_count}**")
        rows.append(f"- ScanRegions: **{scan_count}**")
        rows.append(f"- Approx. ScanRegions per CZI: **{avg_scan}**")
        rows.append(f"- Clean patches: **{patch_count}**\n")

    rows.append("### Important interpretation note\n")
    rows.append(
        "Cluster IDs are **global unsupervised visual morphology groups** computed across "
        "all clean patches from all CZI images and ScanRegions."
    )

    return "\n".join(rows)


# =========================================================
# BASIC HELPERS
# =========================================================

def get_czi_choices(magnification):
    sub = overlay_df[overlay_df["magnification"].astype(str) == str(magnification)]
    return [str(x) for x in sorted(sub["czi_base"].dropna().unique())]


def get_scanregion_choices(magnification, czi_base):
    sub = overlay_df[
        (overlay_df["magnification"].astype(str) == str(magnification)) &
        (overlay_df["czi_base"].astype(str) == str(czi_base))
    ]
    return [str(x) for x in sorted(sub["scanregion"].dropna().unique())]


def show_scanregion_overlay(magnification, czi_base, scanregion):
    sub = overlay_df[
        (overlay_df["magnification"].astype(str) == str(magnification)) &
        (overlay_df["czi_base"].astype(str) == str(czi_base)) &
        (overlay_df["scanregion"].astype(str) == str(scanregion))
    ]

    if len(sub) == 0:
        return None, "No overlay found."

    row = sub.iloc[0]
    overlay_path = str(row["overlay_path"])

    info = f"""
### Selected ScanRegion

**CZI image:** {czi_base}  
**Magnification:** {magnification}  
**ScanRegion:** {scanregion}  
**Clustered patches shown:** {int(row["num_patches"])}  

This is an unsupervised morphology overlay. Each color corresponds to a global morphology cluster.
"""

    return overlay_path, info


def show_cluster(cluster_id):
    cluster_id = int(cluster_id)
    montage_path = str(MONTAGE_DIR / f"cluster_{cluster_id}_montage.png")

    summary_row = cluster_summary[cluster_summary["cluster"] == cluster_id]

    if len(summary_row) > 0:
        r = summary_row.iloc[0]
        summary_text = f"""
### Global Morphology Cluster {cluster_id}

**Total patches:** {int(r["num_patches"])}  
**CZI images represented:** {int(r["num_czi_images"])}  
**ScanRegions represented:** {int(r["num_scanregions"])}  

This cluster was computed globally across all clean patches. It is an unsupervised visual group, not a diagnostic class.
"""
    else:
        summary_text = f"Cluster {cluster_id} summary not found."

    source_csv = OBJECTIVE_DIR / "cluster_source_breakdowns" / f"cluster_{cluster_id}_source_breakdown.csv"

    if source_csv.exists():
        source_df = pd.read_csv(source_csv).head(25)
    else:
        source_df = pd.DataFrame()

    return montage_path, summary_text, source_df


def get_patch_choices(cluster_id):
    cluster_id = int(cluster_id)
    sub = df[df["cluster"] == cluster_id].head(500)

    choices = []
    for _, row in sub.iterrows():
        label = (
            f'{str(row["patch_filename"])} | '
            f'CZI={str(row["czi_base"])} | '
            f'Mag={str(row["magnification"])} | '
            f'ScanRegion={str(row["scanregion"])} | '
            f'x={int(row["x"])}, y={int(row["y"])}'
        )
        choices.append(label)

    return choices


def parse_patch_label(label):
    patch_filename = str(label).split(" | ")[0]
    row = df[df["patch_filename"].astype(str) == patch_filename].iloc[0]
    return row


def show_patch(label):
    if label is None or label == "":
        return None, "No patch selected."

    row = parse_patch_label(label)
    patch_path = str(row["patch_path"])

    tissue_value = row.get("new_tissue_percent", row.get("tissue_percent", "NA"))

    info = f"""
### Selected Patch

**Patch:** {row["patch_filename"]}  
**CZI image:** {row["czi_base"]}  
**Magnification:** {row["magnification"]}  
**ScanRegion:** {row["scanregion"]}  
**Global morphology cluster:** {int(row["cluster"])}  
**Coordinate in ScanRegion:** x={int(row["x"])}, y={int(row["y"])}  
**Patch size:** {int(row["patch_size"])}  
**Tissue percent:** {tissue_value}  
"""

    return patch_path, info


def similar_patches(label, top_k):
    if label is None or label == "":
        return [], pd.DataFrame()

    row = parse_patch_label(label)
    patch_filename = str(row["patch_filename"])

    idx = int(df.index[df["patch_filename"].astype(str) == patch_filename][0])

    query = embeddings[idx:idx + 1]
    sims = cosine_similarity(query, embeddings)[0]

    ranked = np.argsort(-sims)
    ranked = [int(i) for i in ranked if int(i) != idx][:int(top_k)]

    gallery_items = []
    rows = []

    for rank, i in enumerate(ranked, start=1):
        r = df.iloc[i]

        caption = f"Rank {rank} | Cluster {int(r['cluster'])} | Similarity {float(sims[i]):.3f}"
        gallery_items.append((str(r["patch_path"]), caption))

        rows.append({
            "rank": int(rank),
            "similarity": round(float(sims[i]), 4),
            "patch_filename": str(r["patch_filename"]),
            "czi_base": str(r["czi_base"]),
            "magnification": str(r["magnification"]),
            "scanregion": str(r["scanregion"]),
            "cluster": int(r["cluster"]),
            "x": int(r["x"]),
            "y": int(r["y"]),
            "patch_path": str(r["patch_path"])
        })

    return gallery_items, pd.DataFrame(rows)


# =========================================================
# OLLAMA VLM HELPERS
# =========================================================

def image_to_base64(image_path, max_size=768):
    img = Image.open(str(image_path)).convert("RGB")

    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)

    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_ollama_vlm(image_path, question, context_text):
    image_b64 = image_to_base64(image_path)

    prompt = f"""
You are assisting with a research prototype for H&E-stained skin histology images.

Rules:
- Do not diagnose disease.
- Do not claim cancer, malignancy, inflammation, biomarker, prognosis, subtype, or grade.
- Describe only visible morphology and image quality.
- Use cautious wording such as "appears", "may show", "visually suggests".
- Focus on tissue density, nuclear/cellular density, H&E color patterns, texture, tissue edges, artifact, or background.
- Mention that clinician/pathologist review is required.

Context:
{context_text}

Question:
{question}

Answer concisely in clinician-friendly language.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "num_predict": 180,
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        return response.json().get("response", "No response returned from Ollama.")

    except Exception as e:
        return (
            "### Ollama/LLaVA call failed\n\n"
            f"Error:\n\n{str(e)}\n\n"
            "Check that Ollama is running and LLaVA is pulled:\n\n"
            "`ollama pull llava`\n"
        )


def get_ai_target_image(target_type, patch_label, cluster_id, magnification, czi_base, scanregion):
    if target_type == "Selected patch":
        if not patch_label:
            return None, "No patch selected."

        row = parse_patch_label(patch_label)
        image_path = str(row["patch_path"])

        context = f"""
Target type: selected H&E patch
CZI image: {row["czi_base"]}
Magnification: {row["magnification"]}
ScanRegion: {row["scanregion"]}
Global morphology cluster: {int(row["cluster"])}
Patch coordinates: x={int(row["x"])}, y={int(row["y"])}
Patch size: {int(row["patch_size"])}
"""

        return image_path, context

    if target_type == "Cluster montage":
        cluster_id = int(cluster_id)
        image_path = str(MONTAGE_DIR / f"cluster_{cluster_id}_montage.png")

        summary_row = cluster_summary[cluster_summary["cluster"] == cluster_id]

        if len(summary_row) > 0:
            r = summary_row.iloc[0]
            context = f"""
Target type: cluster montage
Global morphology cluster: {cluster_id}
Total patches in cluster: {int(r["num_patches"])}
CZI images represented: {int(r["num_czi_images"])}
ScanRegions represented: {int(r["num_scanregions"])}
"""
        else:
            context = f"Target type: cluster montage\nGlobal morphology cluster: {cluster_id}"

        return image_path, context

    if target_type == "ScanRegion overlay":
        sub = overlay_df[
            (overlay_df["magnification"].astype(str) == str(magnification)) &
            (overlay_df["czi_base"].astype(str) == str(czi_base)) &
            (overlay_df["scanregion"].astype(str) == str(scanregion))
        ]

        if len(sub) == 0:
            return None, "No ScanRegion overlay found."

        row = sub.iloc[0]
        image_path = str(row["overlay_path"])

        context = f"""
Target type: ScanRegion overlay
CZI image: {czi_base}
Magnification: {magnification}
ScanRegion: {scanregion}
Number of clustered patches: {int(row["num_patches"])}
This is an unsupervised morphology overlay.
"""

        return image_path, context

    return None, "Unknown target type."


def ai_answer(target_type, question, patch_label, cluster_id, magnification, czi_base, scanregion):
    if question is None or question.strip() == "":
        return None, "Please type a question."

    image_path, context = get_ai_target_image(
        target_type,
        patch_label,
        cluster_id,
        magnification,
        czi_base,
        scanregion
    )

    if image_path is None:
        return None, context

    answer = call_ollama_vlm(image_path, question, context)
    return image_path, answer




# =========================================================
# DEMO CLASSIFICATION HELPERS
# =========================================================

DEMO_CLASSES = ["Hidradenitis", "Epidermoid cyst"]

CLASS_INFO = {
    "Hidradenitis": {
        "summary": "Demo output class: Hidradenitis. In a future trained model, this class would represent inflammatory skin-disease-associated histopathology patterns.",
        "features": [
            "ROI-level tissue organization and morphology-rich regions",
            "possible dense cellular/inflammatory-appearing areas depending on the selected image",
            "heterogeneous tissue texture that can be linked with downstream clinical and molecular metadata"
        ],
        "clinical_use": "Potential AI-assisted outputs could include ROI prioritization, disease-pattern scoring, morphology summary, and linkage with clinical notes or biomarker profiles."
    },
    "Epidermoid cyst": {
        "summary": "Demo output class: Epidermoid cyst. In a future trained model, this class would represent cyst-associated epithelial/keratin-pattern morphology.",
        "features": [
            "epithelial/keratin-like morphology patterns when visible",
            "localized ROI-level tissue organization",
            "image-derived morphology features suitable for similar-case retrieval and report support"
        ],
        "clinical_use": "Potential AI-assisted outputs could include lesion-pattern classification, ROI highlighting, morphology explanation, and comparison with visually similar cases."
    }
}


def demo_predict_class(image_path):
    """Deterministic placeholder classifier for grant/demo only.

    This does NOT use a trained disease classifier. It uses the image bytes to choose
    one of two demo classes consistently so the app can demonstrate the planned UI.
    """
    if image_path is None:
        return None, "Please upload or select an image first."

    image_path = str(image_path)

    try:
        with open(image_path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        seed_value = int(digest[:8], 16)
    except Exception:
        seed_value = 0

    class_name = DEMO_CLASSES[seed_value % len(DEMO_CLASSES)]
    confidence = 0.76 + ((seed_value % 18) / 100.0)  # 0.76 to 0.93 demo confidence
    other_class = [c for c in DEMO_CLASSES if c != class_name][0]
    other_conf = 1.0 - confidence

    info = CLASS_INFO[class_name]

    feature_lines = "\n".join([f"- {x}" for x in info["features"]])

    output = f"""
### Demo Disease-Class Prediction

**Predicted class:** **{class_name}**  
**Demo confidence:** **{confidence:.2f}**  
**Alternative class:** {other_class} ({other_conf:.2f})

### Class-specific information
{info["summary"]}

### Image/ROI information this module would provide
{feature_lines}

### Potential clinical/research implication
{info["clinical_use"]}

### Important note
This is a **demonstration placeholder**, not a trained diagnostic classifier. In the grant/full version, this tab can be replaced by a supervised model trained on labeled H&E images for classes such as Hidradenitis and Epidermoid cysts, and evaluated using accuracy, F1-score, ROC-AUC, and expert review.
"""

    return image_path, output


# =========================================================
# GRADIO APP
# =========================================================

with gr.Blocks(title="Skin Histology AI Explorer") as demo:

    gr.Markdown("# Skin Histology AI Explorer")
    gr.Markdown(
        "Prototype for unsupervised morphology exploration of unlabeled H&E skin histology images. "
        "This tool is for research visualization."
    )

    default_mag = magnifications[0] if magnifications else None
    initial_czi_choices = get_czi_choices(default_mag)
    default_czi = initial_czi_choices[0] if initial_czi_choices else None
    initial_scan_choices = get_scanregion_choices(default_mag, default_czi)
    default_scan = initial_scan_choices[0] if initial_scan_choices else None

    with gr.Tab("1. Dataset + Pipeline Overview"):
        gr.Markdown(dataset_overview_text())

    with gr.Tab("2. ScanRegion Morphology Overlay"):
        gr.Markdown(
            "This tab shows one selected ScanRegion with global unsupervised cluster colors overlaid on tissue patches."
        )

        mag_dd = gr.Dropdown(choices=magnifications, label="Magnification", value=default_mag)
        czi_dd = gr.Dropdown(choices=initial_czi_choices, value=default_czi, label="CZI image")
        scan_dd = gr.Dropdown(choices=initial_scan_choices, value=default_scan, label="ScanRegion")

        overlay_img = gr.Image(label="Cluster overlay", type="filepath")
        overlay_info = gr.Markdown()

        def update_czi(mag):
            choices = get_czi_choices(mag)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def update_scan(mag, czi):
            choices = get_scanregion_choices(mag, czi)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        mag_dd.change(update_czi, inputs=mag_dd, outputs=czi_dd)
        czi_dd.change(update_scan, inputs=[mag_dd, czi_dd], outputs=scan_dd)

        gr.Button("Show ScanRegion Overlay").click(
            show_scanregion_overlay,
            inputs=[mag_dd, czi_dd, scan_dd],
            outputs=[overlay_img, overlay_info]
        )

    with gr.Tab("3. Global Morphology Cluster Explorer"):
        gr.Markdown(
            "This tab explores global morphology clusters computed across all clean patches from all CZI images and ScanRegions."
        )

        cluster_dd = gr.Dropdown(
            choices=clusters,
            label="Global morphology cluster ID",
            value=clusters[0] if clusters else None
        )

        montage_img = gr.Image(label="Representative cluster montage", type="filepath")
        cluster_info = gr.Markdown()
        source_table = gr.Dataframe(label="Top CZI/ScanRegion contributors")

        gr.Button("Load Cluster").click(
            show_cluster,
            inputs=cluster_dd,
            outputs=[montage_img, cluster_info, source_table]
        )

    with gr.Tab("4. Patch Explorer + Similar Morphology Search"):
        gr.Markdown(
            "This tab filters patches by global cluster, shows patch provenance, and retrieves visually similar patches using DINOv2 embeddings."
        )

        patch_cluster_dd = gr.Dropdown(
            choices=clusters,
            label="Filter patches by global morphology cluster",
            value=clusters[0] if clusters else None
        )

        patch_dd = gr.Dropdown(label="Patch")
        patch_img = gr.Image(label="Selected patch", type="filepath")
        patch_info = gr.Markdown()

        top_k = gr.Slider(3, 20, value=8, step=1, label="Number of similar patches")
        sim_gallery = gr.Gallery(label="Similar morphology patches", columns=4, height=500)
        sim_table = gr.Dataframe(label="Similar patch metadata")

        def update_patch_dropdown(cluster_id):
            choices = get_patch_choices(cluster_id)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        patch_cluster_dd.change(update_patch_dropdown, inputs=patch_cluster_dd, outputs=patch_dd)

        gr.Button("Show Patch").click(
            show_patch,
            inputs=patch_dd,
            outputs=[patch_img, patch_info]
        )

        gr.Button("Find Similar Patches").click(
            similar_patches,
            inputs=[patch_dd, top_k],
            outputs=[sim_gallery, sim_table]
        )

    with gr.Tab("5. UMAP Morphology Landscape"):
        gr.Markdown(
            "This tab shows the 2D UMAP projection of DINOv2 patch embeddings. Each point is one patch."
        )

        gr.Image(value=str(UMAP_CLUSTER_PNG), label="UMAP colored by global cluster", type="filepath")
        gr.Image(value=str(UMAP_MAG_PNG), label="UMAP colored by magnification", type="filepath")


    with gr.Tab("6. Demo Disease-Class Prediction"):
        gr.Markdown(
            "This tab demonstrates how a future supervised classification module will look. "
            "For now, it returns a deterministic placeholder class for grant/demo visualization."
        )

        demo_cls_img_in = gr.Image(
            label="Upload H&E patch / ScanRegion / ROI image",
            type="filepath"
        )
        demo_cls_img_out = gr.Image(
            label="Image used for demo prediction",
            type="filepath"
        )
        demo_cls_output = gr.Markdown()

        gr.Button("Run Demo Class Prediction").click(
            demo_predict_class,
            inputs=demo_cls_img_in,
            outputs=[demo_cls_img_out, demo_cls_output]
        )

    with gr.Tab("7. AI Morphology Assistant"):
        gr.Markdown(
            """
This tab uses local Ollama + LLaVA to generate cautious morphology-only descriptions.

"""
        )

        ai_target = gr.Dropdown(
            choices=["Selected patch", "Cluster montage", "ScanRegion overlay"],
            value="Selected patch",
            label="AI target"
        )

        # ---------- Patch controls ----------
        gr.Markdown("### Patch selection")
        ai_patch_cluster_dd = gr.Dropdown(
            choices=clusters,
            label="Filter patches by global morphology cluster",
            value=clusters[0] if clusters else None
        )

        ai_patch_dd = gr.Dropdown(
            label="Patch"
        )

        def update_ai_patch_dropdown(cluster_id):
            choices = get_patch_choices(cluster_id)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        ai_patch_cluster_dd.change(
            update_ai_patch_dropdown,
            inputs=ai_patch_cluster_dd,
            outputs=ai_patch_dd
        )

        # ---------- Cluster montage controls ----------
        gr.Markdown("### Cluster montage selection")
        ai_montage_cluster_dd = gr.Dropdown(
            choices=clusters,
            label="Global morphology cluster ID for montage",
            value=clusters[0] if clusters else None
        )

        # ---------- ScanRegion overlay controls ----------
        gr.Markdown("### ScanRegion overlay selection")
        ai_mag_dd = gr.Dropdown(
            choices=magnifications,
            label="Magnification",
            value=default_mag
        )

        ai_czi_dd = gr.Dropdown(
            choices=initial_czi_choices,
            value=default_czi,
            label="CZI image"
        )

        ai_scan_dd = gr.Dropdown(
            choices=initial_scan_choices,
            value=default_scan,
            label="ScanRegion"
        )

        def update_ai_czi(mag):
            choices = get_czi_choices(mag)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def update_ai_scan(mag, czi):
            choices = get_scanregion_choices(mag, czi)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        ai_mag_dd.change(
            update_ai_czi,
            inputs=ai_mag_dd,
            outputs=ai_czi_dd
        )

        ai_czi_dd.change(
            update_ai_scan,
            inputs=[ai_mag_dd, ai_czi_dd],
            outputs=ai_scan_dd
        )

        ai_question = gr.Textbox(
            label="Ask AI",
            lines=4,
            placeholder="Type your morphology question here..."
        )

        ai_image = gr.Image(
            label="Image sent to local VLM",
            type="filepath"
        )

        ai_output = gr.Markdown()

        def ai_answer_clean(
            target_type,
            question,
            patch_label,
            patch_cluster_id,
            montage_cluster_id,
            magnification,
            czi_base,
            scanregion
        ):
            if question is None or question.strip() == "":
                return None, "Please type a question."

            if target_type == "Selected patch":
                image_path, context = get_ai_target_image(
                    "Selected patch",
                    patch_label,
                    patch_cluster_id,
                    magnification,
                    czi_base,
                    scanregion
                )

            elif target_type == "Cluster montage":
                image_path, context = get_ai_target_image(
                    "Cluster montage",
                    patch_label,
                    montage_cluster_id,
                    magnification,
                    czi_base,
                    scanregion
                )

            elif target_type == "ScanRegion overlay":
                image_path, context = get_ai_target_image(
                    "ScanRegion overlay",
                    patch_label,
                    montage_cluster_id,
                    magnification,
                    czi_base,
                    scanregion
                )

            else:
                return None, "Unknown AI target."

            if image_path is None:
                return None, context

            answer = call_ollama_vlm(
                image_path=image_path,
                question=question,
                context_text=context
            )

            return image_path, answer

        gr.Button("Ask AI").click(
            ai_answer_clean,
            inputs=[
                ai_target,
                ai_question,
                ai_patch_dd,
                ai_patch_cluster_dd,
                ai_montage_cluster_dd,
                ai_mag_dd,
                ai_czi_dd,
                ai_scan_dd
            ],
            outputs=[
                ai_image,
                ai_output
            ]
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(BASE_DIR)]
    )
