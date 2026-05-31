# DermPath-AI-Explorer

AI-assisted dermatopathology image exploration using foundation-model embeddings, morphology clustering, similarity search, ScanRegion visualization, and local vision-language model (VLM) interpretation.

---

## Overview

DermPath-AI-Explorer is an interactive AI platform for exploring dermatopathology H&E whole-slide image regions. The system converts ZEISS CZI scan regions into tissue patches, extracts DINOv2 foundation-model embeddings, organizes tissue morphology into visual clusters, retrieves similar image regions, visualizes morphology distributions using UMAP, and provides AI-assisted morphology interpretation through a local vision-language model.

The goal is to help researchers efficiently explore large-scale dermatopathology datasets, discover morphological patterns, understand tissue heterogeneity, and interactively investigate image regions through an AI-assisted workflow.

---

## Key Features

### Dataset Exploration

- Supports ZEISS CZI-derived dermatopathology scan regions
- Organizes images by magnification and ScanRegion
- Provides dataset-level statistics and metadata summaries

### AI-Based Morphology Discovery

- DINOv2 foundation-model feature extraction
- Global morphology clustering using KMeans
- UMAP visualization of morphology embeddings
- Discovery of recurring tissue patterns across the entire dataset

### ROI and ScanRegion Analysis

- ScanRegion-level morphology overlays
- Visual prioritization of tissue regions
- Morphology distribution analysis within whole tissue sections

### Similar Patch Retrieval

- Interactive patch selection
- Cosine similarity-based retrieval
- Foundation-model embedding search
- Exploration of visually related tissue structures

### AI Morphology Assistant

- Local Ollama + LLaVA integration
- Morphology-aware image interpretation
- Contextual descriptions of selected tissue regions
- Interactive pathology exploration interface

---

## System Workflow

The complete workflow consists of the following stages:

```text
ZEISS CZI Images
        ↓
QuPath ScanRegion Export
        ↓
256×256 Tissue Patch Extraction
        ↓
Background / Blank Patch Filtering
        ↓
DINOv2 Feature Extraction
        ↓
Embedding Aggregation
        ↓
UMAP Visualization
        ↓
KMeans Morphology Clustering
        ↓
Cluster Montage Generation
        ↓
ScanRegion Morphology Overlay
        ↓
Similar Patch Retrieval
        ↓
Local VLM Interpretation
```

---

## User Interface

The application currently contains seven interactive modules.

### 1. Dataset + Pipeline Overview

Provides:

- Dataset statistics
- Number of CZI images
- Number of ScanRegions
- Number of tissue patches
- Global cluster statistics
- End-to-end processing workflow

---

### 2. ScanRegion Morphology Overlay

Displays:

- Selected ScanRegion
- Global morphology cluster overlays
- Spatial distribution of morphology patterns
- ROI-level visual exploration

This view helps identify how different morphology clusters are distributed across tissue regions.

---

### 3. Global Morphology Cluster Explorer

Provides:

- Global cluster selection
- Representative cluster montages
- Cluster statistics
- Top contributing images and ScanRegions

This module allows users to understand dominant tissue morphology groups discovered across the dataset.

---

### 4. Patch Explorer + Similar Morphology Search

Provides:

- Patch selection
- Patch metadata inspection
- Similar patch retrieval
- Similarity scores
- Embedding-based image search

This enables efficient exploration of related tissue structures throughout the dataset.

---

### 5. UMAP Morphology Landscape

Provides:

- UMAP visualization of DINOv2 embeddings
- Cluster-colored morphology landscape
- Global tissue organization view

Each point represents a tissue patch, and nearby points correspond to morphologically similar regions.

---

### 6. Disease Classification Module

Provides:

- Disease prediction
- Confidence estimation
- Morphology interpretation
- AI-generated explanatory summaries

This module demonstrates how foundation-model features and AI interpretation can support future disease classification workflows.

---

### 7. AI Morphology Assistant

Provides:

- AI-assisted tissue interpretation
- Local LLaVA analysis
- Morphology-focused explanations
- Interactive visual reasoning

The assistant operates entirely through local Ollama-hosted vision-language models.

---

## Example Screenshots

### Dataset Overview
<p align="center">
  <img width="1625" height="801" alt="image" src="https://github.com/user-attachments/assets/e05383b8-9633-4956-8a70-4ddcf72f2b68" />
</p>

---

### ScanRegion Morphology Overlay
<p align="center">
  <img width="1910" height="470" alt="image" src="https://github.com/user-attachments/assets/9a6971eb-e2fa-4f45-9457-d30728e71c23" />
  <img width="1910" height="877" alt="image" src="https://github.com/user-attachments/assets/32b770fb-59ce-4225-a0ca-f9b54dd7c8e0" />
</p>

---

### Global Morphology Cluster Explorer
<p align="center">
  <img width="1253" height="822" alt="image" src="https://github.com/user-attachments/assets/6d71ec73-88a6-4726-87d9-4eb466796ed7" />
  <img width="1254" height="510" alt="image" src="https://github.com/user-attachments/assets/6bb259d8-82f2-4cb6-859d-47db41122075" />
</p>


Include screenshot from Tab 3.

---

### Similar Patch Retrieval

Include screenshot from Tab 4.

---

### UMAP Morphology Landscape

Include screenshot from Tab 5.

---

### AI Morphology Assistant

Include screenshot from Tab 7.

---

## Example Dataset

Current prototype dataset:

| Item | Count |
|--------|--------|
| Whole-slide CZI Images | 30 |
| ScanRegions | 217 |
| Tissue Patches | ~96,000 |
| Global Morphology Clusters | 8 |
| Magnifications | 10×, 20× |

---

## AI Components

### Foundation Model

- DINOv2

Used for:

- Feature extraction
- Morphology representation learning
- Similarity search

---

### Vision-Language Model

- LLaVA (via Ollama)

Used for:

- Morphology interpretation
- Tissue description
- Interactive AI assistance

---

### Clustering

- KMeans

Used for:

- Global morphology grouping
- Cluster montage generation
- ROI exploration

---

### Dimensionality Reduction

- UMAP

Used for:

- Visualization
- Embedding exploration
- Morphology landscape analysis

---

## Repository Structure

```text
DermPath-AI-Explorer
│
├── app/
│   └── app_gradio.py
│
├── preprocessing/
│   ├── convert_czi.py
│   ├── patchify_scanregions.py
│   ├── filter_background.py
│   └── metadata_generation.py
│
├── embeddings/
│   ├── extract_dinov2_embeddings.py
│   └── merge_embeddings.py
│
├── clustering/
│   ├── run_umap.py
│   ├── run_kmeans.py
│   └── build_cluster_montages.py
│
├── retrieval/
│   └── similarity_search.py
│
├── assets/
│   ├── overview.png
│   ├── scanregion_overlay.png
│   ├── cluster_explorer.png
│   ├── similarity_search.png
│   ├── umap.png
│   └── ai_assistant.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Create environment:

```bash
conda create -n dermpath_ai python=3.10
conda activate dermpath_ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

```text
gradio
numpy
pandas
opencv-python
pillow
matplotlib
scikit-learn
umap-learn
torch
torchvision
transformers
```

---

## Ollama Setup

Install Ollama:

https://ollama.com

Pull LLaVA:

```bash
ollama pull llava
```

Start server:

```bash
ollama serve
```

For example the default endpoint:

```text
http://localhost:11434
```

---

## Running the Application

Launch:

```bash
python 9_app_skin_histology_ai_with_AI_3.py
```

Open:

```text
http://127.0.0.1:7860
```

---

## Potential Applications

- Dermatopathology research
- Morphology discovery
- Whole-slide image exploration
- Foundation-model evaluation
- Histopathology AI prototyping
- Similar image retrieval
- ROI prioritization
- Interactive pathology education
- Multimodal AI research

---

## Future Directions

Planned extensions include:

- Multimodal integration of pathology images, genomics, and clinical notes
- Agentic AI pathology copilot
- Whole-slide foundation-model analysis
- Disease classification pipelines
- Explainable AI reporting
- Precision medicine applications
- Retrieval-augmented pathology systems
- Multi-agent dermatopathology assistants

---

## Citation

If you use this repository in academic work, please cite:

```text
DermPath-AI-Explorer:
AI-Assisted Dermatopathology Exploration Using Foundation Models,
Morphology Clustering, Similarity Search, and Local VLM Interpretation.
```

---

## Disclaimer

This software is intended for research and educational purposes only and is not intended for clinical diagnosis, treatment planning, or medical decision-making.

---

## License

MIT License
