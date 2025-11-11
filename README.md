# VPR Pipeline — Final Complete (English)

**Visual Place Recognition (VPR)** pipeline implemented in Python — modular, GPU-ready, and Colab-friendly.
This repository provides a clean and extensible baseline for research and experiments in image-based place recognition.

## Project overview (short)
Goal: given a *query image*, find the database images that show the same place (despite viewpoint/lighting changes).
Pipeline: **Feature Extraction → Retrieval (KNN/FAISS) → Re-ranking (geometric verification) → Evaluation (Recall@K)**.

## Architecture diagram
Images → Feature Extraction → Retrieval (FAISS / KNN) → Re-ranking (Light: ORB/RANSAC / Full: LightGlue/LoFTR) → Evaluation

## Pipeline phases — purpose & example
1. **Feature extraction** — produce a compact descriptor for each image (global embedding). Example: ResNet50 global pooled vector (2048-d).
2. **Retrieval (KNN / FAISS)** — for each query descriptor, return top-K nearest neighbors in descriptor space. Example: top-100 candidates.
3. **Re-ranking** — refine the top-K by checking geometric consistency between query and candidate (local features + RANSAC or learned matchers). Example: promote candidates with many inliers.
4. **Evaluation** — measure Recall@1, @5, @10 across all queries to quantify performance.

## Selectable backbones
You can choose different feature extractors when running the pipeline:

| Backbone | Description | Recommended Mode |
|---|---:|---|
| **resnet50** | Standard CNN global pooling (baseline) | `light` or `full` |
| **netvlad** | CNN + VLAD pooling for robust retrieval (placeholder included) | `full` |
| **mixvpr** | State-of-the-art learned descriptor (placeholder included) | `full` |

## Quick start (local)
1. create venv and activate
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. (optional) install heavy matchers and MixVPR (if you plan to use full mode):
```
bash scripts/setup_colab.sh
# or manually install:
# pip install git+https://github.com/amaralibey/MixVPR.git
```
3. (optional) download weights manually and place under `weights/` (see scripts/download_matcher_weights.sh)
4. run a quick smoke test with example data:
```
python3 src/cli.py pipeline --db example_data/database --query example_data/queries --backbone resnet50 --mode light
```

## Run on Google Colab
Click the badge to open the demo notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SabraHashemi/VPR_Pipeline/blob/main/VPR_Pipeline_Demo.ipynb)

Follow the notebook steps to setup environment, optionally download weights, and run pipeline in GPU-enabled Colab.

## Datasets
This project supports popular VPR datasets such as **GSV-XS, Tokyo-XS, SF-XS, SVOX**.
See `scripts/download_datasets.sh` for helper steps and links — note that many datasets require manual acceptance of terms or request forms; the script automates cloning helper repos where possible and provides instructions.

## Weights & matchers
- Light (ORB) requires no pretrained weights.
- Full matchers (LightGlue / LoFTR) and backbones (NetVLAD / MixVPR) require pretrained weights. See `scripts/download_matcher_weights.sh` for manual download instructions and target paths (`weights/netvlad/`, `weights/mixvpr/`, `weights/lightglue/`, `weights/loftr/`).

## Files of interest
- `src/cli.py` — main CLI and `pipeline` command
- `src/models/*` — backbones (resnet50, placeholder netvlad, mixvpr)
- `src/retrieval/*` — extractor & FAISS index/search
- `src/rerank/*` — light & full re-ranking modules
- `scripts/setup_colab.sh` — dependency installation for Colab
- `scripts/download_datasets.sh` — dataset helper script (links & clones)
- `scripts/download_matcher_weights.sh` — manual weight download instructions
- `VPR_Pipeline_Demo.ipynb` — Colab-ready demo notebook

## Example command
```
python3 src/cli.py pipeline --db example_data/database --query example_data/queries --backbone resnet50 --mode light --use_gpu
```

## License & acknowledgements
This repo collects and integrates ideas from public VPR works (NetVLAD, LoFTR, SuperGlue, MixVPR, FAISS).
Please cite the original papers when using their models and respect dataset licenses.
