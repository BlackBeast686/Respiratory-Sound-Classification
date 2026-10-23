# Respiratory Sound Classification

Classifying **respiratory / lung-sound recordings** into five categories — **asthma**, **bronchial**, **COPD**, **healthy**, and **pneumonia** — from audio. The project extracts acoustic features from `.wav` clips and compares a broad range of models, from classical ML and CNNs up to graph neural networks and a hybrid GNN ensemble.

> **Note on scope:** This is a research/experimental notebook for comparing models on respiratory audio. It is **not** a medical device and must not be used for diagnosis.

## Task & data

- **Inputs:** `.wav` respiratory-sound recordings
- **Outputs:** one of five classes — `asthma`, `Bronchial`, `copd`, `healthy`, `pneumonia`
- **Expected layout:** `data_root/<class_name>/*.wav`

The dataset is **not** included in this repository — the notebook reads it from a local path (originally Google Drive: `Asthma Detection Dataset`). Point `data_root` at your own copy with the folder structure above.

### Features

Each clip is loaded at 16 kHz, fixed to a 3-second length, pre-emphasised and normalised, then summarised with **MFCCs**, a **mel-spectrogram**, **chroma**, **spectral contrast**, **tonnetz**, **zero-crossing rate**, and **RMS energy** (mean/std pooled). The GNN models additionally build a k-nearest-neighbour graph over the feature vectors, with PCA for dimensionality reduction.

## Models compared

- **Graph neural networks** — single-model GCN, GAT, GraphSAGE, and GIN baselines, plus a **hybrid GNN ensemble** (GCN / GAT / TransformerConv branches with stacking) as the headline model.
- **Classical ML** — SVM (RBF), Random Forest, XGBoost on pooled features.
- **CNN baselines** — 1D and 2D convolutional networks (TensorFlow/Keras) on MFCC / mel-spectrogram inputs.

## Results

Test metrics as reported in the notebook (single runs):

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| Hybrid GNN ensemble (stacking) | 0.982 | 0.982 | 0.999 |
| GraphSAGE | 0.960 | 0.960 | 0.995 |
| GCN | 0.937 | 0.937 | 0.995 |
| GIN | 0.930 | 0.929 | 0.990 |
| GAT | 0.925 | 0.925 | 0.992 |
| SVM (RBF) | 0.905 | 0.895 | 0.983 |
| Random Forest | 0.901 | 0.883 | 0.983 |
| XGBoost | 0.885 | 0.863 | 0.985 |
| CNN baselines (1D / 2D) | 0.33–0.86 | — | — |

> The GNN models clearly outperform the classical and CNN baselines here, with the hybrid ensemble best overall. Figures are single runs on the notebook's split — treat them as indicative rather than definitive.

## Requirements

Developed in Google Colab with a GPU runtime. Main dependencies:

- `torch`
- `torch_geometric`
- `tensorflow`
- `librosa`
- `audiomentations`
- `scikit-learn`
- `xgboost`
- `numpy`
- `scipy`
- `networkx`
- `matplotlib`
- `seaborn`
- `tqdm`

Install everything with:

```bash
pip install -r requirements.txt
```

## Getting started

1. Open the notebook in Google Colab or Jupyter with a GPU.
2. Set `data_root` to your dataset path (folders of `.wav` files, one per class).
3. Run the cells for whichever models you want to compare — each major section (hybrid ensemble, GCN, GAT, GraphSAGE, GIN, classical, CNN) is self-contained.

## Repository structure

```
.
├── Astama_Detection.ipynb
└── README.md
```
