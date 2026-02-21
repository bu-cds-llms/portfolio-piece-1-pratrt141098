# Interpretable Readmission Risk from Clinical Discharge Notes

## Overview
Predicts 30-day readmission from real de-identified discharge summaries (MIMIC-IV), blending classical NLP, neural nets, and attention for interpretability. Inspired by clinical decision support needs.

## Methods
- Lab1: TF-IDF + logistic regression
- Lab2: PyTorch MLP with loss visualization
- Lab3: GloVe embeddings + custom self-attention, with phrase highlighting

## Key Results
Attention model achieves highest AUC (0.XX) and surfaces clinical risk phrases like "poor social support".

## How to Run
1. `pip install -r requirements.txt`
2. Download MIMIC-IV-Note discharge.csv (link in 01_data.ipynb)
3. `jupyter nb 01_data...` → `03...`

## Requirements
torch scikit-learn gensim pandas ...

## Repository Structure
├── README.md
├── requirements.txt
├── notebooks/
│ ├── 01_data_and_baselines.ipynb (Lab 1)
│ ├── 02_neural_network_experiments.ipynb (Lab 2)
│ └── 03_embeddings_and_attention.ipynb (Lab 3)
├── src/ (reusable code)
├── data/ (gitignore large files)
└── outputs/ (figures, models)

## Key Results
| Model | AUC | Interpretability |
|-------|-----|------------------|
| TF-IDF Unigrams | 1.000 | Feature weights. |
| TF-IDF **Bigrams** | **1.000** | Clinical phrases ("worsening renal") |
| ReLU MLP | 1.000 | Loss curves converge fast |
| Tanh MLP | 1.000 | Slower but stable |
| **Attention** | **1.000** | **Heatmaps highlight "poor compliance high readmission risk"** |

**See attention viz**: !

## Acknowledgments
Built on MIMIC-IV-Note dataset [PhysioNet].
=======

