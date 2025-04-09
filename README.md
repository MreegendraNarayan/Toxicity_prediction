# Toxicity Prediction

A machine learning project for predicting molecular toxicity using both classical DeepChem baselines and advanced Graph Neural Networks (GNNs), aiming to reduce the need for animal testing in lab experiments.

---

## 🧬 Overview

This repository contains multiple approaches to molecular toxicity prediction, focusing on the Tox21 dataset. It includes:

-  Baseline DeepChem models
-  GNN-based models using PyTorch Geometric
-  Advanced GNNs with enriched features
- 🔜 Utility scripts for feature engineering, training, and evaluation
- 🔜 Pretrained models for inference

---

## 📂 Project Structure

toxicity-prediction/ 
├── data/   # optional metadata or processed data files 
├── notebooks/   # main development notebooks 
│ ├── 01_baseline_deepchem.ipynb 
│ ├── 02_gnn_baseline.ipynb 
│ └── 03_gnn_advanced_features.ipynb 
├── models/   # trained model files or weights (optional) 
├── utils/ # helper functions for loading, training, evaluation 
├── README.md # project overview 
└── requirements.txt # dependencies


---

## Dataset

- **Tox21** dataset from DeepChem
- Contains ~12,000 compounds with binary toxicity labels across 12 targets
- [Link to dataset and paper](https://pubs.acs.org/doi/10.1021/ci400187y)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/LazyCOMMAnd3r/Toxicity_prediction.git
cd Toxicity_prediction
