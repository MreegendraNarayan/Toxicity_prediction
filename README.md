# Toxicity Prediction

A machine learning project for predicting molecular toxicity using both classical DeepChem baselines and advanced Graph Neural Networks (GNNs)
<!--, aiming to reduce the need for animal testing in lab experiments.
-->
---

## Overview

This repository contains multiple approaches to molecular toxicity prediction, focusing on the Tox21 dataset. It includes:

-  Baseline DeepChem models
-  GNN-based models using PyTorch Geometric
-  Advanced GNNs with enriched features
- 🔜 Utility scripts for feature engineering, training, and evaluation
- 🔜 Pretrained models for inference

---

## What’s in this download

This folder is a **notebook-only snapshot**. In this copy, the implemented work lives in:

- `notebooks/01_baseline_deepchem.ipynb`
- `notebooks/02_gnn_baseline.ipynb`
- `notebooks/03_gnn_advanced_features.ipynb`

If you don’t see `utils/`, `models/`, or `requirements.txt`, that’s expected for this download.

For a practical “how to run + how it works” walkthrough, see `PROJECT_GUIDE.md`.

---

## Project Structure

<pre>toxicity-prediction/ 
├── data/ # optional - just metadata (not huge files) 
├── notebooks/ 
│ ├── 01_baseline_deepchem.ipynb 
│ ├── 02_gnn_baseline.ipynb 
│ └── 03_gnn_advanced_features.ipynb # (the one we're about to build) 
├── models/ # trained model files (optional) 
├── utils/ # helper functions 
├── README.md 
└── requirements.txt </pre>


---

## Dataset

- **Tox21** dataset from DeepChem
- Contains ~12,000 compounds with binary toxicity labels across 12 targets
- [Link to dataset and paper](https://pubs.acs.org/doi/10.1021/ci400187y)

---
### 1. Clone the repo
```bash
git clone https://github.com/MreegendraNarayan/Toxicity_prediction.git
cd Toxicity_prediction
```
Contributions
Feel free to open issues, submit pull requests, or suggest new features!

## Goal
This project aims to contribute toward ethical AI-driven alternatives to traditional lab-based toxicity testing.
