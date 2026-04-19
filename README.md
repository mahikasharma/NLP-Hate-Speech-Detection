# Hate Speech Detection: CS 4120 Final Project

**Mahika Sharma, Aparajitha Karipineni, Maithili Ubgade**

## Overview

This project explores automated hate speech detection on Twitter using the Davidson et al. (2017) dataset, which categorizes tweets into three classes: hate speech, offensive language, and neither. We compare three models: Random Forest, Logistic Regression, and LSTM, to evaluate their ability to distinguish between these closely related categories.

## Dataset

[Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)

Download `labeled_data.csv` from the link above and place it in the project root before running the notebook.

## Models

- **Random Forest** — TF-IDF + engineered features, hyperparameter tuning via RandomizedSearchCV
- **Logistic Regression** — TF-IDF with regularization tuning, used as an interpretable linear baseline
- **LSTM** — Sequential model with learned token embeddings, bidirectional with 256 hidden units

## Requirements

Python 3.12. Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn scipy torch
```

## Usage

1. Clone the repo:
```bash
git clone <your-repo-url>
```
2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn scipy torch
```
3. Download `labeled_data.csv` from the dataset link above and place it in the project root
4. Run `NLP_ProjectCode_final.ipynb` top to bottom

## Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.8616 | 0.7317 | 0.9292 |
| Random Forest | 0.9041 | 0.7400 | 0.9291 |
| LSTM | 0.8600 | 0.7100 | 0.8900 |
