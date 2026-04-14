# Hate Speech Detection — CS 4120 Final Project
**Mahika Sharma, Aparajitha Karipineni, Maithili Ubgade**

## Overview
This project explores automated hate speech detection on Twitter using the Davidson et al. (2017) dataset, which categorizes tweets into three classes: hate speech, offensive language, and neither. We compare three models — Random Forest, Logistic Regression, and LSTM — to evaluate their ability to distinguish between these closely related categories.

## Dataset
[Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language)

## Models
- **Random Forest** — TF-IDF + engineered features, hyperparameter tuning via RandomizedSearchCV
- **Logistic Regression** — TF-IDF with regularization tuning, used as an interpretable linear baseline
- **LSTM** — Sequential model with learned token embeddings

## Requirements
- pandas, numpy, matplotlib, seaborn, nltk, scikit-learn, scipy, torch

## Usage
1. Download `labeled_data.csv` from the dataset link above and place it in the project root
2. Run `NLP_ProjectCode_final.ipynb` top to bottom
