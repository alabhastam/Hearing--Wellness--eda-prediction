# 2025 Hearing Wellness Survey – Recall‑Focused Classification

[This project analyzes the [2025 Hearing Wellness Survey](https://www.kaggle.com/) to predict individuals potentially **at risk of hearing health issues** based on survey responses.  
The primary objective is **maximizing Recall** for the "at risk" class, ensuring minimal false negatives in health‑critical predictions.

---

## 📋 Project Overview

Hearing wellness is a critical but often overlooked aspect of public health.  
This project:
- Loads the 2025 Hearing Wellness Survey dataset automatically from Kaggle
- Performs extensive **EDA** to understand distribution and missing data
- Cleans, encodes, and scales features
- Sets up a supervised learning pipeline with a **Recall optimization focus**
- Compares baseline Logistic Regression with LightGBM

---

## ⚙️ Pipeline Steps

1. **Data Loading**
   - Auto‑detect and load CSV file directly from Kaggle input directory

2. **Exploratory Data Analysis (EDA)**
   - Dataset shape, info, and missing value analysis
   - Target distribution check
   - Feature statistics and visualization

3. **Data Cleaning & Preprocessing**
   - Median imputation for numeric features
   - Mode imputation for categorical features
   - Label Encoding for categorical variables
   - Standard Scaling for numerical variables

4. **Target Variable Setup**
   - Identification of "at risk" class
   - Optional binary conversion `(At risk → 1, Others → 0)`

5. **Modeling**
   - **Logistic Regression** (with `class_weight='balanced'`)
   - **LightGBM Classifier** (scale‑pos‑weight for imbalance handling)
   - Hyperparameters tuned for recall

6. **Evaluation**
   - **Recall Score** as the main metric
   - Confusion Matrix visualization
   - Support metrics (Precision, F1, Accuracy provided, but secondary priority)

---

## 📊 Results Summary

| Model              | Recall (At Risk) | Precision | F1 Score |
|--------------------|------------------|-----------|----------|
| Logistic Regression| 0.84             | 0.56      | 0.67     |
| LightGBM           | 0.88             | 0.59      | 0.71     |


---


