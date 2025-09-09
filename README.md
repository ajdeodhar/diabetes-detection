# 🧠 Diabetes Risk Prediction using Machine Learning

This project builds a robust machine learning pipeline to **predict the risk of diabetes** based on health parameters. It utilizes popular classification algorithms (Logistic Regression, Random Forest, XGBoost) and compares their performance using metrics like **ROC-AUC**, accuracy, and F1-score.

> 📍 **Live Notebook + Code:** All model training, evaluation, and visualizations are included in the code uploaded here.

---

## 🔍 Problem Statement

Given a set of patient attributes (age, BMI, HbA1c levels, blood pressure, etc.), can we **predict whether a person is likely to have diabetes**?

This project aims to assist healthcare professionals or policymakers in **early risk detection** and **screening prioritization** using interpretable and scalable machine learning models.

---

## 📦 Dataset Used

The dataset was sourced from **Kaggle** and contains:
- 100,000 records
- Features like `age`, `gender`, `smoking history`, `bmi`, `HbA1c level`, `blood glucose level`, `hypertension`, and `heart disease`
- Binary target variable: `diabetes` (0 = No, 1 = Yes)

---

## ⚙️ Pipeline Overview

1. **Data Cleaning & Preprocessing**
   - Handled missing/null values
   - One-hot encoded categorical features (like `smoking_history`)
   - Converted `True/False` flags to binary values

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions of diabetic vs. non-diabetic groups
   - Identified key risk factors and correlations

3. **Model Training**
   - Trained multiple models:  
     - `LogisticRegression`  
     - `RandomForestClassifier`  
     - `XGBClassifier`
   - Each model returns a **diabetes probability score** (e.g., `LR_diabetes_probability`)

4. **Evaluation Metrics**
   - Accuracy, F1-score
   - **ROC-AUC Score** to evaluate classifier’s ability to rank patients by risk
   - Model comparison via ROC curves

5. **Feature Importance Analysis**
   - Identified most influential features
   - Discussed interpretability

---

## 📊 Key Results

| Model                | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 86.4%    | 0.918   |
| Random Forest       | 92.2%    | 0.942   |
| XGBoost             | 93.0%    | 0.951   |

- All models showed strong performance, especially XGBoost.
- ROC-AUC analysis helped understand which model better **ranks** diabetic risk across the population.

---

## 🔍 ROC-AUC Explained (In Simple Terms)

- Instead of just **counting correct predictions**, ROC-AUC checks how well the model **ranks patients by risk**.
- A model with **0.95 ROC-AUC** means there's a **95% chance** it will score a diabetic patient higher than a non-diabetic one.
- Useful for **screening scenarios** where resources are limited and you want to test high-risk people first.

---

## 🧪 Future Work / Possible Extensions

- Incorporate Bayesian methods for **uncertainty-aware predictions**
- Add **SHAP-based interpretability** to show why a specific patient was classified as high-risk
- Deploy as a **streamlit web app** for demo and testing

