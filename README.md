# Heart Disease Prediction using Machine Learning

**CSE-4115 Machine Learning — Project Assignment**  
**Student:** Zarif Ahnaf Dhrubo — **ID:** 0812310205101023  
**Department of CSE, BAUET — Batch 17th, Summer-26**

A complete, reproducible supervised-learning project that predicts the presence of heart disease
from routine clinical measurements.

## Dataset

**Heart Failure Prediction Dataset** — [fedesoriano, Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

- 918 patients, 11 clinical features, binary target `HeartDisease`
- Class balance: 55.3% heart disease / 44.7% no heart disease
- Local copy: `data/heart.csv`
- Data-quality handling: 172 impossible zero cholesterol values and 1 zero resting blood pressure
  are treated as missing and median-imputed inside the model pipeline

The Heart Disease dataset was selected after benchmarking all four datasets suggested in the
assignment. It gave the strongest balanced classification results (≈ 88.6% accuracy, 0.933 ROC-AUC)
with clear clinical interpretability.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.8859** | 0.8857 | **0.9118** | **0.8986** | 0.9323 |
| Gaussian Naive Bayes | 0.8750 | **0.9072** | 0.8627 | 0.8844 | **0.9430** |
| Support Vector Machine | 0.8696 | 0.8679 | 0.9020 | 0.8846 | 0.9360 |
| Random Forest | 0.8641 | 0.8812 | 0.8725 | 0.8768 | 0.9307 |
| K-Nearest Neighbors | 0.8478 | 0.8627 | 0.8627 | 0.8627 | 0.9236 |
| Decision Tree | 0.7391 | 0.7700 | 0.7549 | 0.7624 | 0.7372 |

- Best held-out model: **Logistic Regression** (88.59% accuracy, 0.8986 F1, 0.9323 ROC-AUC).
- Best cross-validated model: **SVM** (85.84% mean accuracy over 5 stratified folds).
- Top predictors: **ST slope, MaxHR, Oldpeak, exercise-induced angina, chest-pain type, sex, age**.

## What the Notebook Covers

1. Data loading and inspection
2. Domain-aware cleaning (invalid zeros → missing, median imputation)
3. Feature encoding and standardisation with leakage-safe `Pipeline`s
4. EDA: target distribution, histograms, categorical disease rates, box plots, correlation heat map,
   pair plot
5. Stratified 80/20 train/test split
6. Six classifiers: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, Random Forest, SVM
7. Evaluation: accuracy, precision, recall, F1, confusion matrices, ROC curves
8. 5-fold stratified cross-validation
9. `GridSearchCV` tuning for Random Forest and Logistic Regression
10. Feature importance: impurity, permutation importance, logistic coefficients
11. Saved pipelines and a full results summary

## Project Structure

```
.
├── data/heart.csv
├── notebooks/Heart_Disease_Prediction.ipynb   # executed notebook (code + outputs)
├── report/
│   ├── Project_Report.md                      # full written report
│   ├── Project_Report.html                    # print-ready report
│   ├── Project_Report.pdf                     # PDF version of the report
│   ├── Heart_Disease_Prediction.html          # rendered notebook
│   ├── results.json
│   ├── model_comparison.csv
│   ├── cross_validation.csv
│   ├── final_model_comparison.csv
│   ├── tuning_random_forest.csv
│   ├── tuning_logistic_regression.csv
│   └── figures/                               # 14 EDA and result figures
├── models/
│   ├── random_forest_tuned.joblib
│   └── logistic_regression_tuned.joblib
├── main.py
├── pyproject.toml
└── README.md
```

## Reproduce

```bash
uv sync
uv run python main.py
```

`main.py` executes the notebook in place, regenerates every figure, metric file and saved model,
and re-exports the rendered HTML report.

To explore interactively:

```bash
uv run jupyter lab notebooks/Heart_Disease_Prediction.ipynb
```

## Environment

Python 3.14.5, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.

## Report

The full methodology, results, discussion, limitations and future work are in
[`report/Project_Report.md`](report/Project_Report.md).
