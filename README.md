#  End-to-End ML Pipeline — Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Joblib](https://img.shields.io/badge/joblib-Export-4B8BBE?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge)

**DevelopersHub Corporation — AI/ML Engineering Internship**  
**Task 2 | Advanced Internship Tasks**

</div>

---

##  Objective

Build a **reusable, production-ready ML pipeline** that predicts customer churn for a telecom company.

The pipeline must:
- Handle all preprocessing (imputation, scaling, encoding) automatically via `sklearn.pipeline.Pipeline`
- Train two classifiers: **Logistic Regression** and **Random Forest**
- Use **GridSearchCV** for systematic hyperparameter optimization
- Export the final trained pipeline using **joblib** for production deployment
- Allow raw data to be passed directly at inference time — no manual preprocessing needed

---

##  Business Context

Customer churn — when a subscriber cancels their service — costs telecom companies billions annually. Acquiring a new customer costs **5–10× more** than retaining an existing one.

A reliable churn prediction model allows the business to:
- Identify at-risk customers **before** they leave
- Target them with retention offers (discounts, service upgrades)
- Prioritize retention budget on the highest-value churning customers

---

##  Pipeline Architecture

![Pipeline Architecture](assets/09_pipeline_architecture.png)

> **Key principle:** The `Pipeline` object guarantees that all preprocessing steps are fit only on training data — preventing **data leakage**.

```
Raw CSV  →  TotalCharges Fix  →  ColumnTransformer  →  Train/Test Split
    →  GridSearchCV (5-Fold CV)  →  Best Estimator  →  joblib Export (.pkl)
```

**Each full pipeline contains:**

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `SimpleImputer(median)` | Handle missing numerical values |
| 2 | `StandardScaler` | Normalize numerical features |
| 3 | `SimpleImputer(most_frequent)` | Handle missing categorical values |
| 4 | `OneHotEncoder` | Encode categorical features |
| 5 | `ColumnTransformer` | Apply steps 1–4 to correct columns |
| 6 | `LogisticRegression` / `RandomForest` | Classification |

---

##  Project Structure

```
task2-customer-churn-pipeline/
│
├── task2_customer_churn_ml_pipeline.ipynb   ← Main Jupyter Notebook
│
├── assets/                                  ← All visualizations
│   ├── 01_churn_distribution.png
│   ├── 02_numerical_vs_churn.png
│   ├── 03_categorical_vs_churn.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_confusion_matrices.png
│   ├── 06_roc_curves.png
│   ├── 07_feature_importance.png
│   ├── 08_metrics_comparison.png
│   └── 09_pipeline_architecture.png
│
├── models/
│   ├── logistic_regression_pipeline.pkl    ← Exported LR pipeline
│   └── random_forest_pipeline.pkl          ← Exported RF pipeline
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Dataset (place here)
├── requirements.txt
└── README.md
```

---

##  Dataset

**IBM Telco Customer Churn Dataset**

| Property | Value |
|----------|-------|
| Source | [Kaggle — IBM Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 (19 predictive + customerID + target) |
| Target | `Churn` — Yes / No |
| Class Ratio | ~73.5% No Churn · ~26.5% Churn |

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/task2-customer-churn-pipeline.git
cd task2-customer-churn-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.

>  If CSV not found, the notebook auto-generates a realistic synthetic dataset.

### 4. Run the Notebook
```bash
jupyter notebook task2_customer_churn_ml_pipeline.ipynb
```

---

##  Exploratory Data Analysis

### Churn Distribution
![Churn Distribution](assets/01_Churn_Distribution.png)

> **Insight:** ~26.5% of customers churn. This class imbalance means accuracy alone is misleading — a naive "predict no churn always" model scores 73.5%. We use **F1-Score** and **ROC-AUC** as primary metrics.

---

### Numerical Features vs Churn
![Numerical Features vs Churn](assets/02_numerical_vs_churn.png)

> **Key findings:**
> - **Tenure:** New customers (low tenure) churn far more — long-term customers are loyal
> - **Monthly Charges:** Higher bills → more dissatisfaction → more churn
> - **Total Charges:** Low total charges = new customers = high churn risk

---

### Categorical Features vs Churn Rate
![Categorical Features vs Churn](assets/03_categorical_vs_churn.png)

> **Key findings:**
> - **Contract type** is the single biggest driver — Month-to-month customers churn at ~42% vs ~11% for 2-year contracts
> - **Fiber optic** internet users churn more than DSL (possibly due to higher cost/competition)
> - **Electronic check** payment method has the highest churn rate among payment types
> - **Paperless billing** customers churn more — possibly tech-savvy users who compare offers more easily

---

### Correlation Heatmap
![Correlation Heatmap](assets/04_correlation_heatmap.png)

> **Key findings:**
> - `tenure` has the strongest **negative** correlation with churn (-0.35) — longer customers stay
> - `MonthlyCharges` has a positive correlation (+0.19) with churn
> - `TotalCharges` = `tenure × MonthlyCharges` — so it's highly correlated with both (multicollinearity)

---

##  Model Development

### Preprocessing Pipeline (No Data Leakage)

```python
# Numerical branch
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # handles TotalCharges NaN
    ('scaler',  StandardScaler())
])

# Categorical branch
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

# Combined ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_transformer,   numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(...))
])
```

### GridSearchCV — Hyperparameter Tuning

| Model | Parameters Searched | Best Config |
|-------|-------------------|-------------|
| Logistic Regression | C: [0.01, 0.1, 1, 10, 100], solver: [lbfgs, liblinear] | C=10, liblinear |
| Random Forest | n_estimators: [100,200,300], max_depth: [None,10,20], min_samples: [2,5] | 200 trees, depth=10 |

> Scoring metric: **F1-score** with **5-fold StratifiedKFold** — ensures class ratios are preserved in each fold.

---

##  Results & Visualizations

### Confusion Matrices
![Confusion Matrices](assets/05_confusion_matrices.png)

> **Reading the matrix:**
> - **TN (top-left):** Correctly predicted No Churn → these customers are retained
> - **TP (bottom-right):** Correctly predicted Churn → these can be targeted for retention
> - **FN (bottom-left):** Missed churners → most costly business error
> - **FP (top-right):** Non-churners flagged as churn → unnecessary retention spend

---

### ROC Curves
![ROC Curves](assets/06_roc_curves.png)

> **ROC-AUC = 0.84** means the model correctly ranks a random churning customer above a non-churner **84% of the time**.
>
> Both models significantly outperform the random classifier baseline (AUC = 0.50), confirming genuine predictive power.

---

### Feature Importance — Random Forest
![Feature Importance](assets/07_feature_importance.png)

> **Top 5 churn predictors identified by Random Forest:**
>
> | Rank | Feature | Why It Matters |
> |------|---------|----------------|
> | 1 | `tenure` | New customers have no loyalty — they leave |
> | 2 | `TotalCharges` | Proxy for how much value a customer has received |
> | 3 | `Contract_Month-to-month` | No commitment = easy to leave |
> | 4 | `MonthlyCharges` | High bills → price-sensitive customers more likely to churn |
> | 5 | `OnlineSecurity_No` | Less engaged with the service |

---

### Model Performance Comparison
![Metrics Comparison](assets/08_metrics_comparison.png)

> Both models achieve comparable performance. **Logistic Regression is preferred for production** due to:
> - Faster inference (~10ms vs ~200ms per batch)
> - Fully interpretable coefficients
> - 20× smaller `.pkl` file size

---

##  Final Results Summary

| Metric | Logistic Regression | Random Forest |
|--------|:------------------:|:-------------:|
| **Accuracy** | ~80.5% | ~80.4% |
| **Precision** | ~66% | ~67% |
| **Recall** | ~56% | ~52% |
| **F1-Score** | ~60% | ~59% |
| **ROC-AUC** | **~0.841** | **~0.841** |
| **CV F1 (5-fold)** | ~0.598 ± 0.014 | ~0.578 ± 0.018 |

---

##  Model Export & Reuse

```python
import joblib

# Export (done inside notebook)
joblib.dump(best_pipeline, 'models/random_forest_pipeline.pkl', compress=3)

# ── Load and predict on new raw data ──────────────────────────
model = joblib.load('models/random_forest_pipeline.pkl')

new_customer = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.70, 'TotalCharges': '151.65'
}])

prediction = model.predict(new_customer)[0]
prob       = model.predict_proba(new_customer)[0][1]

print(f'Prediction : {"CHURN ⚠️" if prediction == 1 else "NO CHURN ✅"}')
print(f'Churn Prob : {prob:.1%}')
```

---

##  Key Insights & Observations

1. **Class Imbalance (Critical):** 73.5%/26.5% split — accuracy is not enough. F1-Score and ROC-AUC are the correct metrics here.

2. **Hidden Missing Values Fixed:** `TotalCharges` had 11 rows with whitespace `' '` that pandas `.isnull()` does NOT detect. Fix: `.str.strip().replace('', np.nan)` before conversion, then median imputation inside the Pipeline.

3. **Pipeline = Zero Leakage:** All transformers (scaler, encoder, imputer) are fit exclusively on `X_train`. This prevents the model from "peeking" at test data during training — a common mistake in manual preprocessing.

4. **LR ≈ RF Performance:** Despite Random Forest's complexity, both achieve ~0.841 AUC. For production, Logistic Regression wins on speed, interpretability, and file size.

5. **Business Takeaway:** The top retention strategy should focus on **converting month-to-month customers to annual contracts** — this alone could reduce churn by ~30 percentage points for those customers.

---

##  Possible Improvements

- Add `class_weight='balanced'` to handle class imbalance directly
- Apply SMOTE oversampling on training data for minority class augmentation
- Test XGBoost / LightGBM for potentially better F1 performance
- Add `SelectFromModel` inside the pipeline for automatic feature selection
- Deploy as REST API using FastAPI + the exported `.pkl` pipeline

---

##  Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| `scikit-learn` | 1.5.x | Pipelines, models, GridSearchCV, metrics |
| `pandas` | 2.x | Data loading, cleaning, EDA |
| `numpy` | 2.x | Numerical operations |
| `matplotlib` | 3.9.x | Visualizations |
| `seaborn` | 0.13.x | Statistical plots |
| `joblib` | 1.4.x | Pipeline serialization |
| `jupyter` | 1.1.x | Interactive notebook |

---

##  Skills Demonstrated

-  **ML Pipeline construction** — `sklearn.pipeline.Pipeline` + `ColumnTransformer`
-  **Hyperparameter tuning** — `GridSearchCV` with 5-fold `StratifiedKFold`
-  **Production-readiness** — exported pipeline handles raw input at inference
-  **Model export & reusability** — `joblib.dump/load` with compression
-  **Proper evaluation** — F1, ROC-AUC, cross-validation, confusion matrix, precision/recall
-  **Data quality** — hidden whitespace missing values identified and fixed
-  **EDA & visualization** — 9 publication-quality charts

---

## 👤 Author

**[Bilal Ahmed]**  
AI/ML Engineering Intern — DevelopersHub Corporation  
 May 2026

---

<div align="center">
  <sub>Built with ❤️ as part of the DevelopersHub AI/ML Engineering Internship · Task 2</sub>
</div>
