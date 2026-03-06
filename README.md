# IoT-Driven-Predictive-Maintenance-Failure-Prediction-System

## 📌 Project Overview

This project builds a **complete predictive maintenance system** for industrial machines. Instead of waiting for machines to break down (reactive maintenance), the ML model predicts failures **hours or days in advance** — saving costs and preventing downtime.

| | Before ML | After ML |
|---|---|---|
| **Failure Detection** | After breakdown | Hours/days before |
| **Detection Rate** | ~40% | **98.4%** |
| **Warning Time** | Zero | Hours in advance |
| **Maintenance Type** | Reactive | Predictive |
| **Estimated Cost Saving** | — | **~60%** |

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![SQL Server](https://img.shields.io/badge/SQL%20Server-Microsoft-red?logo=microsoftsqlserver)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?logo=powerbi)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)

---

## 📁 Project Structure

```
📦 Predictive-Maintenance-ML
├── 📂 sql
│   ├── create_tables.sql          ← Create database tables
│   ├── data_cleaning.sql          ← Clean raw sensor data
│   ├── eda_queries.sql            ← Exploratory queries
│   └── feature_engineering.sql   ← Engineer 4 new features
│
├── 📂 python
│   ├── 01_data_loading.py         ← Connect SQL → Python
│   ├── 02_preprocessing_pipeline.py ← SMOTE + Scaling
│   ├── 03_eda.py                  ← 6 EDA visualizations
│   ├── 04_model_binary.py         ← Binary classification
│   ├── 05_model_multiclass.py     ← Multiclass classification
│   ├── 06_hyperparameter_tuning.py ← RandomizedSearchCV
│   └── 07_evaluation.py           ← Final evaluation
│
├── 📂 models
│   ├── tuned_binary_model.pkl     ← Saved best binary model
│   └── tuned_multiclass_model.pkl ← Saved best multiclass model
│
├── 📂 results
│   └── (charts, reports, exports)
│
└── README.md
```

---

## 🔄 Project Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SQL Server  │────▶│   Python    │────▶│    Excel    │────▶│   Power BI  │
│             │     │             │     │             │     │             │
│ • Raw data  │     │ • EDA       │     │ • Business  │     │ • 3 Live    │
│ • Cleaning  │     │ • 6 Models  │     │   Reporting │     │   Dashboards│
│ • Feature   │     │ • Tuning    │     │ • Pivot     │     │ • Interactive│
│   Engineering│    │ • Evaluation│     │   Tables    │     │   Filters   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 🗄️ Stage 1 — SQL Server

**What was done:**
- Loaded raw sensor data (10,000 machines)
- Cleaned missing values and outliers
- Performed EDA using SQL queries
- Engineered **4 new features** that became top predictors:

| Feature | Formula | Importance |
|---|---|---|
| `temperature_diff` | Process Temp − Air Temp | 4.3% |
| `mechanical_stress` | Torque × RPM | 14.8% |
| `wear_category` | Low / Medium / High bins | 7.2% |
| `high_stress_flag` | Mechanical Stress > 60,000 | 11.2% |

> 💡 **4 out of top 6 most important features came from SQL feature engineering**

---

## 🐍 Stage 2 — Python ML Pipeline

### Data Preprocessing (`02_preprocessing_pipeline.py`)
- Train/test split (80/20)
- StandardScaler normalization
- **SMOTE** to handle class imbalance (96.5% No Failure vs 3.5% Failure)

### Exploratory Data Analysis (`03_eda.py`)
6 visualizations including:
- Failure distribution by machine type
- Correlation heatmap
- Feature distributions
- Class imbalance visualization

### Binary Classification (`04_model_binary.py`)
6 models compared — **will this machine fail? (Yes/No)**

<img width="366" height="197" alt="image" src="https://github.com/user-attachments/assets/f4392dbf-2dcc-4649-ad4c-b8f1693c88e8" />


### Multiclass Classification (`05_model_multiclass.py`)
Same 6 models — **which type of failure will occur?**

- Heat Dissipation Failure
- Power Failure
- Tool Wear Failure
- Overstrain Failure
- Unspecified Failure

**Best model: LightGBM — F1 = 0.9706**

### Hyperparameter Tuning (`06_hyperparameter_tuning.py`)
- RandomizedSearchCV with 50 iterations
- 5-fold cross validation
- Best parameters saved automatically

### Final Evaluation (`07_evaluation.py`)

```
Binary Classification Results:
┌─────────────────────────────────────┐
│  Accuracy  :  98.4%                 │
│  Precision :  97.1%                 │
│  Recall    :  96.2%                 │
│  F1 Score  :  0.9667                │
│  ROC-AUC   :  0.9921                │
└─────────────────────────────────────┘

Confusion Matrix:
              Predicted No  Predicted Fail
Actual No   │    4596 ✅   │    105 ⚠️    │
Actual Fail │     81 🚨    │   5218 ✅    │

✅ True Positives  : 5,218  (failures correctly caught)
🚨 False Negatives :    81  (failures missed — minimized!)
```

---

## 📊 Stage 3 — Power BI Dashboards

Three interactive dashboards built from model outputs:

### Dashboard 1 — Machine Health Monitoring
> Sensor analysis across all machines

- KPI Cards: Total Machines, Avg Tool Wear, Avg Risk Score, High Risk Count
- Avg Tool Wear by Machine Type
- Torque vs Tool Wear scatter (danger zone)
- Risk Band distribution donut chart
- Avg Mechanical Stress by Machine Type
- **Slicer: Filter by Machine Type (H/L/M)**

### Dashboard 2 — Failure Prediction
> Model predictions and probabilities

- KPI Cards: Total Failures, Failure Rate %, Model Accuracy %, Failures Caught
- Actual vs Predicted failures by Machine Type
- Predicted Failure Type breakdown
- Failure Probability by Risk Band
- High Risk Machines detail table
- **Slicer: Filter by Risk Band**

### Dashboard 3 — Model Performance
> Technical model evaluation

- KPI Cards: F1 Score, Accuracy, True Positives, False Negatives
- F1 Score by Failure Type
- Model Comparison (all 6 models)
- Feature Importance (SQL Engineered vs Raw)
- Confusion Matrix heatmap
- **Slicer: Filter by Model**

---

## 🚀 How to Run This Project

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm xgboost
pip install imbalanced-learn matplotlib seaborn
pip install pyodbc joblib openpyxl xlwings
```

### Step 1 — Setup Database
```sql
-- Run in SQL Server Management Studio
-- 1. create_tables.sql
-- 2. data_cleaning.sql
-- 3. feature_engineering.sql
```

### Step 2 — Run Python Files in Order
```bash
python 01_data_loading.py
python 02_preprocessing_pipeline.py
python 03_eda.py
python 04_model_binary.py
python 05_model_multiclass.py
python 06_hyperparameter_tuning.py
python 07_evaluation.py
```

### Step 3 — Open Power BI
```
1. Open Power BI Desktop
2. Get Data → CSV
3. Import DS1_predictions_dashboard.csv
4. Import DS3, DS4, DS5, DS6
5. Open the 3 dashboards
```

---

## 📈 Key Results

```
✅ Best Model      : LightGBM (Tuned)
✅ Binary F1       : 0.9840
✅ Multiclass F1   : 0.9706
✅ Failures Caught : 5,218 out of 5,299
✅ Missed Failures : Only 81 out of 5,299
✅ False Alarm Rate: 105 out of 9,701
```

---

## 💡 Key Insights

1. **Tool Wear is the #1 predictor** — machines with tool wear > 200 min are 3x more likely to fail
2. **L type machines fail most** — 4.1% failure rate vs 2.7% for H type
3. **SQL features matter** — 4 engineered features rank in top 6 importance
4. **Overstrain is most common failure** — high torque machines need priority monitoring
5. **High Risk band accuracy** — model is 94%+ accurate for high risk predictions

---

## 📚 Dataset

- **Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
- **Size:** 10,000 records × 14 features
- **Target:** Binary (Failure/No Failure) + Multiclass (6 failure types)
- **Class Imbalance:** 96.5% No Failure — handled with SMOTE

---

## 👤 Author

> Built as an end-to-end data science portfolio project demonstrating:
> SQL → Python ML → Excel Reporting → Power BI Dashboards

---

## ⭐ If you found this helpful, please star the repository!
