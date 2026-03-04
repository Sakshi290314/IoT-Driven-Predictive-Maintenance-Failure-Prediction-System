# ============================================================
# 04_model_binary.py — Binary Classification
# Goal: Predict whether a machine will FAIL or NOT (0 or 1)
# ============================================================

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, RocCurveDisplay)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#3a3d52",   "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#a0a0b0",      "ytick.color": "#a0a0b0",
    "text.color": "#e0e0e0",       "grid.color": "#2a2d3e",
    "grid.linestyle": "--",        "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})
PALETTE   = ["#4fc3f7", "#f06292", "#aed581", "#ffb74d", "#ce93d8", "#80cbc4"]
ACCENT    = "#4fc3f7"
HIGHLIGHT = "#f06292"


# ============================================================
# STEP 1 — LOAD FROM SQL
# ❗ Change only SERVER and DATABASE
# ============================================================
SERVER   = ".\SQLEXPRESS"
DATABASE = "predictive_maintenance_db"

conn = pyodbc.connect(
    f"DRIVER={{SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
    f"Connection Timeout=30;"
)
df = pd.read_sql("SELECT * FROM vw_feature_engineered_data", conn)
conn.close()
print(f"✅ Data loaded — Shape: {df.shape}")


# ============================================================
# STEP 2 — PREPARE FEATURES & TARGET
# ============================================================
# Drop columns that should NOT go into the model:
#   udi, product_id  → just ID numbers, no predictive value
#   failure_type     → this is the OTHER target (for file 05)
#                      keeping it would be CHEATING (data leakage!)
DROP_COLS = ["udi", "product_id", "failure_type"]
TARGET    = "target"

X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y = df[TARGET]

# Encode machine_type: L→0, M→1, H→2
if "machine_type" in X.columns:
    X["machine_type"] = X["machine_type"].str.upper().map({"L":0,"M":1,"H":2})

# Encode wear_category if it's text
if "wear_category" in X.columns and X["wear_category"].dtype == object:
    le = LabelEncoder()
    X["wear_category"] = le.fit_transform(X["wear_category"])

print(f"\n📐 Features used : {X.columns.tolist()}")
print(f"🎯 Target        : '{TARGET}' (0=No Failure, 1=Failure)")
print(f"⚖️  Class balance : {y.value_counts().to_dict()}")


# ============================================================
# STEP 3 — TRAIN / TEST SPLIT
# stratify=y → keeps same class ratio in train and test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")


# ============================================================
# STEP 4 — SCALE + SMOTE
# Scale first → then SMOTE on training data ONLY
# NEVER apply SMOTE to test data (that would be cheating!)
# ============================================================
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled   = scaler.transform(X_test)         # only transform test

smote           = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"\n⚖️  After SMOTE — Train shape : {X_train_bal.shape}")
print(f"   Class balance after SMOTE  : {pd.Series(y_train_bal).value_counts().to_dict()}")


# ============================================================
# STEP 5 — TRAIN MULTIPLE MODELS & COMPARE
# ============================================================
# Why multiple models?
# → Each model has strengths/weaknesses
# → We pick the best one based on F1 score
# → F1 is better than accuracy for imbalanced data
#   (accuracy can be 96% by just predicting "No Failure" always!)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(random_state=42),
    "XGBoost"             : XGBClassifier(random_state=42, eval_metric="logloss"),
    "LightGBM"            : LGBMClassifier(random_state=42, verbose=-1),
}

print("\n" + "="*65)
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("="*65)

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred  = model.predict(X_test_scaled)
    y_prob  = (model.predict_proba(X_test_scaled)[:,1]
               if hasattr(model, "predict_proba") else None)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1 Score": f1, "ROC-AUC": auc
    })
    trained_models[name] = model
    print(f"  {name:<22} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>9.4f}")

print("="*65)

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
best_name  = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"\n🏆 Best Model: {best_name}  (F1 = {results_df.iloc[0]['F1 Score']:.4f})")


# ============================================================
# STEP 6 — PLOT 1: Model Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_title("Model Comparison — Binary Classification (sorted by F1 Score)",
             fontsize=13, fontweight="bold")

x      = np.arange(len(results_df))
width  = 0.18
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

for i, (metric, color) in enumerate(zip(metrics, PALETTE)):
    ax.bar(x + i*width, results_df[metric], width,
           label=metric, color=color, alpha=0.85, edgecolor="none")

ax.set_xticks(x + width*2)
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, loc="upper right")
ax.axhline(0.9, color="white", linestyle="--", alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig("binary_plot1_model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 1 saved")


# ============================================================
# STEP 7 — PLOT 2: Confusion Matrix for Best Model
# ============================================================
# What is a confusion matrix?
# A table showing how many predictions were:
#   True Positive  → predicted Failure,  actually Failure  ✅
#   True Negative  → predicted No Fail,  actually No Fail  ✅
#   False Positive → predicted Failure,  actually No Fail  ❌ (false alarm)
#   False Negative → predicted No Fail,  actually Failure  ❌ (missed failure!)
# For machines, False Negative is DANGEROUS → we missed a real failure!

y_pred_best = best_model.predict(X_test_scaled)
cm          = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            linewidths=0.5, linecolor="#0f1117",
            annot_kws={"size": 14, "weight": "bold"}, ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual", fontsize=11)
ax.set_xlabel("Predicted", fontsize=11)

plt.tight_layout()
plt.savefig("binary_plot2_confusion_matrix.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 2 saved")
print(f"\n📌 False Negatives (missed failures) = {cm[1][0]} ← want this LOW!")
print(f"   False Positives (false alarms)    = {cm[0][1]} ← less critical")


# ============================================================
# STEP 8 — PLOT 3: ROC Curve for Best Model
# ============================================================
# What is ROC-AUC?
# AUC = 1.0 → perfect model
# AUC = 0.5 → random guessing (useless)
# AUC = 0.9+ → excellent for this use case

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_facecolor("#1a1d2e")
fig.patch.set_facecolor("#0f1117")

if hasattr(best_model, "predict_proba"):
    RocCurveDisplay.from_estimator(
        best_model, X_test_scaled, y_test,
        ax=ax, color=ACCENT, lw=2,
        name=best_name
    )
ax.plot([0,1],[0,1], color="#888", linestyle="--", lw=1, label="Random (AUC=0.5)")
ax.set_title(f"ROC Curve — {best_name}", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=10)
ax.tick_params(colors="#a0a0b0")

plt.tight_layout()
plt.savefig("binary_plot3_roc_curve.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 3 saved")


# ============================================================
# STEP 9 — PLOT 4: Feature Importance
# ============================================================
# Which features does the model rely on most?
# This is a KEY interview question:
# "What were your most important features?"

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature"   : X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
                   color=PALETTE[0], edgecolor="none", alpha=0.85)

    # Highlight top 3
    top3_idx = importance_df["Importance"].nlargest(3).index
    for bar, idx in zip(bars, importance_df.index):
        if idx in top3_idx:
            bar.set_color(HIGHLIGHT)

    ax.set_title(f"Feature Importance — {best_name}\n(Pink = Top 3 most important)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("binary_plot4_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print("✅ Plot 4 saved")
    print(f"\n📌 Top 3 features: {importance_df.nlargest(3,'Importance')['Feature'].tolist()}")


# ============================================================
# STEP 10 — DETAILED REPORT for Best Model
# ============================================================
print("\n" + "="*55)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("="*55)
print(classification_report(y_test, y_pred_best,
      target_names=["No Failure", "Failure"]))


# ============================================================
# STEP 11 — SAVE MODEL & SCALER
# ============================================================
import joblib
joblib.dump(best_model, "best_binary_model.pkl")
joblib.dump(scaler,     "binary_scaler.pkl")
print("✅ Model saved → best_binary_model.pkl")
print("✅ Scaler saved → binary_scaler.pkl")
print("   (Load these in production with joblib.load())")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("  BINARY MODEL — FINAL SUMMARY")
print("="*55)
for _, row in results_df.iterrows():
    star = " 🏆" if row["Model"] == best_name else ""
    print(f"  {row['Model']:<22} F1={row['F1 Score']:.4f}  AUC={row['ROC-AUC']:.4f}{star}")
print("="*55)
print("\n🚀 NEXT → Run 05_model_multiclass.py")
print("   (Predict WHICH TYPE of failure it is)")
