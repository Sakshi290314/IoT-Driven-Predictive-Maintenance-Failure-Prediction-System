# ============================================================
# 05_model_multiclass.py — Multiclass Classification
# Goal: Predict WHICH TYPE of failure will happen
#       (Heat Dissipation / Power / Tool Wear / Overstrain / Random / No Failure)
# ============================================================

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib

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
# This time target = failure_type (6 classes)
# Drop target (binary) to prevent leakage — same rule as file 04!
# Drop udi, product_id — just ID numbers, no value

DROP_COLS = ["udi", "product_id", "target"]   # ← drop binary target this time
TARGET    = "failure_type"

X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y = df[TARGET]

# ── Clean failure_type column ───────────────────────────────
# Remove any None/NaN rows in target (safety check)
mask    = y.notna()
X       = X[mask]
y       = y[mask]
print(f"\n   Failure types found: {y.unique().tolist()}")

# ── Encode machine_type: L→0, M→1, H→2
if "machine_type" in X.columns:
    X["machine_type"] = X["machine_type"].str.upper().map({"L":0,"M":1,"H":2})

# Encode wear_category if text
if "wear_category" in X.columns and X["wear_category"].dtype == object:
    le_wear = LabelEncoder()
    X["wear_category"] = le_wear.fit_transform(X["wear_category"])

# Encode failure_type labels → numbers
# "No Failure"→0, "Heat Dissipation Failure"→1 etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_   # save original names for plots

print(f"\n📐 Features      : {X.columns.tolist()}")
print(f"🎯 Target        : '{TARGET}'")
print(f"\n🏷️  Class Mapping :")
for i, name in enumerate(class_names):
    count = (y == name).sum()
    name_str = str(name) if name is not None else "Unknown"
    print(f"   {i} -> {name_str:<30} ({int(count):,} samples)")


# ============================================================
# STEP 3 — TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded      # keeps class ratios in both splits
)
print(f"\n✂️  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")


# ============================================================
# STEP 4 — SCALE + SMOTE
# ============================================================
# Same logic as file 04 but now SMOTE balances 6 classes
# instead of 2

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"\n⚖️  After SMOTE — Train shape: {X_train_bal.shape}")
print(f"   Class counts after SMOTE:")
for cls, cnt in zip(*np.unique(y_train_bal, return_counts=True)):
    print(f"   {cls} ({class_names[cls]}): {cnt:,}")


# ============================================================
# STEP 5 — TRAIN 6 MODELS & COMPARE
# ============================================================
# For multiclass we use average='weighted' in metrics
# WHY weighted?
# → Each class has different number of samples
# → Weighted average gives more importance to bigger classes
# → More fair than simple average

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(random_state=42),
    "XGBoost"             : XGBClassifier(random_state=42,
                                          eval_metric="mlogloss",
                                          num_class=len(class_names)),
    "LightGBM"            : LGBMClassifier(random_state=42, verbose=-1),
}

print("\n" + "="*65)
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("="*65)

results        = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({"Model": name, "Accuracy": acc,
                    "Precision": prec, "Recall": rec, "F1 Score": f1})
    trained_models[name] = model
    print(f"  {name:<22} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

print("="*65)

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
best_name  = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
y_pred_best = best_model.predict(X_test_scaled)

print(f"\n🏆 Best Model: {best_name}  (F1 = {results_df.iloc[0]['F1 Score']:.4f})")


# ============================================================
# STEP 6 — PLOT 1: Model Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_title("Model Comparison — Multiclass Classification (sorted by F1)",
             fontsize=13, fontweight="bold")

x      = np.arange(len(results_df))
width  = 0.2
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for i, (metric, color) in enumerate(zip(metrics, PALETTE)):
    bars = ax.bar(x + i*width, results_df[metric], width,
                  label=metric, color=color, alpha=0.85, edgecolor="none")

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score")

# Start Y axis at 0.80 so small differences are clearly visible
min_val = results_df[["Accuracy","Precision","Recall","F1 Score"]].min().min()
ax.set_ylim(max(0.75, min_val - 0.05), 1.02)
ax.legend(fontsize=9)

# Add value labels ON TOP of each bar so exact numbers are visible
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.001,
            f"{height:.3f}", ha="center", va="bottom",
            fontsize=6.5, color="white", rotation=90)

ax.axhline(0.9, color="white", linestyle="--", alpha=0.4, linewidth=0.8)
ax.text(len(results_df)-0.3, 0.901, "0.90 threshold",
        fontsize=8, color="white", alpha=0.5)

plt.tight_layout()
plt.savefig("multi_plot1_model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 1 saved")


# ============================================================
# STEP 7 — PLOT 2: Confusion Matrix (Best Model)
# ============================================================
# For multiclass the confusion matrix is 6x6
# Each row = actual class
# Each column = predicted class
# Diagonal = correct predictions ✅
# Off-diagonal = wrong predictions ❌
# You want a bright diagonal and dark everywhere else!

cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5, linecolor="#0f1117",
            annot_kws={"size": 11, "weight": "bold"}, ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}\n"
             f"(Diagonal = correct | Off-diagonal = wrong predictions)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Actual Failure Type", fontsize=11)
ax.set_xlabel("Predicted Failure Type", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig("multi_plot2_confusion_matrix.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 2 saved")
print("📌 Bright diagonal = model predicts each failure type correctly!")


# ============================================================
# STEP 8 — PLOT 3: Per-Class F1 Score
# ============================================================
# Overall F1 is one number but this shows
# HOW WELL the model predicts EACH failure type separately
# Great interview talking point:
# "My model was excellent at Tool Wear but struggled with Random Failures
#  because Random Failures have very few samples even after SMOTE"

report      = classification_report(y_test, y_pred_best,
                                    target_names=class_names,
                                    output_dict=True,
                                    zero_division=0)
per_class   = pd.DataFrame(report).T.loc[class_names, ["precision","recall","f1-score"]]

fig, ax = plt.subplots(figsize=(11, 5))
x      = np.arange(len(class_names))
width  = 0.25

ax.bar(x - width, per_class["precision"], width,
       label="Precision", color=PALETTE[0], alpha=0.85, edgecolor="none")
ax.bar(x,         per_class["recall"],    width,
       label="Recall",    color=PALETTE[1], alpha=0.85, edgecolor="none")
ax.bar(x + width, per_class["f1-score"],  width,
       label="F1 Score",  color=PALETTE[2], alpha=0.85, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score")

# Start Y near min so differences between classes are obvious
min_score = per_class[["precision","recall","f1-score"]].min().min()
ax.set_ylim(max(0.70, min_score - 0.08), 1.05)

ax.set_title(f"Per-Class Metrics — {best_name}\n"
             f"(How well does the model predict EACH failure type?)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.9, color="white", linestyle="--", alpha=0.4, linewidth=0.8)

# Add value labels on top of each bar
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.002,
            f"{height:.2f}", ha="center", va="bottom",
            fontsize=8, color="white", rotation=90)

plt.tight_layout()
plt.savefig("multi_plot3_per_class_metrics.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 3 saved")
print("📌 Lower scores on rare failure types is normal — fewer training samples!")


# ============================================================
# STEP 9 — PLOT 4: Feature Importance
# ============================================================
if hasattr(best_model, "feature_importances_"):
    imp_df = pd.DataFrame({
        "Feature"   : X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                   color=PALETTE[0], edgecolor="none", alpha=0.85)

    top3 = imp_df["Importance"].nlargest(3).index
    for bar, idx in zip(bars, imp_df.index):
        if idx in top3:
            bar.set_color(HIGHLIGHT)

    ax.set_title(f"Feature Importance — {best_name}\n"
                 f"(Pink = Top 3 most important features)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("multi_plot4_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print("✅ Plot 4 saved")
    print(f"\n📌 Top 3 features: {imp_df.nlargest(3,'Importance')['Feature'].tolist()}")


# ============================================================
# STEP 10 — DETAILED CLASSIFICATION REPORT
# ============================================================
print("\n" + "="*60)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("="*60)
print(classification_report(y_test, y_pred_best,
      target_names=class_names, zero_division=0))


# ============================================================
# STEP 11 — SAVE MODEL
# ============================================================
joblib.dump(best_model, "best_multiclass_model.pkl")
joblib.dump(scaler,     "multiclass_scaler.pkl")
joblib.dump(le,         "label_encoder.pkl")

print("✅ Model saved        → best_multiclass_model.pkl")
print("✅ Scaler saved       → multiclass_scaler.pkl")
print("✅ Label encoder saved → label_encoder.pkl")
print("\n   To predict on new data:")
print("   model = joblib.load('best_multiclass_model.pkl')")
print("   le    = joblib.load('label_encoder.pkl')")
print("   pred  = le.inverse_transform(model.predict(new_data))")
print("   → gives you back 'Tool Wear Failure', 'No Failure' etc.")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("  MULTICLASS MODEL — FINAL SUMMARY")
print("="*55)
for _, row in results_df.iterrows():
    star = " 🏆" if row["Model"] == best_name else ""
    print(f"  {row['Model']:<22} F1={row['F1 Score']:.4f}{star}")
print("="*55)
print("\n🎯 YOUR PROJECT IS NOW COMPLETE!")
print("   File 04 → Binary    → Will machine fail?")
print("   File 05 → Multiclass → Which type of failure?")
print("\n🚀 NEXT → Run 06_hyperparameter_tuning.py to improve scores!")
