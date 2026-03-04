# ============================================================
# 06_hyperparameter_tuning.py — Hyperparameter Tuning
# Goal: Squeeze the best possible performance from LightGBM
#       (our best model from files 04 and 05)
# ============================================================
# WHAT IS HYPERPARAMETER TUNING?
# Every model has "settings" (hyperparameters) that control how it learns. Like tuning a guitar — same guitar, but adjusting the strings gives better sound.
# Example LightGBM hyperparameters:
#   n_estimators  = how many trees to build (100? 500? 1000?)
#   max_depth     = how deep each tree grows (3? 5? 10?)
#   learning_rate = how fast it learns (0.01? 0.1? 0.3?)
#
# We try many combinations and find which gives best F1 score
# ============================================================

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix, accuracy_score)
from imblearn.over_sampling import SMOTE
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
# STEP 2 — PREPARE DATA (same as files 04 & 05)
# We tune for BOTH tasks in one file
# ============================================================
DROP_ALWAYS = ["udi", "product_id"]

def prepare_data(df, task="binary"):
    """Prepare X, y for binary or multiclass task"""
    if task == "binary":
        target = "target"
        drop   = DROP_ALWAYS + ["failure_type"]
    else:
        target = "failure_type"
        drop   = DROP_ALWAYS + ["target"]

    X = df.drop(columns=[target] + drop, errors="ignore").copy()
    y = df[target].copy()

    # Clean nulls in target
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Encode machine_type
    if "machine_type" in X.columns:
        X["machine_type"] = X["machine_type"].str.upper().map(
            {"L": 0, "M": 1, "H": 2})

    # Encode wear_category
    if "wear_category" in X.columns and X["wear_category"].dtype == object:
        le_w = LabelEncoder()
        X["wear_category"] = le_w.fit_transform(X["wear_category"])

    # Encode multiclass target
    le = None
    if task == "multiclass":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), index=y.index)

    return X, y, le


# ============================================================
# STEP 3 — WHAT IS RANDOMIZEDSEARCHCV?
# ============================================================
# Two ways to tune:
#
# GridSearchCV       → tries EVERY combination (slow!)
#   e.g. 5 values × 5 values × 5 values = 125 combinations
#
# RandomizedSearchCV → tries RANDOM combinations (fast!)
#   You say "try 50 random combinations" → finds good enough answer
#   much faster. Perfect for interview projects!
#
# Cross Validation (cv=5):
#   Instead of one train/test split, we split data 5 ways
#   Train on 4 parts, test on 1 part → repeat 5 times
#   Take average score → more reliable than single split
#
#   [  1  |  2  |  3  |  4  |  5  ]
#   [train|train|train|train| TEST ]  → score 1
#   [train|train|train| TEST|train ]  → score 2
#   [train|train| TEST|train|train ]  → score 3
#   ... average all 5 scores = final CV score

# ============================================================
# STEP 4 — DEFINE HYPERPARAMETER SEARCH SPACE
# ============================================================
# These are the "knobs" we're tuning on LightGBM
# We give a RANGE for each knob and RandomizedSearch picks combos

param_grid = {
    # How many trees — more trees = better but slower
    "n_estimators"    : [100, 200, 300, 500, 700, 1000],

    # How deep each tree grows — deeper = learns more detail
    # but too deep = overfitting (memorizes instead of learning)
    "max_depth"       : [3, 5, 7, 10, 15, -1],

    # How fast the model learns — lower = more careful = better
    # but needs more trees to compensate
    "learning_rate"   : [0.01, 0.05, 0.1, 0.2, 0.3],

    # What fraction of features to use per tree
    # Adds randomness → prevents overfitting
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],

    # What fraction of data to use per tree
    "subsample"       : [0.6, 0.7, 0.8, 0.9, 1.0],

    # Minimum samples needed to create a leaf node
    # Higher = simpler tree = less overfitting
    "min_child_samples": [5, 10, 20, 30, 50],

    # Regularization — penalizes complex trees
    "reg_alpha"       : [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda"      : [0, 0.01, 0.1, 0.5, 1.0],
}

print("\n📐 Search space size:")
total = 1
for k, v in param_grid.items():
    total *= len(v)
    print(f"   {k:<20}: {len(v)} options")
print(f"\n   Total combinations : {total:,}")
print(f"   We will try        : 50 random combos (RandomizedSearchCV)")
print(f"   With 5-fold CV     : 50 × 5 = 250 model fits total")


# ============================================================
# STEP 5 — TUNE FOR BINARY CLASSIFICATION
# ============================================================
print("\n" + "="*55)
print("  TUNING TASK 1 — BINARY (Fail vs No Fail)")
print("="*55)

X_bin, y_bin, _ = prepare_data(df, task="binary")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

scaler_bin     = StandardScaler()
X_tr_sc        = scaler_bin.fit_transform(X_tr)
X_te_sc        = scaler_bin.transform(X_te)

smote          = SMOTE(random_state=42)
X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sc, y_tr)

# Default model score first (baseline)
default_lgbm = LGBMClassifier(random_state=42, verbose=-1)
default_lgbm.fit(X_tr_bal, y_tr_bal)
default_f1   = f1_score(y_te, default_lgbm.predict(X_te_sc),
                        average="weighted", zero_division=0)
print(f"\n📊 Default LightGBM F1  : {default_f1:.4f}  ← before tuning")

# Tune!
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search_bin = RandomizedSearchCV(
    estimator  = LGBMClassifier(random_state=42, verbose=-1),
    param_distributions = param_grid,
    n_iter     = 50,          # try 50 random combinations
    cv         = cv,          # 5-fold cross validation
    scoring    = "f1_weighted",
    n_jobs     = -1,          # use all CPU cores
    random_state = 42,
    verbose    = 1
)

print("\n🔍 Searching best hyperparameters for Binary task...")
start = time.time()
search_bin.fit(X_tr_bal, y_tr_bal)
elapsed = time.time() - start

# Best model score
best_bin_pred = search_bin.best_estimator_.predict(X_te_sc)
tuned_f1_bin  = f1_score(y_te, best_bin_pred,
                          average="weighted", zero_division=0)

print(f"\n⏱️  Time taken          : {elapsed:.1f} seconds")
print(f"📊 Default LightGBM F1 : {default_f1:.4f}")
print(f"📊 Tuned LightGBM F1   : {tuned_f1_bin:.4f}")
print(f"📈 Improvement         : +{(tuned_f1_bin - default_f1)*100:.2f}%")
print(f"\n🏆 Best Parameters (Binary):")
for k, v in search_bin.best_params_.items():
    print(f"   {k:<22}: {v}")


# ============================================================
# STEP 6 — TUNE FOR MULTICLASS CLASSIFICATION
# ============================================================
print("\n" + "="*55)
print("  TUNING TASK 2 — MULTICLASS (Failure Type)")
print("="*55)

X_mc, y_mc, le_mc = prepare_data(df, task="multiclass")
class_names = le_mc.classes_

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_mc, y_mc, test_size=0.2, random_state=42, stratify=y_mc)

scaler_mc      = StandardScaler()
X_tr2_sc       = scaler_mc.fit_transform(X_tr2)
X_te2_sc       = scaler_mc.transform(X_te2)

smote2         = SMOTE(random_state=42)
X_tr2_bal, y_tr2_bal = smote2.fit_resample(X_tr2_sc, y_tr2)

# Default model score
default_lgbm2 = LGBMClassifier(random_state=42, verbose=-1)
default_lgbm2.fit(X_tr2_bal, y_tr2_bal)
default_f1_mc = f1_score(y_te2, default_lgbm2.predict(X_te2_sc),
                          average="weighted", zero_division=0)
print(f"\n📊 Default LightGBM F1  : {default_f1_mc:.4f}  ← before tuning")

search_mc = RandomizedSearchCV(
    estimator  = LGBMClassifier(random_state=42, verbose=-1),
    param_distributions = param_grid,
    n_iter     = 50,
    cv         = cv,
    scoring    = "f1_weighted",
    n_jobs     = -1,
    random_state = 42,
    verbose    = 1
)

print("\n🔍 Searching best hyperparameters for Multiclass task...")
start2 = time.time()
search_mc.fit(X_tr2_bal, y_tr2_bal)
elapsed2 = time.time() - start2

best_mc_pred  = search_mc.best_estimator_.predict(X_te2_sc)
tuned_f1_mc   = f1_score(y_te2, best_mc_pred,
                          average="weighted", zero_division=0)

print(f"\n⏱️  Time taken          : {elapsed2:.1f} seconds")
print(f"📊 Default LightGBM F1 : {default_f1_mc:.4f}")
print(f"📊 Tuned LightGBM F1   : {tuned_f1_mc:.4f}")
print(f"📈 Improvement         : +{(tuned_f1_mc - default_f1_mc)*100:.2f}%")
print(f"\n🏆 Best Parameters (Multiclass):")
for k, v in search_mc.best_params_.items():
    print(f"   {k:<22}: {v}")


# ============================================================
# STEP 7 — PLOT 1: Before vs After Tuning Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Hyperparameter Tuning — Before vs After",
             fontsize=14, fontweight="bold")

for ax, task, before, after in zip(
    axes,
    ["Binary Classification", "Multiclass Classification"],
    [default_f1, default_f1_mc],
    [tuned_f1_bin, tuned_f1_mc]
):
    bars = ax.bar(["Default\nLightGBM", "Tuned\nLightGBM"],
                  [before, after],
                  color=[ACCENT, HIGHLIGHT],
                  edgecolor="none", width=0.4)

    # Value labels
    for bar, val in zip(bars, [before, after]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="white")

    # Improvement arrow
    improvement = (after - before) * 100
    ax.annotate(f"+{improvement:.2f}%\nimprovement",
                xy=(1, after), xytext=(0.5, (before + after)/2),
                fontsize=11, color="#aed581", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#aed581", lw=1.5))

    ax.set_title(task, fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score (Weighted)")
    ymin = min(before, after) - 0.02
    ax.set_ylim(max(0.8, ymin), min(1.0, max(before, after) + 0.03))

plt.tight_layout()
plt.savefig("tuning_plot1_before_after.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 1 saved")


# ============================================================
# STEP 8 — PLOT 2: Top 20 Hyperparameter Combinations
# ============================================================
# Shows which combinations the search tried and their scores
# Great for understanding which parameters matter most

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Top 20 Hyperparameter Combinations Tried",
             fontsize=14, fontweight="bold")

for ax, search, title in zip(
    axes,
    [search_bin, search_mc],
    ["Binary", "Multiclass"]
):
    cv_results = pd.DataFrame(search.cv_results_)
    top20      = cv_results.nlargest(20, "mean_test_score")

    ax.barh(range(20), top20["mean_test_score"],
            color=PALETTE[0], alpha=0.8, edgecolor="none")
    ax.errorbar(top20["mean_test_score"], range(20),
                xerr=top20["std_test_score"],
                fmt="none", color=HIGHLIGHT, capsize=3, linewidth=1.5)

    ax.set_yticks(range(20))
    ax.set_yticklabels([f"Combo {i+1}" for i in range(20)], fontsize=8)
    ax.set_xlabel("CV F1 Score")
    ax.set_title(f"{title} — Top 20 Combos\n(error bars = std across 5 folds)",
                 fontsize=11)

    xmin = top20["mean_test_score"].min() - 0.01
    ax.set_xlim(max(0.85, xmin), top20["mean_test_score"].max() + 0.01)
    ax.axvline(top20["mean_test_score"].iloc[0], color=HIGHLIGHT,
               linestyle="--", alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig("tuning_plot2_top_combinations.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 2 saved")


# ============================================================
# STEP 9 — PLOT 3: Learning Curve
# ============================================================
# WHAT IS A LEARNING CURVE?
# Shows model performance as we give it MORE training data
# Good model:  train score ≈ validation score (both high)
# Overfitting: train score high, validation score low
# Underfitting: both scores low

from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Learning Curves — Does model improve with more data?",
             fontsize=14, fontweight="bold")

for ax, model, X_tr_data, y_tr_data, title in zip(
    axes,
    [search_bin.best_estimator_, search_mc.best_estimator_],
    [X_tr_bal, X_tr2_bal],
    [y_tr_bal, y_tr2_bal],
    ["Binary", "Multiclass"]
):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_tr_data, y_tr_data,
        cv=3,
        scoring="f1_weighted",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, color=ACCENT,
            lw=2, label="Training Score", marker="o")
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color=ACCENT)

    ax.plot(train_sizes, val_mean, color=HIGHLIGHT,
            lw=2, label="Validation Score", marker="s")
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color=HIGHLIGHT)

    ax.set_title(f"{title} — Learning Curve", fontsize=11)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("F1 Score")
    ax.legend(fontsize=10)
    ax.set_ylim(0.7, 1.05)
    ax.axhline(0.9, color="white", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("tuning_plot3_learning_curves.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Plot 3 saved")
print("📌 If train ≈ validation → good fit!")
print("   If train >> validation → overfitting → need more regularization")
print("   If both low → underfitting → need more complex model")


# ============================================================
# STEP 10 — CROSS VALIDATION SCORE (Final check)
# ============================================================
# CV score = most reliable performance estimate
# Tests model on 5 different data splits → average = true score

print("\n" + "="*55)
print("  CROSS VALIDATION — FINAL RELIABILITY CHECK")
print("="*55)

cv_scores_bin = cross_val_score(
    search_bin.best_estimator_,
    X_tr_bal, y_tr_bal,
    cv=5, scoring="f1_weighted", n_jobs=-1
)
cv_scores_mc = cross_val_score(
    search_mc.best_estimator_,
    X_tr2_bal, y_tr2_bal,
    cv=5, scoring="f1_weighted", n_jobs=-1
)

print(f"\n  Binary    CV scores : {[round(s,4) for s in cv_scores_bin]}")
print(f"  Binary    Mean ± Std: {cv_scores_bin.mean():.4f} ± {cv_scores_bin.std():.4f}")
print(f"\n  Multiclass CV scores: {[round(s,4) for s in cv_scores_mc]}")
print(f"  Multiclass Mean±Std : {cv_scores_mc.mean():.4f} ± {cv_scores_mc.std():.4f}")
print(f"\n  📌 Low std = consistent model = trustworthy!")


# ============================================================
# STEP 11 — SAVE TUNED MODELS
# ============================================================
joblib.dump(search_bin.best_estimator_, "tuned_binary_model.pkl")
joblib.dump(search_mc.best_estimator_,  "tuned_multiclass_model.pkl")
joblib.dump(scaler_bin,                 "tuned_binary_scaler.pkl")
joblib.dump(scaler_mc,                  "tuned_multiclass_scaler.pkl")
joblib.dump(le_mc,                      "tuned_label_encoder.pkl")

print("\n✅ Saved: tuned_binary_model.pkl")
print("✅ Saved: tuned_multiclass_model.pkl")
print("✅ Saved: tuned_binary_scaler.pkl")
print("✅ Saved: tuned_multiclass_scaler.pkl")
print("✅ Saved: tuned_label_encoder.pkl")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("  TUNING COMPLETE — FINAL SUMMARY")
print("="*55)
print(f"  Task         Before    After     Improvement")
print(f"  {'─'*45}")
print(f"  Binary       {default_f1:.4f}    {tuned_f1_bin:.4f}    "
      f"+{(tuned_f1_bin-default_f1)*100:.2f}%")
print(f"  Multiclass   {default_f1_mc:.4f}    {tuned_f1_mc:.4f}    "
      f"+{(tuned_f1_mc-default_f1_mc)*100:.2f}%")
print("="*55)
print("\n🚀 NEXT → Run 07_evaluation.py for final report!")
