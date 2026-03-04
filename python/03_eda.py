# ============================================================
# 03_eda.py — EDA for Machine Predictive Maintenance
# ============================================================

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#3a3d52",   "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#a0a0b0",      "ytick.color": "#a0a0b0",
    "text.color": "#e0e0e0",       "grid.color": "#2a2d3e",
    "grid.linestyle": "--",        "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",  "axes.titlesize": 13,
    "axes.titleweight": "bold",    "axes.titlepad": 12,
})
PALETTE   = ["#4fc3f7", "#f06292", "#aed581", "#ffb74d", "#ce93d8", "#80cbc4"]
ACCENT    = "#4fc3f7"
HIGHLIGHT = "#f06292"


# ============================================================
# STEP 1 — CONNECT TO SQL & LOAD DATA
# ❗ Only change SERVER and DATABASE below
# ============================================================
SERVER   = ".\SQLEXPRESS"    # e.g. "DESKTOP-ABC123\\SQLEXPRESS"
DATABASE = "predictive_maintenance_db"  # e.g. "PredictiveMaintenance"

conn = pyodbc.connect(
    f"DRIVER={{SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
    f"Connection Timeout=30;"
)

df = pd.read_sql("SELECT * FROM vw_feature_engineered_data", conn)
conn.close()

# Clean column names → "Air temperature [K]" becomes "air_temperature_k"
df.columns = (
    df.columns.str.strip().str.lower()
    .str.replace(r"[\s\[\]\(\)/]+", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)

print("✅ Data loaded from SQL!")
print(f"   Shape   : {df.shape}")
print(f"   Columns : {df.columns.tolist()}")


# ============================================================
# STEP 2 — BASIC OVERVIEW
# ============================================================
print("\n" + "="*55)
print(f"  Rows           : {df.shape[0]:,}")
print(f"  Columns        : {df.shape[1]}")
print(f"  Missing values : {df.isna().sum().sum()}")
print(f"  Duplicates     : {df.duplicated().sum()}")
print(f"\n── Failure Type Counts ──")
print(df["failure_type"].value_counts())
print(f"\n── Numerical Summary ────")
print(df.describe().round(2))


# ============================================================
# STEP 3 — PLOT 1: Class Imbalance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PLOT 1 — Class Imbalance (Why we need SMOTE)",
             fontsize=14, fontweight="bold", color="white")

target_counts = df["target"].value_counts()
bars = axes[0].bar(["No Failure (0)", "Failure (1)"],
                   target_counts.values,
                   color=[ACCENT, HIGHLIGHT], edgecolor="none", width=0.5)
for bar, val in zip(bars, target_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 80,
                 f"{val:,}\n({val/len(df)*100:.1f}%)",
                 ha="center", fontsize=11, color="white")
axes[0].set_title("Binary — Fail vs No Fail")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, max(target_counts.values) * 1.2)

ft_counts = df["failure_type"].value_counts()
bars2 = axes[1].barh(ft_counts.index, ft_counts.values,
                     color=PALETTE, edgecolor="none")
for bar, val in zip(bars2, ft_counts.values):
    axes[1].text(val + 20, bar.get_y() + bar.get_height()/2,
                 f"{val:,}", va="center", fontsize=9, color="white")
axes[1].set_title("Multi-Class — Failure Types")
axes[1].set_xlabel("Count")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("plot1_class_imbalance.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("📌 ~96% No Failure → severe imbalance → that's why SMOTE is used!")


# ============================================================
# STEP 4 — PLOT 2: Feature Distributions
# ============================================================
num_cols = [c for c in ["air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min"] if c in df.columns]

fig, axes = plt.subplots(2, len(num_cols), figsize=(18, 8))
fig.suptitle("PLOT 2 — Feature Distributions",
             fontsize=14, fontweight="bold", color="white")

for i, col in enumerate(num_cols):
    color = PALETTE[i % len(PALETTE)]
    axes[0, i].hist(df[col], bins=40, color=color, alpha=0.85, edgecolor="none")
    axes[0, i].set_title(col.replace("_", "\n"), fontsize=9)
    axes[0, i].set_ylabel("Frequency", fontsize=8)
    bp = axes[1, i].boxplot(df[col], patch_artist=True,
                             medianprops=dict(color="white", linewidth=2),
                             boxprops=dict(facecolor=color, alpha=0.7),
                             whiskerprops=dict(color="#aaa"), capprops=dict(color="#aaa"),
                             flierprops=dict(marker="o", markerfacecolor=HIGHLIGHT,
                                             markersize=3, alpha=0.5))
    axes[1, i].set_title("Outliers", fontsize=9)
    axes[1, i].set_xticks([])

plt.tight_layout()
plt.savefig("plot2_distributions.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()


# ============================================================
# STEP 5 — PLOT 3: Correlation Heatmap
# ============================================================
df_corr = df.copy()
if "type" in df_corr.columns:
    df_corr["type_encoded"] = df_corr["type"].str.lower().map({"l":0,"m":1,"h":2})

corr_cols = [c for c in num_cols + ["target","type_encoded"] if c in df_corr.columns]
corr_matrix = df_corr[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("PLOT 3 — Correlation Heatmap",
             fontsize=14, fontweight="bold", color="white")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, linecolor="#0f1117",
            annot_kws={"size": 10}, ax=ax, vmin=-1, vmax=1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("plot3_correlation.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("📌 Torque & RPM negatively correlated (−0.88) — makes physical sense!")


# ============================================================
# STEP 6 — PLOT 4: Features by Failure Type
# ============================================================
failure_order = df["failure_type"].value_counts().index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("PLOT 4 — Feature Values by Failure Type",
             fontsize=14, fontweight="bold", color="white")

for i, col in enumerate(num_cols):
    ax = axes[i//3, i%3]
    data = [df[df["failure_type"]==ft][col].values for ft in failure_order]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#888"), capprops=dict(color="#888"),
                    flierprops=dict(marker="o", markerfacecolor=HIGHLIGHT,
                                    markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(failure_order)+1))
    ax.set_xticklabels([ft.replace(" Failure","").replace(" ","\n")
                        for ft in failure_order], fontsize=7)
    ax.set_title(col.replace("_"," "), fontsize=10)

axes[1,2].axis("off")
axes[1,2].text(0.5, 0.5,
    "📌 KEY FINDINGS:\n\n• Overstrain → high torque,\n  low RPM\n\n"
    "• Tool Wear → high tool\n  wear minutes\n\n"
    "• Heat Dissipation → higher\n  temperatures\n\n"
    "• Power Failure → extreme RPM",
    transform=axes[1,2].transAxes, ha="center", va="center",
    fontsize=11, color="#e0e0e0",
    bbox=dict(boxstyle="round,pad=0.8", facecolor="#1a1d2e",
              edgecolor=ACCENT, linewidth=1.5))
plt.tight_layout()
plt.savefig("plot4_features_by_failure.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()


# ============================================================
# STEP 7 — PLOT 5: Failure Rate by Machine Type
# ============================================================
if "type" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PLOT 5 — Failure Rate by Machine Quality (L/M/H)",
                 fontsize=14, fontweight="bold", color="white")

    pivot = (df.groupby(["type","failure_type"]).size()
               .reset_index(name="count")
               .pivot(index="type", columns="failure_type", values="count")
               .fillna(0))
    pivot.plot(kind="bar", ax=axes[0], color=PALETTE,
               edgecolor="none", width=0.7)
    axes[0].set_title("Failure Count by Type")
    axes[0].set_xlabel("Type (L=Low, M=Medium, H=High)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=7)

    failure_rate = df.groupby("type")["target"].mean() * 100
    bars = axes[1].bar(failure_rate.index, failure_rate.values,
                       color=PALETTE[:len(failure_rate)], edgecolor="none", width=0.5)
    for bar, val in zip(bars, failure_rate.values):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f"{val:.2f}%", ha="center", fontsize=12, color="white")
    axes[1].set_title("Failure RATE by Type (%)")
    axes[1].set_ylabel("Failure Rate (%)")
    axes[1].set_ylim(0, max(failure_rate.values) * 1.3)
    plt.tight_layout()
    plt.savefig("plot5_failure_by_type.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print("📌 Low quality (L) machines fail more often than High quality (H)!")


# ============================================================
# STEP 8 — ENGINEERED FEATURES (already in your SQL view!)
# ============================================================
# Your SQL view vw_feature_engineered_data already has these.
# This just visualises them. If they don't exist we create them.
if "temp_diff" not in df.columns:
    df["temp_diff"] = df["process_temperature_k"] - df["air_temperature_k"]
if "rpm_torque_ratio" not in df.columns:
    df["rpm_torque_ratio"] = df["torque_nm"] / df["rotational_speed_rpm"]

eng_cols = ["temp_diff", "rpm_torque_ratio"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PLOT 6 — Engineered Features vs Failure Type",
             fontsize=14, fontweight="bold", color="white")
for ax, col in zip(axes, eng_cols):
    for ft, color in zip(failure_order, PALETTE):
        ax.hist(df[df["failure_type"]==ft][col],
                bins=30, alpha=0.6, label=ft, color=color, density=True)
    ax.set_title(col.replace("_"," "), fontsize=11)
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("plot6_engineered.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()


# ============================================================
# STEP 9 — INTERVIEW SUMMARY
# ============================================================
print("\n" + "="*55)
print("  EDA COMPLETE — INTERVIEW SUMMARY")
print("="*55)
print(f"  · Total records        : {len(df):,}")
print(f"  · Features             : {df.shape[1]}")
print(f"  · Missing values       : {df.isna().sum().sum()}")
print(f"  · Failure rate         : {df['target'].mean()*100:.2f}%")
print(f"  · Most common failure  : {df[df['target']==1]['failure_type'].value_counts().index[0]}")
print(f"  · Rarest failure       : {df[df['target']==1]['failure_type'].value_counts().index[-1]}")
print(f"  · Strongest correlation: Torque vs RPM (−0.88)")
print(f"  · Imbalance fix        : SMOTE in preprocessing pipeline")
print("="*55)
print("🚀 NEXT STEP → Run 04_model_binary.py")
