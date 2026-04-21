"""
================================================================================
  Heart Disease Classification: XGBoost Baseline vs. Optuna-Optimized Pipeline
================================================================================
  Dataset   : Cleveland Heart Disease (heart_cleveland_upload.csv)
  Target    : condition  (0 = No Disease, 1 = Disease Present)
  Goal      : Compare a default XGBoost model against one whose hyperparameters
              were found by Optuna with StratifiedKFold cross-validation.
  Sections  :
      0. Google Colab Setup  ← Run this cell FIRST in Colab
      1. Imports & Configuration
      2. Data Loading & EDA Summary
      3. Preprocessing (OHE + Standard Scaling)
      4. Model 1 – XGBoost Baseline (default params)
      5. Model 2 – Optuna-Optimized XGBoost
      6. Evaluation  (metrics table, confusion matrices, ROC curves)
      7. Explainability (SHAP / Feature Importance)
================================================================================
"""

# ==============================================================================
# SECTION 0 ▸ Google Colab Setup  (run this cell first, then restart runtime)
# ==============================================================================
# XGBoost & scikit-learn ship with Colab by default; optuna and shap do not.
# Uncomment and run the line below ONE TIME before executing the rest.

# !pip install -q optuna shap

# ── Upload the dataset ────────────────────────────────────────────────────────
# Option A (simplest) — Files panel:
#   1. Click the folder icon in the left sidebar.
#   2. Drag-and-drop heart_cleveland_upload.csv into /content/.
#   3. Set DATA_PATH = "/content/heart_cleveland_upload.csv" below.
#
# Option B — Google Drive mount:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   DATA_PATH = "/content/drive/MyDrive/YOUR_FOLDER/heart_cleveland_upload.csv"
#
# Option C — Programmatic file-picker:
#   from google.colab import files
#   uploaded = files.upload()
#   DATA_PATH = list(uploaded.keys())[0]
# =============================================================================


# ==============================================================================
# SECTION 1 ▸ Imports & Global Configuration
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")          # Keep output clean

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import xgboost as xgb

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.metrics          import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"       : 150,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
})
PALETTE = {"baseline": "#4C9BE8", "optimized": "#E8724C"}  # blue / orange

# ── Silence per-trial Optuna logs ─────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("  HEART DISEASE CLASSIFICATION — XGBoost + Optuna Pipeline")
print("=" * 70)


# ==============================================================================
# SECTION 2 ▸ Data Loading & Quick EDA
# ==============================================================================

# ▸ SET YOUR PATH HERE (see Section 0 options above)
DATA_PATH = "/content/heart_cleveland_upload.csv"

print("\n[1/7]  Loading dataset …")
df = pd.read_csv(DATA_PATH)

print(f"       Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"       Missing values: {df.isnull().sum().sum()}")
print(f"       Target balance (condition):")
print(df["condition"].value_counts().rename({0: "No Disease", 1: "Disease"}))

# ── Feature taxonomy ──────────────────────────────────────────────────────────
# Categorical (nominal/ordinal) → One-Hot Encoding
# Even though cp, restecg, slope, thal are stored as integers, they represent
# *categories*, not quantities — treating them as continuous would impose a
# false ordinal relationship on the model.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Continuous / count → Standard Scaling
NUMERICAL_FEATURES   = ["age", "sex", "trestbps", "chol", "fbs",
                         "thalach", "exang", "oldpeak", "ca"]
TARGET = "condition"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ==============================================================================
# SECTION 3 ▸ Preprocessing — 80/20 Stratified Split + ColumnTransformer
# ==============================================================================
print("\n[2/7]  Splitting data (80/20 stratified) and building preprocessor …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = RANDOM_STATE,
    stratify     = y          # preserves class ratio in both splits
)
print(f"       Train set : {X_train.shape[0]} samples")
print(f"       Test  set : {X_test.shape[0]}  samples")

# ── ColumnTransformer ─────────────────────────────────────────────────────────
# drop="first" on OHE avoids the dummy-variable trap (multicollinearity).
# StandardScaler centres & normalises continuous features.
# XGBoost itself is scale-invariant, but scaling keeps SHAP magnitudes
# comparable across features.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                                  NUMERICAL_FEATURES),
    ("cat", OneHotEncoder(drop="first", sparse_output=False),  CATEGORICAL_FEATURES),
], remainder="drop")

# Fit ONLY on training data — never expose test statistics to the transformer
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# ── Recover interpretable feature names for SHAP axis labels ─────────────────
cat_names     = (preprocessor
                 .named_transformers_["cat"]
                 .get_feature_names_out(CATEGORICAL_FEATURES)
                 .tolist())
FEATURE_NAMES = NUMERICAL_FEATURES + cat_names
print(f"       Total features after encoding: {len(FEATURE_NAMES)}")


# ==============================================================================
# SECTION 4 ▸ Model 1 — XGBoost Baseline (Default Hyperparameters)
# ==============================================================================
print("\n[3/7]  Training Model 1 — XGBoost Baseline (default params) …")

# Purpose: establish a reference point using XGBoost "out of the box".
# No tuning whatsoever — this is the unconstrained, potentially overfit baseline.
baseline_model = xgb.XGBClassifier(
    objective    = "binary:logistic",
    eval_metric  = "logloss",
    random_state = RANDOM_STATE,
)
baseline_model.fit(X_train_proc, y_train)

y_pred_baseline = baseline_model.predict(X_test_proc)
y_prob_baseline = baseline_model.predict_proba(X_test_proc)[:, 1]
print("       ✓ Baseline model trained.")


# ==============================================================================
# SECTION 5 ▸ Model 2 — Optuna-Optimized XGBoost
# ==============================================================================
print("\n[4/7]  Running Optuna hyperparameter search …")

# ── Why StratifiedKFold inside Optuna? ────────────────────────────────────────
# A single train/validation split during HPO can overfit to that specific
# partition. StratifiedKFold gives a robust generalisation estimate across
# 5 different splits while preserving the class ratio in each fold.

N_OPTUNA_TRIALS = 80   # ↑ trials → better search space coverage
CV_FOLDS        = 5

def objective(trial: optuna.Trial) -> float:
    """
    Maximise mean AUC-ROC across 5 stratified folds.
    AUC-ROC is preferred over accuracy: it measures the model's ability to
    *rank* diseased vs. healthy patients — critical in medical screening.
    """
    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : "logloss",
        "random_state"     : RANDOM_STATE,
        # ── Search space ──────────────────────────────────────────────────────
        "n_estimators"     : trial.suggest_int  ("n_estimators",    50,  500),
        "max_depth"        : trial.suggest_int  ("max_depth",        3,   10),
        "learning_rate"    : trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample",        0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight" : trial.suggest_int  ("min_child_weight", 1,   10),
        "gamma"            : trial.suggest_float("gamma",            0,    5),
        "reg_alpha"        : trial.suggest_float("reg_alpha",        0,    1),
        "reg_lambda"       : trial.suggest_float("reg_lambda",       0.5,  5),
    }

    model = xgb.XGBClassifier(**params)
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        model, X_train_proc, y_train,
        cv      = skf,
        scoring = "roc_auc",
        n_jobs  = -1,
    )
    return cv_scores.mean()


study = optuna.create_study(
    direction  = "maximize",
    study_name = "xgboost_heart_hpo",
    sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "objective"   : "binary:logistic",
    "eval_metric" : "logloss",
    "random_state": RANDOM_STATE,
})

print(f"\n       ✓ Best CV AUC-ROC : {study.best_value:.4f}")
print(f"       Best Params     : {best_params}")

# ── Retrain on the FULL training set with the best hyperparameters ─────────────
print("\n[5/7]  Training final optimised model on full training set …")
optimized_model = xgb.XGBClassifier(**best_params)
optimized_model.fit(X_train_proc, y_train)

y_pred_optimized = optimized_model.predict(X_test_proc)
y_prob_optimized = optimized_model.predict_proba(X_test_proc)[:, 1]
print("       ✓ Optimised model trained.")


# ==============================================================================
# SECTION 6 ▸ Evaluation
# ==============================================================================
print("\n[6/7]  Evaluating models …")

def compute_metrics(y_true, y_pred, y_prob, label: str) -> dict:
    """
    Aggregate all key evaluation metrics for one model.
    Recall is marked CRITICAL: a false negative (missed disease) is far
    more dangerous than a false positive in cardiac screening.
    """
    return {
        "Model"    : label,
        "Accuracy" : round(accuracy_score (y_true, y_pred), 4),
        "Recall*"  : round(recall_score   (y_true, y_pred), 4),   # ★ Critical
        "Precision": round(precision_score(y_true, y_pred), 4),
        "F1-Score" : round(f1_score       (y_true, y_pred), 4),
        "AUC-ROC"  : round(roc_auc_score  (y_true, y_prob), 4),
    }

m_base = compute_metrics(y_test, y_pred_baseline,  y_prob_baseline,  "XGBoost Baseline")
m_opt  = compute_metrics(y_test, y_pred_optimized, y_prob_optimized, "XGBoost Optimized")

results = pd.DataFrame([m_base, m_opt]).set_index("Model")

print("\n" + "=" * 62)
print("  MODEL COMPARISON TABLE")
print("  (* Recall is the most critical metric for medical screening)")
print("=" * 62)
print(results.to_string())
print("=" * 62)


# ── 6a. Side-by-Side Confusion Matrices ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices — Baseline vs. Optimized XGBoost",
             fontsize=14, fontweight="bold", y=1.02)

for ax, y_pred, title, color in zip(
    axes,
    [y_pred_baseline,       y_pred_optimized],
    ["Baseline (Default)",  "Optimized (Optuna)"],
    [PALETTE["baseline"],   PALETTE["optimized"]],
):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)

plt.tight_layout()
plt.savefig("confusion_matrices.png", bbox_inches="tight")
plt.show()
print("       ✓ confusion_matrices.png saved")


# ── 6b. ROC Curves — both models on one chart ─────────────────────────────────
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_baseline)
fpr_o, tpr_o, _ = roc_curve(y_test, y_prob_optimized)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_b, tpr_b, color=PALETTE["baseline"], lw=2,
        label=f"Baseline   (AUC = {m_base['AUC-ROC']:.4f})")
ax.plot(fpr_o, tpr_o, color=PALETTE["optimized"], lw=2,
        label=f"Optimized  (AUC = {m_opt['AUC-ROC']:.4f})")
ax.fill_between(fpr_o, tpr_o, alpha=0.08, color=PALETTE["optimized"])
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curve Comparison — Performance Lift of Optuna Optimisation",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("roc_curves.png", bbox_inches="tight")
plt.show()
print("       ✓ roc_curves.png saved")


# ==============================================================================
# SECTION 7 ▸ Explainability — SHAP Values + Feature Importance
# ==============================================================================
print("\n[7/7]  Computing SHAP values for the optimised model …")

X_test_df = pd.DataFrame(X_test_proc, columns=FEATURE_NAMES)

# TreeExplainer is native to tree-based models — exact, no approximation.
explainer   = shap.TreeExplainer(optimized_model)
shap_values = explainer.shap_values(X_test_df)

# ── 7a. SHAP Beeswarm (Summary) Plot ─────────────────────────────────────────
# Each dot = one patient.
# Horizontal position = SHAP value (impact on model output).
# Colour = feature value (red = high, blue = low).
# Shows NOT just which features matter, but HOW they drive predictions.
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_df,
                  plot_type="dot", show=False, max_display=len(FEATURE_NAMES))
plt.title("SHAP Beeswarm — Feature Impact on Disease Prediction\n(Optimized XGBoost)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", bbox_inches="tight")
plt.show()
print("       ✓ shap_beeswarm.png saved")

# ── 7b. SHAP Bar Plot (Mean |SHAP| = Global Importance) ──────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_df,
                  plot_type="bar", show=False, max_display=len(FEATURE_NAMES))
plt.title("Mean |SHAP| — Global Feature Importance (Optimized XGBoost)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_bar.png", bbox_inches="tight")
plt.show()
print("       ✓ shap_bar.png saved")

# ── 7c. XGBoost Native Gain-Based Feature Importance ─────────────────────────
# Gain = average loss improvement brought by a feature across all splits.
# Complements SHAP by giving a tree-structure view of importance.
gain_dict = optimized_model.get_booster().get_score(importance_type="gain")
feat_map  = {f"f{i}": n for i, n in enumerate(FEATURE_NAMES)}
gain_df   = (pd.DataFrame.from_dict(gain_dict, orient="index", columns=["Gain"])
               .reset_index().rename(columns={"index": "Feature"})
               .assign(Feature=lambda d: d["Feature"].map(feat_map)
                                                       .fillna(d["Feature"]))
               .sort_values("Gain", ascending=True))

fig, ax = plt.subplots(figsize=(10, max(5, len(gain_df) * 0.42)))
bars = ax.barh(gain_df["Feature"], gain_df["Gain"],
               color=PALETTE["optimized"], edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
ax.set_xlabel("Feature Gain (XGBoost Native)", fontsize=11)
ax.set_title("XGBoost Gain-Based Feature Importance — Optimized Model",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance_gain.png", bbox_inches="tight")
plt.show()
print("       ✓ feature_importance_gain.png saved")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — FINAL RESULTS")
print("=" * 70)
print(results.to_string())
print("\n  Saved artefacts:")
for f in ["confusion_matrices.png", "roc_curves.png",
          "shap_beeswarm.png", "shap_bar.png", "feature_importance_gain.png"]:
    print(f"    • {f}")
print("\n  Clinical reminder:")
print("  Recall* — how many sick patients we correctly caught.")
print("  A low Recall means missed heart disease cases (false negatives),")
print("  which is the most dangerous error type in medical screening.")
print("=" * 70)
