"""
================================================================================
  Heart Disease Classification: XGBoost Baseline vs. Optuna-Optimized Pipeline
================================================================================
  Datasets  :
      1. heart_cleveland_upload.csv         (297 rows  — original Cleveland)
      2. heart_disease_uci.csv              (920 rows  — UCI multi-source)
      3. heart_statlog_cleveland_hungary_final.csv (1190 rows — Statlog combined)

  Target    : Binary heart disease (0 = No Disease, 1 = Disease Present)
  Goal      : Compare baseline XGBoost vs Optuna-tuned XGBoost on each dataset.
================================================================================
"""

# ==============================================================================
# SECTION 0 ▸ Google Colab Setup  (run this cell first, then restart runtime)
# ==============================================================================
# !pip install -q optuna shap

# ==============================================================================
# SECTION 1 ▸ Imports & Global Configuration
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import xgboost as xgb

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.impute            import SimpleImputer
from sklearn.metrics          import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE    = 42
N_OPTUNA_TRIALS = 80
CV_FOLDS        = 5

np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "figure.dpi"       : 150,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
})
PALETTE = {"baseline": "#4C9BE8", "optimized": "#E8724C"}

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("  HEART DISEASE CLASSIFICATION — XGBoost + Optuna  (Multi-Dataset)")
print("=" * 70)


# ==============================================================================
# SECTION 2 ▸ Dataset-Specific Loaders
#   Each loader returns (X, y) with a consistent feature schema so the
#   rest of the pipeline (preprocessing → training → evaluation) is reused.
# ==============================================================================

# ── 2a. Cleveland ─────────────────────────────────────────────────────────────
def load_cleveland(path: str):
    """
    Original Cleveland dataset.
    Columns are already clean; target is 'condition' (0/1).
    Categoricals are integer-coded → treat as categorical.
    """
    df = pd.read_csv(path)

    # Map integer codes to strings so OHE works uniformly across datasets
    df["cp"]      = df["cp"].map({0:"typical_angina", 1:"atypical_angina",
                                   2:"non_anginal",    3:"asymptomatic"})
    df["restecg"] = df["restecg"].map({0:"normal", 1:"st_t_abnorm", 2:"lv_hypertrophy"})
    df["slope"]   = df["slope"].map({1:"upsloping", 2:"flat", 3:"downsloping"})
    # 0 = not tested → treat as NaN (will be imputed); 3/6/7 are the known codes
    df["thal"]    = df["thal"].map({0:np.nan, 3:"normal", 6:"fixed_defect", 7:"reversable_defect"})

    X = df.drop(columns=["condition"])
    y = df["condition"]
    return X, y


# ── 2b. UCI (heart_disease_uci.csv) ───────────────────────────────────────────
def load_uci(path: str):
    """
    920-row UCI dataset combining Cleveland, Hungary, Switzerland, VA.
    - Extra columns 'id' and 'dataset' are dropped.
    - sex / fbs / exang are stored as strings/booleans → convert.
    - thalch renamed to thalach for consistency.
    - num (0-4) → binarised: 0 = no disease, 1-4 = disease.
    - Heavy missingness in slope (~34%), ca (~66%), thal (~53%) → imputed.
    """
    df = pd.read_csv(path)

    # Drop irrelevant identifier columns
    df = df.drop(columns=["id", "dataset"], errors="ignore")

    # Rename to match Cleveland naming convention
    df = df.rename(columns={"thalch": "thalach"})

    # Binary string columns → 0/1
    df["sex"]   = (df["sex"]   == "Male").astype(int)
    df["fbs"]   = df["fbs"].map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)
    df["exang"] = df["exang"].map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)

    # Normalise categorical string values to snake_case
    df["cp"]      = df["cp"].str.strip().str.lower().str.replace(" ", "_")
    df["restecg"] = df["restecg"].str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    df["slope"]   = df["slope"].str.strip().str.lower().str.replace(" ", "_")
    df["thal"]    = df["thal"].str.strip().str.lower().str.replace(" ", "_")

    # Binarise multi-class target: 0 = healthy, ≥1 = disease
    y = (df["num"] >= 1).astype(int)
    X = df.drop(columns=["num"])
    return X, y


# ── 2c. Statlog (heart_statlog_cleveland_hungary_final.csv) ───────────────────
def load_statlog(path: str):
    """
    1190-row Statlog dataset — no missing values, all integers.
    Column names differ from Cleveland → renamed to a common schema.
    'thal' and 'ca' columns are absent in this dataset.
    """
    df = pd.read_csv(path)

    df = df.rename(columns={
        "chest pain type"    : "cp",
        "resting bp s"       : "trestbps",
        "cholesterol"        : "chol",
        "fasting blood sugar": "fbs",
        "resting ecg"        : "restecg",
        "max heart rate"     : "thalach",
        "exercise angina"    : "exang",
        "ST slope"           : "slope",
        "target"             : "condition",
    })

    # Map integer codes to strings to match Cleveland categorical encoding
    df["cp"]      = df["cp"].map({1:"typical_angina", 2:"atypical_angina",
                                   3:"non_anginal",    4:"asymptomatic"})
    df["restecg"] = df["restecg"].map({0:"normal", 1:"st_t_abnorm", 2:"lv_hypertrophy"})
    df["slope"]   = df["slope"].map({0:"unknown", 1:"upsloping", 2:"flat", 3:"downsloping"})

    y = df["condition"]
    X = df.drop(columns=["condition"])
    return X, y


# ==============================================================================
# SECTION 3 ▸ Feature Schema per Dataset
#   Each dataset has slightly different available features.
#   We define which columns are numerical vs categorical per dataset.
# ==============================================================================

DATASET_CONFIGS = {
    "Cleveland": {
        "path"      : "dataset/heart_cleveland_upload.csv",
        "loader"    : load_cleveland,
        "numerical" : ["age", "sex", "trestbps", "chol", "fbs",
                        "thalach", "exang", "oldpeak", "ca"],
        "categorical": ["cp", "restecg", "slope", "thal"],
    },
    "UCI": {
        "path"       : "dataset/heart_disease_uci.csv",
        "loader"     : load_uci,
        "numerical"  : ["age", "sex", "trestbps", "chol", "fbs",
                         "thalach", "exang", "oldpeak", "ca"],
        "categorical": ["cp", "restecg", "slope", "thal"],
    },
    "Statlog": {
        "path"       : "dataset/heart_statlog_cleveland_hungary_final.csv",
        "loader"     : load_statlog,
        # 'thal' and 'ca' are absent in Statlog
        "numerical"  : ["age", "sex", "trestbps", "chol", "fbs",
                         "thalach", "exang", "oldpeak"],
        "categorical": ["cp", "restecg", "slope"],
    },
}


# ==============================================================================
# SECTION 4 ▸ Preprocessor Builder
#   Builds a ColumnTransformer that:
#     - Imputes + scales numerical features
#     - Imputes + one-hot-encodes categorical features
#   Imputation is included so the pipeline handles UCI's missing values
#   without crashing (Cleveland & Statlog have no missing values, so
#   imputation is a no-op for them).
# ==============================================================================

def build_preprocessor(numerical_features, categorical_features):
    """
    Returns a fitted-ready ColumnTransformer.
    Numerical  : median imputation → StandardScaler
    Categorical: most-frequent imputation → OHE (drop='first')
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="first", sparse_output=False,
                                  handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ], remainder="drop")


# ==============================================================================
# SECTION 5 ▸ Model Training Helpers
# ==============================================================================

def train_baseline(X_train, y_train):
    model = xgb.XGBClassifier(
        objective    = "binary:logistic",
        eval_metric  = "logloss",
        random_state = RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def run_optuna(X_train, y_train):
    """Optuna TPE search maximising mean AUC-ROC over StratifiedKFold."""
    def objective(trial):
        params = {
            "objective"        : "binary:logistic",
            "eval_metric"      : "logloss",
            "random_state"     : RANDOM_STATE,
            "n_estimators"     : trial.suggest_int  ("n_estimators",   50,  500),
            "max_depth"        : trial.suggest_int  ("max_depth",       3,   10),
            "learning_rate"    : trial.suggest_float("learning_rate",  0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample",       0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree",0.5, 1.0),
            "min_child_weight" : trial.suggest_int  ("min_child_weight",1,   10),
            "gamma"            : trial.suggest_float("gamma",           0,    5),
            "reg_alpha"        : trial.suggest_float("reg_alpha",       0,    1),
            "reg_lambda"       : trial.suggest_float("reg_lambda",      0.5,  5),
        }
        model = xgb.XGBClassifier(**params)
        skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction  = "maximize",
        study_name = "xgboost_heart_hpo",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    best.update({"objective": "binary:logistic",
                 "eval_metric": "logloss",
                 "random_state": RANDOM_STATE})
    print(f"       ✓ Best CV AUC-ROC : {study.best_value:.4f}")
    print(f"         Best Params     : {best}")
    return best


# ==============================================================================
# SECTION 6 ▸ Evaluation Helper
# ==============================================================================

def compute_metrics(y_true, y_pred, y_prob, label):
    return {
        "Model"    : label,
        "Accuracy" : round(accuracy_score (y_true, y_pred), 4),
        "Recall*"  : round(recall_score   (y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "F1-Score" : round(f1_score       (y_true, y_pred), 4),
        "AUC-ROC"  : round(roc_auc_score  (y_true, y_prob), 4),
    }


# ==============================================================================
# SECTION 7 ▸ Per-Dataset Pipeline Runner
# ==============================================================================

all_results = {}   # dataset_name → results DataFrame

def run_pipeline(dataset_name: str, cfg: dict):
    print("\n" + "=" * 70)
    print(f"  DATASET: {dataset_name}")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    X, y = cfg["loader"](cfg["path"])
    print(f"  Shape: {X.shape[0]} rows × {X.shape[1]+1} cols  "
          f"| Target balance: {y.value_counts().to_dict()}")

    num_feats = [f for f in cfg["numerical"]  if f in X.columns]
    cat_feats = [f for f in cfg["categorical"] if f in X.columns]

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocessor = build_preprocessor(num_feats, cat_feats)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Recover feature names for SHAP labels.
    # Use get_feature_names_out() on the ColumnTransformer itself — this is
    # robust against SimpleImputer silently dropping all-NaN columns, because
    # it reflects exactly what columns the transformers actually fitted on.
    raw_names     = preprocessor.get_feature_names_out().tolist()
    # Strip sklearn's "num__" / "cat__" prefixes  →  e.g. "num__age" → "age"
    feature_names = [n.split("__", 1)[-1] for n in raw_names]
    print(f"  Features after encoding: {len(feature_names)}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"\n  [1/3] Training Baseline XGBoost …")
    base_model      = train_baseline(X_train_proc, y_train)
    y_pred_base     = base_model.predict(X_test_proc)
    y_prob_base     = base_model.predict_proba(X_test_proc)[:, 1]

    # ── Optuna ────────────────────────────────────────────────────────────────
    print(f"\n  [2/3] Running Optuna ({N_OPTUNA_TRIALS} trials) …")
    best_params     = run_optuna(X_train_proc, y_train)
    opt_model       = xgb.XGBClassifier(**best_params)
    opt_model.fit(X_train_proc, y_train)
    y_pred_opt      = opt_model.predict(X_test_proc)
    y_prob_opt      = opt_model.predict_proba(X_test_proc)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\n  [3/3] Evaluating …")
    m_base = compute_metrics(y_test, y_pred_base, y_prob_base, "XGBoost Baseline")
    m_opt  = compute_metrics(y_test, y_pred_opt,  y_prob_opt,  "XGBoost Optimized")
    results = pd.DataFrame([m_base, m_opt]).set_index("Model")
    all_results[dataset_name] = results

    print(f"\n  {'='*62}")
    print(f"  MODEL COMPARISON — {dataset_name}")
    print(f"  {'='*62}")
    print(results.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    prefix = dataset_name.lower().replace(" ", "_")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices — {dataset_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    for ax, y_pred, title, color in zip(
        axes,
        [y_pred_base, y_pred_opt],
        ["Baseline (Default)", "Optimized (Optuna)"],
        [PALETTE["baseline"], PALETTE["optimized"]],
    ):
        cm   = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrices.png", bbox_inches="tight")
    plt.show()

    # ROC curves
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
    fpr_o, tpr_o, _ = roc_curve(y_test, y_prob_opt)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_b, tpr_b, color=PALETTE["baseline"], lw=2,
            label=f"Baseline   (AUC = {m_base['AUC-ROC']:.4f})")
    ax.plot(fpr_o, tpr_o, color=PALETTE["optimized"], lw=2,
            label=f"Optimized  (AUC = {m_opt['AUC-ROC']:.4f})")
    ax.fill_between(fpr_o, tpr_o, alpha=0.08, color=PALETTE["optimized"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(f"{prefix}_roc_curves.png", bbox_inches="tight")
    plt.show()

    # SHAP (optimized model)
    X_test_df   = pd.DataFrame(X_test_proc, columns=feature_names)
    explainer   = shap.TreeExplainer(opt_model)
    shap_values = explainer.shap_values(X_test_df)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df,
                      plot_type="dot", show=False, max_display=len(feature_names))
    plt.title(f"SHAP Beeswarm — {dataset_name} (Optimized XGBoost)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}_shap_beeswarm.png", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df,
                      plot_type="bar", show=False, max_display=len(feature_names))
    plt.title(f"Mean |SHAP| — {dataset_name} (Optimized XGBoost)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}_shap_bar.png", bbox_inches="tight")
    plt.show()

    print(f"  ✓ All plots saved for {dataset_name}")
    return results


# ==============================================================================
# SECTION 8 ▸ Run All Datasets
# ==============================================================================
for ds_name, ds_cfg in DATASET_CONFIGS.items():
    run_pipeline(ds_name, ds_cfg)


# ==============================================================================
# SECTION 9 ▸ Cross-Dataset Summary Table
# ==============================================================================
print("\n" + "=" * 70)
print("  CROSS-DATASET SUMMARY")
print("=" * 70)

summary_rows = []
for ds_name, res_df in all_results.items():
    for model_label, row in res_df.iterrows():
        summary_rows.append({
            "Dataset" : ds_name,
            "Model"   : model_label,
            **row.to_dict(),
        })

summary_df = pd.DataFrame(summary_rows).set_index(["Dataset", "Model"])
print(summary_df.to_string())

# ── Bar chart: AUC-ROC across datasets ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x       = np.arange(len(all_results))
width   = 0.35

auc_base = [all_results[d].loc["XGBoost Baseline",  "AUC-ROC"] for d in all_results]
auc_opt  = [all_results[d].loc["XGBoost Optimized", "AUC-ROC"] for d in all_results]

bars1 = ax.bar(x - width/2, auc_base, width, label="Baseline",
               color=PALETTE["baseline"], alpha=0.85)
bars2 = ax.bar(x + width/2, auc_opt,  width, label="Optimized",
               color=PALETTE["optimized"], alpha=0.85)

ax.bar_label(bars1, fmt="%.4f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(list(all_results.keys()), fontsize=11)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_ylim([0.5, 1.05])
ax.set_title("AUC-ROC: Baseline vs. Optuna-Optimized XGBoost  (All Datasets)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("cross_dataset_auc_comparison.png", bbox_inches="tight")
plt.show()
print("  ✓ cross_dataset_auc_comparison.png saved")

# ── Bar chart: Recall across datasets ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
rec_base = [all_results[d].loc["XGBoost Baseline",  "Recall*"] for d in all_results]
rec_opt  = [all_results[d].loc["XGBoost Optimized", "Recall*"] for d in all_results]

bars1 = ax.bar(x - width/2, rec_base, width, label="Baseline",
               color=PALETTE["baseline"], alpha=0.85)
bars2 = ax.bar(x + width/2, rec_opt,  width, label="Optimized",
               color=PALETTE["optimized"], alpha=0.85)
ax.bar_label(bars1, fmt="%.4f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(list(all_results.keys()), fontsize=11)
ax.set_ylabel("Recall*", fontsize=11)
ax.set_ylim([0.5, 1.05])
ax.set_title("Recall*: Baseline vs. Optuna-Optimized XGBoost  (All Datasets)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("cross_dataset_recall_comparison.png", bbox_inches="tight")
plt.show()
print("  ✓ cross_dataset_recall_comparison.png saved")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("  Clinical reminder: Recall* = sick patients correctly caught.")
print("  A low Recall means missed heart disease cases (false negatives).")
print("=" * 70)
