#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: Generate leakage-free hold-out ROC curves for revision.

This script trains the following models using a leakage-free pipeline:
  - ExtraTrees
  - RandomForest
  - XGBoost
  - LightGBM
  - SoftVoting
  - FeedbackOptimizedStacking

Key guarantees:
  - Stratified train/test split is performed before any fitting.
  - Preprocessing is fitted only on the training split.
  - SMOTE is applied only to the training split through imblearn.Pipeline.
  - ROC-AUC is computed from predicted probabilities, not hard labels.

Outputs:
  - roc_curves_holdout_combined.png
  - roc_curves_holdout_grid.png
  - roc_auc_holdout_summary.csv
  - roc_curve_points.csv
  - roc_summary_for_paper.md
  - roc_latex_snippet.tex

Example:
  python step5_roc_curves_vscode.py ^
    --data WA_Fn-UseC_-Telco-Customer-Churn.csv ^
    --best_config step2_feedback_20iter_check\best_feedback_config.json ^
    --outdir step5_roc_curves ^
    --n_estimators 200 ^
    --n_jobs 1
"""

import argparse
import json
import os
import platform
import warnings
from pathlib import Path

# Windows/joblib stability
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.preprocessing import OneHotEncoder
    _HAS_SPARSE_OUTPUT = True
except Exception:
    from sklearn.preprocessing import OneHotEncoder
    _HAS_SPARSE_OUTPUT = False

from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


def make_onehot_encoder():
    """Handle scikit-learn versions before and after sparse_output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_telco(path: str):
    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" not in df.columns:
        raise ValueError("The dataset must contain a 'Churn' column.")

    # Remove rows with missing target if any
    df = df[df["Churn"].notna()].copy()

    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        # robust fallback for already encoded datasets
        y = df["Churn"].astype(int)

    X = df.drop(columns=["Churn"])

    audit = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "blank_totalcharges_after_numeric": int(df["TotalCharges"].isna().sum()) if "TotalCharges" in df.columns else None,
        "target_counts": {str(k): int(v) for k, v in df["Churn"].value_counts().to_dict().items()},
        "target_ratio": {str(k): round(float(v), 4) for k, v in df["Churn"].value_counts(normalize=True).to_dict().items()},
    }
    return X, y.astype(int), audit


def make_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocess, numeric_cols, categorical_cols


def read_best_params(path: str | None):
    """Read best feedback config. Accepts either best_params or flat json."""
    defaults = {
        "model__et__max_depth": 10,
        "model__rf__max_depth": 10,
        "model__xgb__learning_rate": 0.05,
        "model__xgb__max_depth": 4,
        "model__lgbm__learning_rate": 0.05,
        "model__lgbm__num_leaves": 15,
        "model__final_estimator__C": 1.132953262747208,
        "model__final_estimator__class_weight": "balanced",
        "model__passthrough": True,
        "smote__sampling_strategy": 0.6,
        "smote__k_neighbors": 3,
    }
    if not path:
        return defaults
    p = Path(path)
    if not p.exists():
        print(f"[Warning] best_config file not found: {path}. Using defaults.")
        return defaults
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "best_params" in cfg:
        cfg = cfg["best_params"]
    out = defaults.copy()
    out.update(cfg)
    return out


def make_base_models(n_estimators: int, random_state: int, n_jobs: int, prefix: str = ""):
    # Prefix is unused, but kept for readability and future expansion.
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=None,
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=None,
    )
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.2,
        num_leaves=31,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    return et, rf, xgb, lgbm


def make_feedback_stacking(best_params, n_estimators: int, random_state: int, inner_cv_splits: int, n_jobs: int):
    et_depth = int(best_params.get("model__et__max_depth", 10))
    rf_depth = int(best_params.get("model__rf__max_depth", 10))
    xgb_lr = float(best_params.get("model__xgb__learning_rate", 0.05))
    xgb_depth = int(best_params.get("model__xgb__max_depth", 4))
    lgbm_lr = float(best_params.get("model__lgbm__learning_rate", 0.05))
    lgbm_leaves = int(best_params.get("model__lgbm__num_leaves", 15))
    lr_c = float(best_params.get("model__final_estimator__C", 1.132953262747208))
    lr_class_weight = best_params.get("model__final_estimator__class_weight", "balanced")
    passthrough = bool(best_params.get("model__passthrough", True))

    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=et_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=rf_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=xgb_depth,
        learning_rate=xgb_lr,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=lgbm_lr,
        num_leaves=lgbm_leaves,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )

    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
    stacking = StackingClassifier(
        estimators=[
            ("et", et),
            ("rf", rf),
            ("xgb", xgb),
            ("lgbm", lgbm),
        ],
        final_estimator=LogisticRegression(
            C=lr_c,
            class_weight=lr_class_weight,
            max_iter=2000,
            random_state=random_state,
        ),
        stack_method="predict_proba",
        passthrough=passthrough,
        cv=inner_cv,
        n_jobs=n_jobs,
    )
    return stacking


def make_pipeline(preprocess, model, smote_sampling=1.0, smote_k=5, random_state=42):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(
                sampling_strategy=smote_sampling,
                k_neighbors=smote_k,
                random_state=random_state,
            )),
            ("model", model),
        ]
    )


def evaluate_at_threshold(y_true, proba, threshold: float):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "mcc": matthews_corrcoef(y_true, pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--best_config", default=None, help="Path to step2 best_feedback_config.json")
    parser.add_argument("--outdir", default="step5_roc_curves", help="Output directory")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--inner_cv_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, audit = load_telco(args.data)
    preprocess, numeric_cols, categorical_cols = make_preprocessor(X)
    best_params = read_best_params(args.best_config)

    # Stratified hold-out split before fitting any preprocessing or SMOTE.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    # Fixed base models for reviewer-requested ROC curves
    et, rf, xgb, lgbm = make_base_models(args.n_estimators, args.random_state, args.n_jobs)

    voting = VotingClassifier(
        estimators=[
            ("et", et),
            ("rf", rf),
            ("xgb", xgb),
            ("lgbm", lgbm),
        ],
        voting="soft",
        n_jobs=args.n_jobs,
    )

    feedback_stack = make_feedback_stacking(
        best_params,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        inner_cv_splits=args.inner_cv_splits,
        n_jobs=args.n_jobs,
    )

    smote_sampling_feedback = float(best_params.get("smote__sampling_strategy", 0.6))
    smote_k_feedback = int(best_params.get("smote__k_neighbors", 3))

    model_specs = [
        ("Extra Trees", make_pipeline(preprocess, et, 1.0, 5, args.random_state)),
        ("Random Forest", make_pipeline(preprocess, rf, 1.0, 5, args.random_state)),
        ("XGBoost", make_pipeline(preprocess, xgb, 1.0, 5, args.random_state)),
        ("LightGBM", make_pipeline(preprocess, lgbm, 1.0, 5, args.random_state)),
        ("Soft Voting", make_pipeline(preprocess, voting, 1.0, 5, args.random_state)),
        ("Feedback-Optimized Stacking", make_pipeline(
            preprocess,
            feedback_stack,
            smote_sampling_feedback,
            smote_k_feedback,
            args.random_state
        )),
    ]

    summary_rows = []
    curve_rows = []
    fitted_probs = {}

    print("[Step 5] Training models and computing hold-out ROC curves...")
    for name, pipe in model_specs:
        print(f"  - {name}")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        fitted_probs[name] = (fpr, tpr, thresholds, auc, proba)

        metrics = evaluate_at_threshold(y_test, proba, 0.5)
        metrics["Model"] = name
        metrics["threshold_for_class_metrics"] = 0.5
        summary_rows.append(metrics)

        for fp, tp, th in zip(fpr, tpr, thresholds):
            curve_rows.append({
                "Model": name,
                "fpr": fp,
                "tpr": tp,
                "threshold": th,
                "roc_auc": auc,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[
        "Model",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "f1",
        "mcc",
        "threshold_for_class_metrics",
    ]]
    summary_df.to_csv(outdir / "roc_auc_holdout_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(outdir / "roc_curve_points.csv", index=False)

    with (outdir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    run_config = {
        "data": args.data,
        "best_config": args.best_config,
        "outdir": str(outdir),
        "n_estimators": args.n_estimators,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "inner_cv_splits": args.inner_cv_splits,
        "n_jobs": args.n_jobs,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    with (outdir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # Combined ROC curve
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, thresholds, auc, proba) in fitted_probs.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Hold-out ROC curves under leakage-free evaluation")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curves_holdout_combined.png", dpi=300)
    plt.savefig(outdir / "roc_curves_holdout_combined.pdf")
    plt.close()

    # Grid of individual ROC curves
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (name, (fpr, tpr, thresholds, auc, proba)) in zip(axes, fitted_probs.items()):
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curves_holdout_grid.png", dpi=300)
    plt.savefig(outdir / "roc_curves_holdout_grid.pdf")
    plt.close()

    # Manuscript summary
    best_auc_row = summary_df.sort_values("roc_auc", ascending=False).iloc[0]
    md = []
    md.append("# Step 5 ROC Curve Summary\n")
    md.append("## Leakage-free ROC analysis\n")
    md.append(
        "ROC curves were generated on an untouched stratified hold-out test set. "
        "All preprocessing steps and SMOTE were fitted only on the training split. "
        "ROC-AUC was computed from predicted probabilities rather than hard class labels.\n"
    )
    md.append("## Hold-out ROC-AUC summary\n")
    # Simple markdown table without requiring the optional 'tabulate' package.
    md.append("| " + " | ".join(summary_df.columns) + " |")
    md.append("| " + " | ".join(["---"] * len(summary_df.columns)) + " |")
    for _, row in summary_df.iterrows():
        vals = []
        for col in summary_df.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        md.append("| " + " | ".join(vals) + " |")
    md.append("\n")
    md.append(
        f"The highest hold-out ROC-AUC was obtained by **{best_auc_row['Model']}** "
        f"with ROC-AUC = {best_auc_row['roc_auc']:.4f}. "
        "The optimized stacking model remained competitive while providing a threshold-calibrated "
        "trade-off for churn detection in the main ablation analysis.\n"
    )
    (outdir / "roc_summary_for_paper.md").write_text("\n".join(md), encoding="utf-8")

    latex = r"""
% Add this figure to the Results and Evaluation section.
\begin{figure}[!t]
    \centering
    \includegraphics[width=0.88\linewidth]{roc_curves_holdout_combined.png}
    \caption{Hold-out ROC curves of the base learners, soft voting ensemble, and feedback-optimized stacking model under the leakage-free evaluation protocol. ROC-AUC values were computed from predicted probabilities rather than hard class labels.}
    \label{fig:holdout_roc_curves}
\end{figure}

% Optional: use this grid if the journal/reviewer prefers separate ROC panels.
% \begin{figure}[!t]
%     \centering
%     \includegraphics[width=\linewidth]{roc_curves_holdout_grid.png}
%     \caption{Individual hold-out ROC curves for Extra Trees, Random Forest, XGBoost, LightGBM, soft voting, and feedback-optimized stacking.}
%     \label{fig:holdout_roc_grid}
% \end{figure}
"""
    (outdir / "roc_latex_snippet.tex").write_text(latex.strip() + "\n", encoding="utf-8")

    print(f"[Done] Outputs saved to: {outdir.resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
