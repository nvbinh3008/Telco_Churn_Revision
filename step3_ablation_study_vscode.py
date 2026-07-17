#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Ablation Study for Telco Customer Churn Revision

Purpose
-------
This script evaluates the contribution of each component required by reviewers:
1) Single strong model baseline (XGBoost) without SMOTE
2) Single strong model + SMOTE
3) Fixed stacking without SMOTE
4) Fixed stacking + SMOTE
5) Feedback-optimized stacking + SMOTE at default threshold 0.50
6) Full feedback-optimized stacking + SMOTE + threshold tuning

Leakage-free design
-------------------
- Preprocessing is fitted only on the training fold.
- SMOTE is applied only to the training fold, never to validation/test folds.
- ROC-AUC is computed from predicted probabilities.
- Threshold tuning is performed only inside the training data of each fold
  using an internal threshold-validation split.

Compatible with Windows / Visual Studio Code.
"""

import os

# Windows/joblib/threading stability settings must be applied before sklearn imports.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free ablation study for optimized stacking churn model."
    )
    parser.add_argument("--data", required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument(
        "--best_config",
        required=True,
        help="Path to best_feedback_config.json from Step 2 feedback-loop optimization.",
    )
    parser.add_argument("--outdir", default="step3_ablation_full", help="Output directory")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--inner_cv_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--threshold_metric",
        choices=["f1", "balanced_accuracy", "mcc", "recall"],
        default="f1",
        help="Metric used for threshold tuning on the training-only threshold split.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: n_estimators=30, cv_splits=3, inner_cv_splits=3.",
    )
    return parser.parse_args()


def load_best_params(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "best_params" in obj:
        return obj["best_params"]
    return obj


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df = pd.read_csv(path)

    audit = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_customerID": int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else None,
    }

    # Convert TotalCharges safely.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        audit["blank_totalcharges_after_numeric"] = int(df["TotalCharges"].isna().sum())
    else:
        audit["blank_totalcharges_after_numeric"] = None

    # Remove rows without target if any.
    df = df.dropna(subset=["Churn"]).copy()

    audit["target_counts"] = {str(k): int(v) for k, v in df["Churn"].value_counts().to_dict().items()}
    audit["target_ratio"] = {
        str(k): round(float(v), 4) for k, v in df["Churn"].value_counts(normalize=True).to_dict().items()
    }

    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn"])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    return X, y, audit


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def make_xgb(n_estimators: int, max_depth: int, learning_rate: float, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def make_lgbm(n_estimators: int, learning_rate: float, num_leaves: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=-1,
    )


def make_fixed_stacking(n_estimators: int, inner_cv_splits: int, n_jobs: int) -> StackingClassifier:
    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=RANDOM_STATE)

    estimators = [
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            ),
        ),
        ("xgb", make_xgb(n_estimators, max_depth=4, learning_rate=0.2, n_jobs=n_jobs)),
        ("lgbm", make_lgbm(n_estimators, learning_rate=0.2, num_leaves=31, n_jobs=n_jobs)),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=False,
        n_jobs=n_jobs,
    )


def make_feedback_stacking(
    best_params: Dict[str, Any],
    n_estimators: int,
    inner_cv_splits: int,
    n_jobs: int,
) -> StackingClassifier:
    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=RANDOM_STATE)

    et_depth = int(get_param(best_params, "model__et__max_depth", 10))
    rf_depth = int(get_param(best_params, "model__rf__max_depth", 10))
    xgb_depth = int(get_param(best_params, "model__xgb__max_depth", 4))
    xgb_lr = float(get_param(best_params, "model__xgb__learning_rate", 0.05))
    lgbm_lr = float(get_param(best_params, "model__lgbm__learning_rate", 0.05))
    lgbm_num_leaves = int(get_param(best_params, "model__lgbm__num_leaves", 15))

    final_c = float(get_param(best_params, "model__final_estimator__C", 1.0))
    final_class_weight = get_param(best_params, "model__final_estimator__class_weight", None)
    passthrough = bool(get_param(best_params, "model__passthrough", True))

    estimators = [
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=et_depth,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=rf_depth,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            ),
        ),
        ("xgb", make_xgb(n_estimators, max_depth=xgb_depth, learning_rate=xgb_lr, n_jobs=n_jobs)),
        ("lgbm", make_lgbm(n_estimators, learning_rate=lgbm_lr, num_leaves=lgbm_num_leaves, n_jobs=n_jobs)),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=final_c,
            class_weight=final_class_weight,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=passthrough,
        n_jobs=n_jobs,
    )


def make_pipeline_for_variant(
    variant_name: str,
    X_reference: pd.DataFrame,
    best_params: Dict[str, Any],
    n_estimators: int,
    inner_cv_splits: int,
    n_jobs: int,
) -> Any:
    preprocess = make_preprocessor(X_reference)

    fixed_smote = SMOTE(random_state=RANDOM_STATE)
    feedback_smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy=float(get_param(best_params, "smote__sampling_strategy", 0.6)),
        k_neighbors=int(get_param(best_params, "smote__k_neighbors", 3)),
    )

    if variant_name == "Single_XGBoost_NoSMOTE":
        model = make_xgb(n_estimators, max_depth=4, learning_rate=0.2, n_jobs=n_jobs)
        return Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    if variant_name == "Single_XGBoost_SMOTE":
        model = make_xgb(n_estimators, max_depth=4, learning_rate=0.2, n_jobs=n_jobs)
        return ImbPipeline(steps=[("preprocess", preprocess), ("smote", fixed_smote), ("model", model)])

    if variant_name == "Stacking_NoSMOTE_Fixed":
        model = make_fixed_stacking(n_estimators, inner_cv_splits, n_jobs)
        return Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    if variant_name == "Stacking_SMOTE_Fixed":
        model = make_fixed_stacking(n_estimators, inner_cv_splits, n_jobs)
        return ImbPipeline(steps=[("preprocess", preprocess), ("smote", fixed_smote), ("model", model)])

    if variant_name in [
        "FeedbackStacking_SMOTE_Threshold0.50",
        "Full_FeedbackStacking_SMOTE_TunedThreshold",
    ]:
        model = make_feedback_stacking(best_params, n_estimators, inner_cv_splits, n_jobs)
        return ImbPipeline(steps=[("preprocess", preprocess), ("smote", feedback_smote), ("model", model)])

    raise ValueError(f"Unknown variant: {variant_name}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_objective(metric: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)

    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    raise ValueError(metric)


def tune_threshold_on_training_split(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str,
    random_state: int,
) -> Tuple[float, pd.DataFrame]:
    # The threshold is selected using a threshold-validation subset taken only from training data.
    X_fit, X_thr, y_fit, y_thr = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state,
    )

    model = clone(pipeline)
    model.fit(X_fit, y_fit)
    y_prob_thr = model.predict_proba(X_thr)[:, 1]

    rows = []
    best_threshold = 0.50
    best_score = -np.inf

    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        score = threshold_objective(metric, y_thr.to_numpy(), y_prob_thr, float(threshold))
        rows.append({"threshold": float(threshold), metric: float(score)})
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(rows)


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    variants: List[str],
    best_params: Dict[str, Any],
    n_estimators: int,
    cv_splits: int,
    inner_cv_splits: int,
    n_jobs: int,
    threshold_metric: str,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "mcc",
    ]

    fold_scores = {variant: {metric: [] for metric in metric_names} for variant in variants}
    fold_details = {variant: [] for variant in variants}
    threshold_details = {variant: [] for variant in variants}

    for variant in variants:
        print(f"[CV] Running {variant} ...", flush=True)
        for fold_id, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = make_pipeline_for_variant(
                variant,
                X_reference=X_train,
                best_params=best_params,
                n_estimators=n_estimators,
                inner_cv_splits=inner_cv_splits,
                n_jobs=n_jobs,
            )

            start = time.time()
            threshold = 0.50

            if variant == "Full_FeedbackStacking_SMOTE_TunedThreshold":
                threshold, threshold_curve = tune_threshold_on_training_split(
                    pipeline,
                    X_train,
                    y_train,
                    metric=threshold_metric,
                    random_state=RANDOM_STATE + fold_id,
                )
                threshold_details[variant].append(
                    {
                        "fold": fold_id,
                        "selected_threshold": float(threshold),
                        "threshold_metric": threshold_metric,
                        "best_threshold_score": float(threshold_curve[threshold_metric].max()),
                    }
                )

            # Refit on the full training fold after threshold selection.
            final_model = clone(pipeline)
            final_model.fit(X_train, y_train)
            y_prob = final_model.predict_proba(X_val)[:, 1]
            elapsed = time.time() - start

            metrics = compute_metrics(y_val.to_numpy(), y_prob, threshold=threshold)

            for metric in metric_names:
                fold_scores[variant][metric].append(metrics[metric])

            fold_details[variant].append(
                {
                    "fold": fold_id,
                    "runtime_seconds": float(elapsed),
                    **metrics,
                }
            )

    return fold_scores, fold_details, threshold_details


def summarize_cv(fold_scores: Dict[str, Dict[str, List[float]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    paper_rows = []

    for variant, metric_dict in fold_scores.items():
        row = {"Model": variant}
        paper_row = {"Model": variant}

        for metric, values in metric_dict.items():
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            paper_row[metric] = f"{mean:.4f} ± {std:.4f}"

        rows.append(row)
        paper_rows.append(paper_row)

    return pd.DataFrame(rows), pd.DataFrame(paper_rows)


def evaluate_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    variants: List[str],
    best_params: Dict[str, Any],
    n_estimators: int,
    inner_cv_splits: int,
    n_jobs: int,
    test_size: float,
    threshold_metric: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    rows = []
    confusion_matrices = {}
    threshold_details = {}

    for variant in variants:
        print(f"[Hold-out] Running {variant} ...", flush=True)

        pipeline = make_pipeline_for_variant(
            variant,
            X_reference=X_train,
            best_params=best_params,
            n_estimators=n_estimators,
            inner_cv_splits=inner_cv_splits,
            n_jobs=n_jobs,
        )

        threshold = 0.50
        if variant == "Full_FeedbackStacking_SMOTE_TunedThreshold":
            threshold, threshold_curve = tune_threshold_on_training_split(
                pipeline,
                X_train,
                y_train,
                metric=threshold_metric,
                random_state=RANDOM_STATE + 999,
            )
            threshold_details[variant] = {
                "selected_threshold": float(threshold),
                "threshold_metric": threshold_metric,
                "best_threshold_score_on_training_split": float(threshold_curve[threshold_metric].max()),
                "threshold_curve": threshold_curve.to_dict(orient="records"),
            }

        start = time.time()
        final_model = clone(pipeline)
        final_model.fit(X_train, y_train)
        y_prob = final_model.predict_proba(X_test)[:, 1]
        elapsed = time.time() - start

        metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)
        metrics["runtime_seconds"] = float(elapsed)

        rows.append({"Model": variant, **metrics})
        confusion_matrices[variant] = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]

    return pd.DataFrame(rows), confusion_matrices, threshold_details


def wilcoxon_against_full(fold_scores: Dict[str, Dict[str, List[float]]], metric: str = "f1") -> pd.DataFrame:
    target = "Full_FeedbackStacking_SMOTE_TunedThreshold"
    rows = []

    if target not in fold_scores:
        return pd.DataFrame()

    target_scores = np.array(fold_scores[target][metric], dtype=float)

    for variant, metrics in fold_scores.items():
        if variant == target:
            continue

        other_scores = np.array(metrics[metric], dtype=float)
        try:
            stat, p_value = stats.wilcoxon(target_scores, other_scores, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p_value = np.nan, np.nan

        rows.append(
            {
                "Compared_with": variant,
                "metric": metric,
                "wilcoxon_stat": float(stat) if not np.isnan(stat) else np.nan,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return ""

    display_df = df.head(max_rows).copy()
    columns = list(display_df.columns)

    def fmt_value(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in display_df.iterrows():
        rows.append("| " + " | ".join(fmt_value(row[c]) for c in columns) + " |")

    return "\n".join([header, sep] + rows)


def save_summary(
    outdir: Path,
    audit: Dict[str, Any],
    run_config: Dict[str, Any],
    best_params: Dict[str, Any],
    cv_paper: pd.DataFrame,
    holdout: pd.DataFrame,
    wilcoxon: pd.DataFrame,
    threshold_details: Dict[str, Any],
) -> None:
    sections = []
    sections.append("# Step 3 Ablation Study Summary\n")

    sections.append("## Dataset audit\n")
    sections.append(f"- Number of rows: {audit.get('n_rows')}")
    sections.append(f"- Number of columns: {audit.get('n_cols')}")
    sections.append(f"- Duplicate customerID: {audit.get('duplicate_customerID')}")
    sections.append(f"- Missing/invalid TotalCharges after numeric conversion: {audit.get('blank_totalcharges_after_numeric')}")
    sections.append(f"- Churn distribution: {audit.get('target_counts')}")
    sections.append(f"- Churn ratio: {audit.get('target_ratio')}\n")

    sections.append("## Run configuration\n")
    sections.append("```json\n" + json.dumps(run_config, indent=2) + "\n```\n")

    sections.append("## Best feedback-loop configuration used in ablation\n")
    sections.append("```json\n" + json.dumps(best_params, indent=2) + "\n```\n")

    sections.append("## Cross-validation ablation results\n")
    sections.append(markdown_table(cv_paper) + "\n")

    sections.append("## Hold-out ablation results\n")
    holdout_cols = [
        "Model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "mcc",
        "threshold",
    ]
    sections.append(markdown_table(holdout[holdout_cols]) + "\n")

    sections.append("## Wilcoxon signed-rank test on F1-score vs full model\n")
    sections.append(markdown_table(wilcoxon) + "\n")

    sections.append("## Threshold tuning details\n")
    sections.append("```json\n" + json.dumps(threshold_details, indent=2)[:8000] + "\n```\n")

    sections.append("## Manuscript-ready interpretation\n")
    sections.append(
        "The ablation study evaluates the contribution of SMOTE, stacking, feedback-loop optimization, "
        "and threshold calibration under the same leakage-free validation protocol. All preprocessing "
        "steps are fitted only on training folds, and SMOTE is applied exclusively to training folds. "
        "The full model additionally tunes the decision threshold using only training data, thereby "
        "improving churn-class detection without exposing validation or test labels during model fitting."
    )

    (outdir / "ablation_summary_for_paper.md").write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_estimators = 30
        args.cv_splits = 3
        args.inner_cv_splits = 3

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    best_params = load_best_params(args.best_config)
    X, y, audit = load_dataset(args.data)

    variants = [
        "Single_XGBoost_NoSMOTE",
        "Single_XGBoost_SMOTE",
        "Stacking_NoSMOTE_Fixed",
        "Stacking_SMOTE_Fixed",
        "FeedbackStacking_SMOTE_Threshold0.50",
        "Full_FeedbackStacking_SMOTE_TunedThreshold",
    ]

    run_config = {
        "data": args.data,
        "best_config": args.best_config,
        "outdir": args.outdir,
        "n_estimators": args.n_estimators,
        "cv_splits": args.cv_splits,
        "inner_cv_splits": args.inner_cv_splits,
        "n_jobs": args.n_jobs,
        "test_size": args.test_size,
        "threshold_metric": args.threshold_metric,
        "quick": bool(args.quick),
        "variants": variants,
        "python": sys.version,
        "platform": platform.platform(),
    }

    print("[INFO] Starting Step 3 ablation study")
    print(json.dumps(run_config, indent=2), flush=True)

    # Save setup files.
    (outdir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (outdir / "best_params_used.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    # Cross-validation evaluation.
    cv_fold_scores, cv_fold_details, cv_thresholds = evaluate_cv(
        X=X,
        y=y,
        variants=variants,
        best_params=best_params,
        n_estimators=args.n_estimators,
        cv_splits=args.cv_splits,
        inner_cv_splits=args.inner_cv_splits,
        n_jobs=args.n_jobs,
        threshold_metric=args.threshold_metric,
    )

    cv_mean_std, cv_paper = summarize_cv(cv_fold_scores)
    wilcoxon = wilcoxon_against_full(cv_fold_scores, metric="f1")

    cv_mean_std.to_csv(outdir / "ablation_cv_mean_std.csv", index=False)
    cv_paper.to_csv(outdir / "ablation_cv_paper_table.csv", index=False)
    wilcoxon.to_csv(outdir / "ablation_wilcoxon_f1_vs_full.csv", index=False)
    (outdir / "ablation_cv_fold_scores.json").write_text(json.dumps(cv_fold_scores, indent=2), encoding="utf-8")
    (outdir / "ablation_cv_fold_details.json").write_text(json.dumps(cv_fold_details, indent=2), encoding="utf-8")
    (outdir / "ablation_cv_thresholds.json").write_text(json.dumps(cv_thresholds, indent=2), encoding="utf-8")

    # Hold-out evaluation.
    holdout, confusion_matrices, holdout_thresholds = evaluate_holdout(
        X=X,
        y=y,
        variants=variants,
        best_params=best_params,
        n_estimators=args.n_estimators,
        inner_cv_splits=args.inner_cv_splits,
        n_jobs=args.n_jobs,
        test_size=args.test_size,
        threshold_metric=args.threshold_metric,
    )

    holdout.to_csv(outdir / "ablation_holdout_results.csv", index=False)
    (outdir / "ablation_holdout_confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2), encoding="utf-8"
    )
    (outdir / "ablation_thresholds.json").write_text(
        json.dumps({"cv_thresholds": cv_thresholds, "holdout_thresholds": holdout_thresholds}, indent=2),
        encoding="utf-8",
    )

    save_summary(
        outdir=outdir,
        audit=audit,
        run_config=run_config,
        best_params=best_params,
        cv_paper=cv_paper,
        holdout=holdout,
        wilcoxon=wilcoxon,
        threshold_details={"cv_thresholds": cv_thresholds, "holdout_thresholds": holdout_thresholds},
    )

    print("\n[DONE] Step 3 ablation study completed.")
    print(f"[DONE] Output directory: {outdir.resolve()}")
    print("[DONE] Key files:")
    print(f"  - {outdir / 'ablation_cv_paper_table.csv'}")
    print(f"  - {outdir / 'ablation_holdout_results.csv'}")
    print(f"  - {outdir / 'ablation_summary_for_paper.md'}")


if __name__ == "__main__":
    main()
