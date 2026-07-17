#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: Explainability / feature contribution analysis for the revised Telco churn paper.

This script is designed for Windows + VS Code and follows the leakage-free protocol:
- split data before preprocessing
- fit preprocessing only on training data
- apply SMOTE only on training data
- evaluate/explain on untouched hold-out data

Outputs:
- xai_permutation_importance.csv
- xai_permutation_importance_top15.png
- xai_xgboost_feature_importance.csv
- xai_xgboost_feature_importance_top15.png
- optional SHAP outputs if shap is installed:
    - xai_shap_preprocessed_features.csv
    - xai_shap_original_features.csv
    - xai_shap_original_top15.png
- xai_summary_for_paper.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Safer behavior on Windows/joblib.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42


def read_data(path: str | Path) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
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

    # Target.
    if "Churn" not in df.columns:
        raise ValueError("The dataset must contain a 'Churn' column.")

    audit["target_counts"] = {str(k): int(v) for k, v in df["Churn"].value_counts().to_dict().items()}
    audit["target_ratio"] = {str(k): round(float(v), 4) for k, v in df["Churn"].value_counts(normalize=True).to_dict().items()}

    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected target values in Churn. Expected only 'Yes' and 'No'.")

    X = df.drop(columns=["Churn"])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    return X, y.astype(int), audit


def make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_param(best_params: Dict[str, Any], key: str, default: Any) -> Any:
    return best_params.get(key, default)


def make_base_models(best_params: Dict[str, Any], n_estimators: int, n_jobs: int) -> List[Tuple[str, Any]]:
    et_depth = get_param(best_params, "model__et__max_depth", 10)
    rf_depth = get_param(best_params, "model__rf__max_depth", 10)
    xgb_depth = get_param(best_params, "model__xgb__max_depth", 4)
    xgb_lr = get_param(best_params, "model__xgb__learning_rate", 0.05)
    lgbm_lr = get_param(best_params, "model__lgbm__learning_rate", 0.05)
    lgbm_leaves = get_param(best_params, "model__lgbm__num_leaves", 15)

    return [
        ("et", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=et_depth,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            class_weight=None,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=rf_depth,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            class_weight=None,
        )),
        ("xgb", XGBClassifier(
            n_estimators=n_estimators,
            max_depth=xgb_depth,
            learning_rate=xgb_lr,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            tree_method="hist",
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=lgbm_lr,
            num_leaves=lgbm_leaves,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=-1,
        )),
    ]


def make_full_pipeline(
    X: pd.DataFrame,
    best_params: Dict[str, Any],
    n_estimators: int,
    inner_cv_splits: int,
    n_jobs: int,
) -> ImbPipeline:
    preprocess = make_preprocess(X)
    base_models = make_base_models(best_params, n_estimators=n_estimators, n_jobs=n_jobs)

    final_C = get_param(best_params, "model__final_estimator__C", 1.0)
    final_class_weight = get_param(best_params, "model__final_estimator__class_weight", "balanced")
    passthrough = bool(get_param(best_params, "model__passthrough", True))

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            C=final_C,
            class_weight=final_class_weight,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        cv=inner_cv_splits,
        stack_method="predict_proba",
        passthrough=passthrough,
        n_jobs=n_jobs,
    )

    smote_strategy = get_param(best_params, "smote__sampling_strategy", 0.6)
    smote_k = get_param(best_params, "smote__k_neighbors", 3)

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("smote", SMOTE(
            sampling_strategy=smote_strategy,
            k_neighbors=smote_k,
            random_state=RANDOM_STATE,
        )),
        ("model", stack),
    ])
    return pipe


def metrics_from_proba(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_top_bar(df: pd.DataFrame, value_col: str, label_col: str, title: str, outpath: Path, top_n: int = 15) -> None:
    d = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.32 * len(d))))
    plt.barh(d[label_col].astype(str), d[value_col].astype(float))
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def tuned_f1_scorer(threshold: float):
    def _score(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return f1_score(y, pred, zero_division=0)
    return _score


def safe_get_feature_names(preprocess: ColumnTransformer) -> List[str]:
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(preprocess.transformers_[0][1].shape[1])]


def original_feature_from_transformed(name: str) -> str:
    # Examples:
    # num__tenure -> tenure
    # cat__Contract_Month-to-month -> Contract
    # cat__InternetService_Fiber optic -> InternetService
    if "__" in name:
        _, rest = name.split("__", 1)
    else:
        rest = name
    # For one-hot encoded names, the original feature is before first "_" where possible.
    # Because original Telco feature names do not contain underscores.
    if "_" in rest:
        return rest.split("_", 1)[0]
    return rest


def run_optional_shap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
    outdir: Path,
    n_estimators: int,
    n_jobs: int,
    shap_sample: int,
) -> str:
    """
    Train a representative XGBoost base learner under the same preprocessing+SMOTE setup
    and compute SHAP-style feature contributions.

    Notes for Windows/VS Code:
    - Some combinations of shap and xgboost>=3.x fail when TreeExplainer parses XGBoost's
      base_score metadata. In that case, this function automatically falls back to
      XGBoost's native TreeSHAP contribution output via Booster.predict(pred_contribs=True).
    - The script should never stop only because optional SHAP fails.
    """
    shap_available = True
    try:
        import shap  # type: ignore
    except Exception as exc:
        shap_available = False
        shap_import_error = exc
    else:
        shap_import_error = None

    preprocess = make_preprocess(X_train)
    X_train_t = preprocess.fit_transform(X_train)
    X_test_t = preprocess.transform(X_test)
    feature_names = list(preprocess.get_feature_names_out())

    smote = SMOTE(
        sampling_strategy=get_param(best_params, "smote__sampling_strategy", 0.6),
        k_neighbors=get_param(best_params, "smote__k_neighbors", 3),
        random_state=RANDOM_STATE,
    )
    X_train_bal, y_train_bal = smote.fit_resample(X_train_t, y_train)

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=get_param(best_params, "model__xgb__max_depth", 4),
        learning_rate=get_param(best_params, "model__xgb__learning_rate", 0.05),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    xgb.fit(X_train_bal, y_train_bal)

    # Use a sample to keep runtime/memory safe.
    n = min(shap_sample, X_test_t.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X_test_t.shape[0], size=n, replace=False)
    X_sample = X_test_t[idx]

    shap_note = ""
    try:
        if not shap_available:
            raise RuntimeError(f"Python shap package is not available: {shap_import_error}")

        # shap.TreeExplainer may fail with recent XGBoost metadata formats.
        if sparse.issparse(X_sample):
            X_sample_for_shap = X_sample.toarray()
        else:
            X_sample_for_shap = np.asarray(X_sample)

        explainer = shap.TreeExplainer(xgb)  # type: ignore[name-defined]
        shap_values = explainer.shap_values(X_sample_for_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)

        shap_note = "SHAP analysis completed using shap.TreeExplainer for the XGBoost base learner."

    except Exception as exc:
        # Robust fallback: XGBoost's native TreeSHAP contribution values.
        # It returns p feature contributions plus one bias column; exclude the bias term.
        try:
            import xgboost as xgb_native  # type: ignore
            dm = xgb_native.DMatrix(X_sample, feature_names=feature_names)
            contribs = xgb.get_booster().predict(dm, pred_contribs=True)
            shap_values = np.asarray(contribs)[:, :-1]
            shap_note = (
                "shap.TreeExplainer was not used because it failed or was unavailable. "
                "The script used XGBoost native TreeSHAP contributions via "
                "Booster.predict(pred_contribs=True). "
                f"Original SHAP error: {exc}"
            )
            (outdir / "xai_shap_fallback_note.txt").write_text(shap_note + "\n", encoding="utf-8")
        except Exception as fallback_exc:
            note = (
                "SHAP-style analysis was skipped because both shap.TreeExplainer and "
                "XGBoost native pred_contribs failed. The manuscript can still report "
                "permutation importance and XGBoost feature importance. "
                f"TreeExplainer/import error: {exc}; native fallback error: {fallback_exc}"
            )
            (outdir / "xai_shap_not_run.txt").write_text(note + "\n", encoding="utf-8")
            return note

    # Save transformed-feature SHAP-style contributions.
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Defensive handling if a library returns unexpected length.
    if len(mean_abs) != len(feature_names):
        n_common = min(len(mean_abs), len(feature_names))
        mean_abs = mean_abs[:n_common]
        used_feature_names = feature_names[:n_common]
    else:
        used_feature_names = feature_names

    shap_df = pd.DataFrame({
        "transformed_feature": used_feature_names,
        "mean_abs_shap": mean_abs,
        "original_feature": [original_feature_from_transformed(f) for f in used_feature_names],
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(outdir / "xai_shap_preprocessed_features.csv", index=False)

    agg = (
        shap_df.groupby("original_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    agg.to_csv(outdir / "xai_shap_original_features.csv", index=False)
    plot_top_bar(
        agg,
        value_col="mean_abs_shap",
        label_col="original_feature",
        title="Aggregated SHAP-style importance by original feature (XGBoost base learner)",
        outpath=outdir / "xai_shap_original_top15.png",
        top_n=15,
    )

    return shap_note + " Contributions were aggregated to the original Telco features."

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--best_config", required=True, help="Path to best_feedback_config.json from Step 2")
    parser.add_argument("--outdir", default="step4_xai_full")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--inner_cv_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.42, help="Decision threshold selected from Step 3")
    parser.add_argument("--n_repeats", type=int, default=10, help="Permutation importance repeats")
    parser.add_argument("--shap_sample", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_estimators = min(args.n_estimators, 50)
        args.inner_cv_splits = min(args.inner_cv_splits, 3)
        args.n_repeats = min(args.n_repeats, 3)
        args.shap_sample = min(args.shap_sample, 150)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, audit = read_data(args.data)
    with open(args.best_config, "r", encoding="utf-8") as f:
        best_config_json = json.load(f)
    best_params = best_config_json.get("best_params", best_config_json)

    run_config = {
        "data": args.data,
        "best_config": args.best_config,
        "outdir": str(outdir),
        "n_estimators": args.n_estimators,
        "inner_cv_splits": args.inner_cv_splits,
        "n_jobs": args.n_jobs,
        "threshold": args.threshold,
        "n_repeats": args.n_repeats,
        "shap_sample": args.shap_sample,
        "quick": args.quick,
        "python": sys.version,
        "platform": sys.platform,
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (outdir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (outdir / "best_params_used.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    # Hold-out split consistent with the previous protocol.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("[Step 4] Fitting feedback-optimized stacking pipeline...")
    t0 = time.time()
    pipe = make_full_pipeline(
        X_train,
        best_params=best_params,
        n_estimators=args.n_estimators,
        inner_cv_splits=args.inner_cv_splits,
        n_jobs=args.n_jobs,
    )
    pipe.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = metrics_from_proba(y_test, y_proba, threshold=args.threshold)
    metrics["fit_seconds"] = float(fit_seconds)
    (outdir / "xai_holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[Step 4] Running permutation importance for full pipeline...")
    pi = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring=tuned_f1_scorer(args.threshold),
        n_repeats=args.n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
    )
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(outdir / "xai_permutation_importance.csv", index=False)
    plot_top_bar(
        perm_df,
        value_col="importance_mean",
        label_col="feature",
        title=f"Permutation importance for full stacking pipeline (F1, threshold={args.threshold})",
        outpath=outdir / "xai_permutation_importance_top15.png",
        top_n=15,
    )

    # Representative XGBoost feature importance under preprocessing + SMOTE.
    print("[Step 4] Training representative XGBoost base learner for feature importance...")
    preprocess = make_preprocess(X_train)
    X_train_t = preprocess.fit_transform(X_train)
    feature_names = list(preprocess.get_feature_names_out())
    smote = SMOTE(
        sampling_strategy=get_param(best_params, "smote__sampling_strategy", 0.6),
        k_neighbors=get_param(best_params, "smote__k_neighbors", 3),
        random_state=RANDOM_STATE,
    )
    X_train_bal, y_train_bal = smote.fit_resample(X_train_t, y_train)

    xgb = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=get_param(best_params, "model__xgb__max_depth", 4),
        learning_rate=get_param(best_params, "model__xgb__learning_rate", 0.05),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
        tree_method="hist",
    )
    xgb.fit(X_train_bal, y_train_bal)
    xgb_imp = pd.DataFrame({
        "transformed_feature": feature_names,
        "importance": xgb.feature_importances_,
        "original_feature": [original_feature_from_transformed(f) for f in feature_names],
    }).sort_values("importance", ascending=False)
    xgb_imp.to_csv(outdir / "xai_xgboost_feature_importance.csv", index=False)

    xgb_agg = (
        xgb_imp.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    xgb_agg.to_csv(outdir / "xai_xgboost_feature_importance_original.csv", index=False)
    plot_top_bar(
        xgb_agg,
        value_col="importance",
        label_col="original_feature",
        title="Aggregated XGBoost feature importance by original feature",
        outpath=outdir / "xai_xgboost_feature_importance_top15.png",
        top_n=15,
    )

    print("[Step 4] Attempting optional SHAP analysis...")
    shap_note = run_optional_shap(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        best_params=best_params,
        outdir=outdir,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        shap_sample=args.shap_sample,
    )

    # Markdown summary.
    top_perm = perm_df.head(10)
    top_xgb = xgb_agg.head(10)

    summary_lines = []
    summary_lines.append("# Step 4 Explainability / Feature Contribution Summary\n")
    summary_lines.append("## Dataset audit\n")
    summary_lines.append(f"- Number of rows: {audit.get('n_rows')}")
    summary_lines.append(f"- Number of columns: {audit.get('n_cols')}")
    summary_lines.append(f"- Duplicate customerID: {audit.get('duplicate_customerID')}")
    summary_lines.append(f"- Missing/invalid TotalCharges after numeric conversion: {audit.get('blank_totalcharges_after_numeric')}")
    summary_lines.append(f"- Churn distribution: {audit.get('target_counts')}")
    summary_lines.append(f"- Churn ratio: {audit.get('target_ratio')}\n")
    summary_lines.append("## Hold-out performance of explained full model\n")
    for k, v in metrics.items():
        summary_lines.append(f"- {k}: {v}")
    summary_lines.append("\n## Top features by permutation importance for full stacking pipeline\n")
    summary_lines.append(top_perm.to_markdown(index=False) if hasattr(top_perm, "to_markdown") else top_perm.to_string(index=False))
    summary_lines.append("\n## Top original features by aggregated XGBoost importance\n")
    summary_lines.append(top_xgb.to_markdown(index=False) if hasattr(top_xgb, "to_markdown") else top_xgb.to_string(index=False))
    summary_lines.append("\n## SHAP status\n")
    summary_lines.append(shap_note)
    summary_lines.append("\n## Suggested manuscript interpretation\n")
    summary_lines.append(
        "The explainability analysis was conducted on an untouched hold-out set using the leakage-free pipeline. "
        "Permutation importance was computed for the full feedback-optimized stacking model using F1-score at the selected threshold. "
        "In addition, feature contribution was examined using a representative XGBoost base learner trained under the same preprocessing and SMOTE protocol. "
        "The resulting feature rankings provide practical insight into customer attributes most associated with churn-risk detection."
    )

    (outdir / "xai_summary_for_paper.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[Step 4] Completed. Outputs saved in: {outdir}")


if __name__ == "__main__":
    main()
