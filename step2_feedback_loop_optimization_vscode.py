# step2_feedback_loop_optimization_vscode.py
# Leakage-free feedback-loop optimization for IBM Telco Customer Churn.
# Designed for Windows/VS Code. Run with n_jobs=1 for stability.

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_validate, cross_val_predict,
    ParameterSampler
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def safe_metric_dict(y_true, y_pred, y_proba, runtime_seconds=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if runtime_seconds is not None:
        out["runtime_seconds"] = runtime_seconds
    return out


def load_telco_dataset(path):
    df = pd.read_csv(path)
    audit = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_customerID": int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else None,
    }

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        audit["blank_totalcharges_after_numeric"] = int(df["TotalCharges"].isna().sum())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' was not found.")

    target_counts = df["Churn"].value_counts(dropna=False).to_dict()
    audit["target_counts"] = {str(k): int(v) for k, v in target_counts.items()}
    audit["target_ratio"] = {str(k): round(float(v / len(df)), 4) for k, v in target_counts.items()}

    df = df.dropna(subset=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn"])
    return X, y, audit


def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def make_base_models(n_estimators, n_jobs):
    return [
        ("et", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs
        )),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs
        )),
        ("xgb", XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.2,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            tree_method="hist"
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.2,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=-1
        ))
    ]


def make_stacking_pipeline(X, n_estimators=200, inner_cv_splits=5, n_jobs=1):
    preprocess = make_preprocessor(X)
    inner_cv = StratifiedKFold(
        n_splits=inner_cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    stack = StackingClassifier(
        estimators=make_base_models(n_estimators=n_estimators, n_jobs=n_jobs),
        final_estimator=LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight=None,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=False,
        n_jobs=n_jobs
    )

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=1.0, k_neighbors=5)),
        ("model", stack)
    ])
    return pipe


def mean_std(scores, key):
    vals = scores[f"test_{key}"]
    return float(np.mean(vals)), float(np.std(vals))


def run_feedback_loop(X_train, y_train, args, outdir):
    outer_cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "mcc": "matthews_corrcoef",
    }

    # Parameter names follow the imblearn Pipeline:
    # pipeline step "model" is StackingClassifier,
    # inside it, named base estimators are et/rf/xgb/lgbm,
    # final estimator is LogisticRegression.
    param_distributions = {
        "smote__sampling_strategy": [0.6, 0.8, 1.0],
        "smote__k_neighbors": [3, 5],
        "model__final_estimator__C": loguniform(0.05, 20.0),
        "model__final_estimator__class_weight": [None, "balanced"],
        "model__passthrough": [False, True],

        "model__et__max_depth": [6, 8, 10, None],
        "model__rf__max_depth": [6, 8, 10, None],
        "model__xgb__max_depth": [3, 4, 5],
        "model__xgb__learning_rate": [0.05, 0.1, 0.2],
        "model__lgbm__learning_rate": [0.05, 0.1, 0.2],
        "model__lgbm__num_leaves": [15, 31, 63],
    }

    sampler = list(ParameterSampler(
        param_distributions,
        n_iter=args.max_iter,
        random_state=RANDOM_STATE
    ))

    history = []
    best_score = -np.inf
    best_params = None
    no_improve = 0

    for iteration, params in enumerate(sampler, start=1):
        start = time.time()
        estimator = make_stacking_pipeline(
            X_train,
            n_estimators=args.n_estimators,
            inner_cv_splits=args.inner_cv_splits,
            n_jobs=args.n_jobs
        )
        estimator.set_params(**params)

        print(f"[Feedback loop] Iteration {iteration}/{len(sampler)}")
        print("  Params:", params)

        try:
            scores = cross_validate(
                estimator,
                X_train,
                y_train,
                cv=outer_cv,
                scoring=scoring,
                n_jobs=args.n_jobs,
                error_score="raise",
                return_train_score=False
            )
        except Exception as exc:
            row = {
                "iteration": iteration,
                "status": "failed",
                "error": repr(exc),
                "params": json.dumps(params, default=str)
            }
            history.append(row)
            pd.DataFrame(history).to_csv(outdir / "feedback_loop_history.csv", index=False)
            print("  FAILED:", repr(exc))
            continue

        runtime = time.time() - start
        f1_mean, f1_std = mean_std(scores, "f1")
        auc_mean, auc_std = mean_std(scores, "roc_auc")
        bal_mean, bal_std = mean_std(scores, "balanced_accuracy")
        mcc_mean, mcc_std = mean_std(scores, "mcc")
        acc_mean, acc_std = mean_std(scores, "accuracy")
        pre_mean, pre_std = mean_std(scores, "precision")
        rec_mean, rec_std = mean_std(scores, "recall")

        current_score = {
            "f1": f1_mean,
            "roc_auc": auc_mean,
            "balanced_accuracy": bal_mean
        }[args.refit_metric]

        improved = current_score > best_score + args.min_delta
        if improved:
            best_score = current_score
            best_params = params
            no_improve = 0
            status = "best"
        else:
            no_improve += 1
            status = "no_improvement"

        row = {
            "iteration": iteration,
            "status": status,
            "refit_metric": args.refit_metric,
            "refit_score_mean": current_score,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "balanced_accuracy_mean": bal_mean,
            "balanced_accuracy_std": bal_std,
            "precision_mean": pre_mean,
            "precision_std": pre_std,
            "recall_sensitivity_mean": rec_mean,
            "recall_sensitivity_std": rec_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "roc_auc_mean": auc_mean,
            "roc_auc_std": auc_std,
            "mcc_mean": mcc_mean,
            "mcc_std": mcc_std,
            "runtime_seconds": runtime,
            "no_improve_count": no_improve,
            "params": json.dumps(params, default=str),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(outdir / "feedback_loop_history.csv", index=False)

        print(f"  {args.refit_metric}={current_score:.4f}, best={best_score:.4f}, status={status}")

        if no_improve >= args.patience:
            print(f"[Feedback loop] Early stopping: no improvement for {args.patience} iterations.")
            break

    if best_params is None:
        raise RuntimeError("All feedback-loop candidates failed.")

    best_config = {
        "best_refit_metric": args.refit_metric,
        "best_refit_score_cv_mean": best_score,
        "best_params": best_params,
        "stopping_rule": {
            "max_iter": args.max_iter,
            "patience": args.patience,
            "min_delta": args.min_delta
        }
    }
    with open(outdir / "best_feedback_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, default=str)

    return best_params, pd.DataFrame(history)


def threshold_search(y_true, y_proba, metric="f1"):
    thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
    rows = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        metrics = safe_metric_dict(y_true, y_pred, y_proba)
        metrics["threshold"] = float(thr)
        rows.append(metrics)
    df = pd.DataFrame(rows)
    best_idx = df[metric].idxmax()
    return float(df.loc[best_idx, "threshold"]), df


def evaluate_best_model(X_train, X_test, y_train, y_test, best_params, args, outdir):
    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=RANDOM_STATE)

    best_pipe = make_stacking_pipeline(
        X_train,
        n_estimators=args.n_estimators,
        inner_cv_splits=args.inner_cv_splits,
        n_jobs=args.n_jobs
    )
    best_pipe.set_params(**best_params)

    print("[Threshold tuning] Generating out-of-fold training probabilities...")
    train_oof_proba = cross_val_predict(
        best_pipe,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=args.n_jobs
    )[:, 1]

    best_thr, thr_df = threshold_search(y_train, train_oof_proba, metric=args.threshold_metric)
    thr_df.to_csv(outdir / "threshold_search_train.csv", index=False)

    print(f"[Final fit] Fitting best pipeline on full training set. Best threshold={best_thr:.2f}")
    start = time.time()
    best_pipe.fit(X_train, y_train)
    fit_runtime = time.time() - start

    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    pred_default = (test_proba >= 0.50).astype(int)
    pred_tuned = (test_proba >= best_thr).astype(int)

    default_metrics = safe_metric_dict(y_test, pred_default, test_proba, runtime_seconds=fit_runtime)
    tuned_metrics = safe_metric_dict(y_test, pred_tuned, test_proba, runtime_seconds=fit_runtime)

    pd.DataFrame([{"threshold": 0.50, **default_metrics}]).to_csv(
        outdir / "holdout_feedback_default_threshold.csv", index=False
    )
    pd.DataFrame([{"threshold": best_thr, **tuned_metrics}]).to_csv(
        outdir / "holdout_feedback_tuned_threshold.csv", index=False
    )

    with open(outdir / "holdout_feedback_confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump({
            "default_threshold_0.50": confusion_matrix(y_test, pred_default, labels=[0, 1]).tolist(),
            f"tuned_threshold_{best_thr:.2f}": confusion_matrix(y_test, pred_tuned, labels=[0, 1]).tolist(),
        }, f, indent=2)

    with open(outdir / "final_feedback_model_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_thr,
            "default_threshold_metrics": default_metrics,
            "tuned_threshold_metrics": tuned_metrics,
        }, f, indent=2)

    return best_thr, default_metrics, tuned_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--outdir", default="step2_feedback_outputs", help="Output directory")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--inner_cv_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=20, help="Maximum feedback-loop candidates")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement for feedback loop")
    parser.add_argument("--refit_metric", choices=["f1", "roc_auc", "balanced_accuracy"], default="f1")
    parser.add_argument("--threshold_metric", choices=["f1", "balanced_accuracy", "mcc"], default="f1")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_estimators = min(args.n_estimators, 50)
        args.cv_splits = min(args.cv_splits, 3)
        args.inner_cv_splits = min(args.inner_cv_splits, 3)
        args.max_iter = min(args.max_iter, 5)
        args.patience = min(args.patience, 3)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    X, y, audit = load_telco_dataset(args.data)
    with open(outdir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("[Data]")
    print("  Full data:", X.shape, "Class counts:", y.value_counts().to_dict())
    print("  Train:", X_train.shape, "Test:", X_test.shape)

    best_params, history = run_feedback_loop(X_train, y_train, args, outdir)
    best_thr, default_metrics, tuned_metrics = evaluate_best_model(
        X_train, X_test, y_train, y_test, best_params, args, outdir
    )

    summary = []
    summary.append("# Step 2 Feedback-loop optimization summary\n")
    summary.append("## Best configuration\n")
    summary.append(json.dumps({"best_params": best_params, "best_threshold": best_thr}, indent=2, default=str))
    summary.append("\n\n## Hold-out results, default threshold 0.50\n")
    summary.append(json.dumps(default_metrics, indent=2))
    summary.append("\n\n## Hold-out results, tuned threshold\n")
    summary.append(json.dumps(tuned_metrics, indent=2))
    summary.append("\n\n## Interpretation\n")
    summary.append(
        "The feedback loop evaluates candidate configurations using Stratified K-Fold Cross-Validation "
        "on the training set only. It updates the best configuration when the selected refit metric improves "
        "by at least min_delta and stops when no improvement is observed for the specified patience. "
        "The final selected pipeline is then evaluated on an untouched stratified hold-out test set."
    )
    (outdir / "step2_feedback_summary_for_paper.md").write_text("\n".join(summary), encoding="utf-8")

    print("[Done] Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
