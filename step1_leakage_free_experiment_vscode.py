# step1_leakage_free_experiment_vscode.py
# -*- coding: utf-8 -*-
"""
BƯỚC 1 - Thực nghiệm Telco Churn theo pipeline KHÔNG RÒ RỈ DỮ LIỆU
Tương thích Visual Studio Code / Windows / macOS / Linux.

Điểm sửa chính so với bản thảo:
1) Train/test split và Stratified K-Fold được thực hiện TRƯỚC khi SMOTE.
2) Imputer, scaler, one-hot encoder chỉ được fit trên train fold.
3) SMOTE chỉ áp dụng trên train fold thông qua imblearn.Pipeline.
4) ROC-AUC được tính bằng xác suất predict_proba[:, 1], không dùng nhãn cứng.
5) Stacking dùng out-of-fold probabilities và Logistic Regression meta-model.
6) Xuất mean ± std cho các metrics phục vụ sửa Table 4 trong paper.

Ví dụ chạy nhanh trong VS Code terminal:
python step1_leakage_free_experiment_vscode.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir step1_outputs_quick --quick

Ví dụ chạy đầy đủ hơn:
python step1_leakage_free_experiment_vscode.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir step1_outputs_full --n_estimators 200 --cv_splits 5 --inner_cv_splits 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
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
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42


def package_version(package_name: str) -> str:
    """Lấy version package để ghi vào output phục vụ reproducibility."""
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "not_installed"


def save_environment_report(outdir: Path) -> None:
    """Ghi thông tin môi trường chạy để báo cáo trong paper/appendix."""
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("sklearn"),
            "imbalanced_learn": package_version("imblearn"),
            "xgboost": package_version("xgboost"),
            "lightgbm": package_version("lightgbm"),
            "scipy": package_version("scipy"),
        },
    }
    with open(outdir / "environment_report.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)


def load_telco(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Dict]:
    """Đọc dữ liệu IBM Telco, sửa kiểu TotalCharges, bỏ customerID, mã hóa target."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)

    required_cols = {"Churn", "TotalCharges"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột bắt buộc: {missing}")

    # TotalCharges trong file gốc là object và có 11 giá trị rỗng.
    # Chỉ convert sang numeric; giá trị thiếu được impute TRONG pipeline CV.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    audit = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "blank_totalcharges_after_numeric": int(df["TotalCharges"].isna().sum()),
        "duplicate_customerID": int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else None,
        "target_counts": {str(k): int(v) for k, v in df["Churn"].value_counts().to_dict().items()},
        "target_ratio": {str(k): float(v) for k, v in df["Churn"].value_counts(normalize=True).round(4).to_dict().items()},
    }

    # Bỏ các dòng target khác Yes/No nếu có.
    df = df[df["Churn"].isin(["No", "Yes"])].copy()

    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return X, y, numeric_features, categorical_features, audit


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Preprocessing được fit trong từng train fold, không fit trên toàn bộ dataset."""
    try:
        # sklearn >= 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_models(n_estimators: int = 200, n_jobs: int = 1, inner_cv_splits: int = 5) -> Dict:
    """Các mô hình đúng với mô tả trong bài; meta-model là Logistic Regression."""
    extra_trees = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )

    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )

    xgboost = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        tree_method="hist",
    )

    lightgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.2,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=-1,
    )

    base_estimators = [
        ("ExtraTrees", extra_trees),
        ("RandomForest", random_forest),
        ("XGBoost", xgboost),
        ("LightGBM", lightgbm),
    ]

    soft_voting = VotingClassifier(
        estimators=base_estimators,
        voting="soft",
        n_jobs=n_jobs,
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    optimized_stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=inner_cv,  # tạo out-of-fold predictions cho meta-model
        n_jobs=n_jobs,
        passthrough=False,
    )

    return {
        "ExtraTrees": extra_trees,
        "RandomForest": random_forest,
        "XGBoost": xgboost,
        "LightGBM": lightgbm,
        "SoftVoting": soft_voting,
        "OptimizedStacking": optimized_stacking,
    }


def parse_model_selection(models: Dict, selected: str) -> Dict:
    """Cho phép chạy một vài mô hình để test nhanh trong VS Code."""
    if selected.lower() in {"all", "*"}:
        return models

    wanted = [m.strip() for m in selected.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in models]
    if unknown:
        raise ValueError(f"Tên model không hợp lệ: {unknown}. Hợp lệ: {list(models.keys())}")

    return {name: models[name] for name in wanted}


def make_leakage_free_pipeline(preprocess: ColumnTransformer, model) -> ImbPipeline:
    """
    Thứ tự đúng:
    preprocessing fit trên train fold -> SMOTE chỉ trên train fold -> model.
    Khi cross_validate chia fold, imblearn.Pipeline bảo đảm SMOTE không chạm vào validation/test fold.
    """
    return ImbPipeline(
        steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def specificity_score_func(y_true, y_pred) -> float:
    """Specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return 0.0 if denom == 0 else tn / denom


def get_scoring() -> Dict:
    """Scorers dùng trong Stratified K-Fold CV."""
    return {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall_sensitivity": make_scorer(recall_score, zero_division=0),
        "specificity": make_scorer(specificity_score_func),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",  # sklearn tự dùng predict_proba/decsion_function
        "mcc": make_scorer(matthews_corrcoef),
    }


def summarize_cv_scores(cvres: Dict, metrics: List[str]) -> Dict:
    row = {}
    for metric_name in metrics:
        values = np.asarray(cvres[f"test_{metric_name}"], dtype=float)
        row[f"{metric_name}_mean"] = float(np.mean(values))
        row[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row[f"{metric_name}_folds"] = [float(v) for v in values]
    return row


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    preprocess: ColumnTransformer,
    models: Dict,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict]:
    scoring = get_scoring()
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fold_scores = {}

    for name, model in models.items():
        print(f"\n[CV] Running {name} ...", flush=True)
        start = time.time()

        pipe = make_leakage_free_pipeline(preprocess, model)
        cvres = cross_validate(
            pipe,
            X,
            y,
            scoring=scoring,
            cv=outer_cv,
            n_jobs=1,  # tránh nested parallel quá tải trên VS Code/Windows
            return_train_score=False,
            error_score="raise",
        )

        summary = {"Model": name, "runtime_seconds": round(time.time() - start, 3)}
        summary.update(summarize_cv_scores(cvres, list(scoring.keys())))
        rows.append(summary)
        fold_scores[name] = {
            m: [float(v) for v in cvres[f"test_{m}"]] for m in scoring.keys()
        }

        print(f"[CV] Finished {name} in {summary['runtime_seconds']} seconds.", flush=True)

    return pd.DataFrame(rows), fold_scores


def evaluate_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    preprocess: ColumnTransformer,
    models: Dict,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, Dict]:
    """Đánh giá test hold-out cuối, tuyệt đối không tuning trên test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    rows = []
    confusion = {}

    for name, model in models.items():
        print(f"\n[Hold-out] Running {name} ...", flush=True)
        start = time.time()

        pipe = make_leakage_free_pipeline(preprocess, model)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        row = {
            "Model": name,
            "runtime_seconds": round(time.time() - start, 3),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall_sensitivity": recall_score(y_test, y_pred, zero_division=0),
            "specificity": specificity_score_func(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "mcc": matthews_corrcoef(y_test, y_pred),
        }
        rows.append(row)
        confusion[name] = cm.tolist()

        print(f"[Hold-out] Finished {name} in {row['runtime_seconds']} seconds.", flush=True)

    return pd.DataFrame(rows), confusion


def wilcoxon_against_stacking(fold_scores: Dict, metric: str = "f1") -> pd.DataFrame:
    """
    Kiểm định sơ bộ: so từng mô hình với OptimizedStacking trên cùng outer folds.
    Với bài báo, bước Statistical Validation tiếp theo nên bổ sung Holm correction/Friedman/Nemenyi.
    """
    target = "OptimizedStacking"
    if target not in fold_scores:
        return pd.DataFrame()

    rows = []
    stack_scores = np.array(fold_scores[target][metric])
    for name, scores_dict in fold_scores.items():
        if name == target:
            continue

        other_scores = np.array(scores_dict[metric])
        try:
            stat, p_value = wilcoxon(stack_scores, other_scores, alternative="greater")
            rows.append(
                {
                    "Compared_with": name,
                    "metric": metric,
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_value),
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "Compared_with": name,
                    "metric": metric,
                    "wilcoxon_stat": np.nan,
                    "p_value": np.nan,
                    "note": str(exc),
                }
            )

    return pd.DataFrame(rows)


def make_paper_table(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng gọn dạng mean ± std để copy vào manuscript."""
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "mcc",
    ]

    rows = []
    for _, row in cv_results.iterrows():
        out = {"Model": row["Model"]}
        for m in metrics:
            out[m] = f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
        out["runtime_seconds"] = row.get("runtime_seconds", np.nan)
        rows.append(out)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free Telco Churn experiment for paper revision.")
    parser.add_argument("--data", type=str, required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--outdir", type=str, default="step1_outputs")
    parser.add_argument("--n_estimators", type=int, default=200, help="50 for quick test, 200 for paper-level run")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--inner_cv_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1, help="Dùng 1 cho Windows/VS Code ổn định; có thể tăng 2-4 nếu máy khỏe")
    parser.add_argument("--models", type=str, default="all", help="all hoặc ví dụ: ExtraTrees,XGBoost,OptimizedStacking")
    parser.add_argument("--skip_cv", action="store_true", help="Chỉ chạy hold-out test")
    parser.add_argument("--skip_holdout", action="store_true", help="Chỉ chạy cross-validation")
    parser.add_argument("--quick", action="store_true", help="Preset test nhanh: n_estimators=30, cv_splits=3, inner_cv_splits=3")
    args = parser.parse_args()

    if args.quick:
        args.n_estimators = 30
        args.cv_splits = 3
        args.inner_cv_splits = 3
        args.n_jobs = 1

    warnings.filterwarnings("ignore")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_environment_report(outdir)

    X, y, numeric_features, categorical_features, audit = load_telco(args.data)
    preprocess = build_preprocessor(numeric_features, categorical_features)
    all_models = build_models(
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        inner_cv_splits=args.inner_cv_splits,
    )
    models = parse_model_selection(all_models, args.models)

    with open(outdir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\nDATASET AUDIT", flush=True)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    print("Numeric features:", numeric_features, flush=True)
    print("Categorical features:", categorical_features, flush=True)
    print("Selected models:", list(models.keys()), flush=True)

    if not args.skip_cv:
        cv_results, fold_scores = run_cross_validation(
            X,
            y,
            preprocess,
            models,
            n_splits=args.cv_splits,
        )

        cv_results.to_csv(outdir / "cv_mean_std_results.csv", index=False)
        paper_table = make_paper_table(cv_results)
        paper_table.to_csv(outdir / "cv_paper_table_mean_std.csv", index=False)

        with open(outdir / "cv_fold_scores.json", "w", encoding="utf-8") as f:
            json.dump(fold_scores, f, indent=2, ensure_ascii=False)

        stat_f1 = wilcoxon_against_stacking(fold_scores, metric="f1")
        stat_f1.to_csv(outdir / "wilcoxon_f1_vs_stacking.csv", index=False)

        print("\nCV PAPER TABLE: mean ± std", flush=True)
        print(paper_table.to_string(index=False), flush=True)

    if not args.skip_holdout:
        holdout_results, confusion = evaluate_holdout(X, y, preprocess, models)
        holdout_results.to_csv(outdir / "holdout_results.csv", index=False)

        with open(outdir / "holdout_confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(confusion, f, indent=2, ensure_ascii=False)

        print("\nHOLD-OUT RESULTS", flush=True)
        print(holdout_results.to_string(index=False), flush=True)

    print(f"\nDone. Output folder: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    # Quan trọng cho Windows/VS Code khi estimator dùng parallel internally.
    main()
