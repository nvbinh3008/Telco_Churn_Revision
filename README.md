# Telco Churn Revision

This repository contains the Python implementation scripts used for the revised manuscript:

**Enhancing Telecom Customer Churn Prediction with a Leakage-Free Feedback-Optimized Stacking Model**

## Repository contents

- `step1_leakage_free_experiment_vscode.py`  
  Leakage-free model comparison using stratified splitting/cross-validation, preprocessing inside the training fold, SMOTE applied only to training folds, and probability-based ROC-AUC.

- `step2_feedback_loop_optimization_vscode.py`  
  Feedback-loop optimization for SMOTE settings, base-learner hyperparameters, Logistic Regression meta-model settings, and decision-threshold refinement.

- `step3_ablation_study_vscode.py`  
  Ablation experiments comparing the contribution of SMOTE, stacking, feedback-loop optimization, and threshold tuning.

- `step4_xai_feature_importance_vscode_fixed.py`  
  Explainability analysis using permutation importance for the full stacking pipeline and TreeSHAP-style analysis for the XGBoost base learner.

- `step5_roc_curves_vscode.py`  
  ROC-curve generation for individual base learners, voting ensemble, and feedback-optimized stacking.

## Dataset

The experiments use the public IBM Telco Customer Churn dataset. Download the dataset and place it in the expected data path used by the scripts, or update the dataset path variable in each script before running.

## Environment

Install the required packages with:

```bash
pip install -r requirements.txt
```

To reduce CPU/threading issues on Windows, the experiments can be run with single-thread settings, for example in PowerShell:

```powershell
$env:LOKY_MAX_CPU_COUNT="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
```

## Notes on runtime

The original experimental scripts did not record wall-clock runtime logs. Therefore, the manuscript reports qualitative computational overhead rather than exact runtime values. Runtime may vary depending on hardware, package versions, parallelization settings, and operating system.

## Reproducibility

The scripts are intended to reproduce the major experimental components reported in the manuscript, including leakage-free validation, feedback-loop optimization, ablation analysis, explainability analysis, and ROC-curve visualization.
