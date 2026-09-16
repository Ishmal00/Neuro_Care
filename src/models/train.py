"""Train and evaluate one shared XGBoost model for SeizeIT2."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


DATA_PATH = "data/seizeit2_features.csv"
MODEL_PATH = "models/xgb_seizeit2.pkl"
PREDICTIONS_PATH = "data/xgb_predictions.csv"
IMPORTANCE_PATH = "data/xgb_feature_importance.csv"
LOG_PATH = "logs/train_xgb.log"

TARGET_COL = "Label"
PATIENT_COL = "patient_id"
GROUP_COLUMNS = ["patient_id", "run_id", "window_start_time"]
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "EEG_delta",
    "EEG_theta",
    "EEG_alpha",
    "EEG_beta",
    "EEG_gamma",
    "ECG_HR_mean",
    "ECG_RMSSD",
    "ECG_SDNN",
    "EMG_mean",
    "EMG_std",
    "ACC_mean",
    "ACC_std",
    "ACC_energy",
    "EEG_total_power",
    "EEG_delta_relative",
    "EEG_theta_relative",
    "EEG_alpha_relative",
    "EEG_beta_relative",
    "EEG_gamma_relative",
    "EEG_theta_alpha_ratio",
    "EEG_beta_alpha_ratio",
    "EEG_theta_beta_ratio",
    "EEG_spectral_entropy",
]
FIXED_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
THRESHOLD_GRID = np.arange(0.001, 1.0, 0.001)


def build_model(scale_pos_weight: float, n_estimators: int = 500) -> XGBClassifier:
    return XGBClassifier(
        max_depth=3,
        n_estimators=n_estimators,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def grouped_splitter(n_splits: int, logger: logging.Logger, purpose: str):
    if StratifiedGroupKFold is not None:
        logger.info("Using StratifiedGroupKFold for %s", purpose)
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    logger.warning("StratifiedGroupKFold unavailable; using GroupKFold for %s", purpose)
    print(f"WARNING: StratifiedGroupKFold unavailable; using GroupKFold for {purpose}.")
    return GroupKFold(n_splits=n_splits)


def split_indices(splitter, X: pd.DataFrame, y: pd.Series, groups: pd.Series, purpose: str):
    try:
        return list(splitter.split(X, y, groups))
    except ValueError as error:
        if isinstance(splitter, GroupKFold):
            raise
        print(f"WARNING: StratifiedGroupKFold failed for {purpose}; using GroupKFold fallback: {error}")
        return list(GroupKFold(n_splits=splitter.n_splits).split(X, y, groups))


def validate_group_separation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_number: int,
    split_name: str,
) -> None:
    overlap = set(train_df[PATIENT_COL]) & set(test_df[PATIENT_COL])
    if overlap:
        raise AssertionError(f"Fold {fold_number} {split_name} patient leakage: {sorted(overlap)}")


def fit_with_early_stopping(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
            early_stopping_rounds=30,
        )
    except TypeError:
        model.set_params(early_stopping_rounds=30)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float, float]:
    candidates = []
    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)
        candidates.append(
            (
                float(f1_score(y_true, predictions, zero_division=0)),
                float(recall_score(y_true, predictions, zero_division=0)),
                float(threshold),
            )
        )
    best_f1, best_recall, best_threshold = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return best_threshold, best_f1, best_recall


def classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float | int]:
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(true_negatives),
        "fp": int(false_positives),
        "fn": int(false_negatives),
        "tp": int(true_positives),
        "predicted_positive": int(predictions.sum()),
    }


def safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray, logger: logging.Logger, fold_number: int) -> float:
    if y_true.nunique() < 2:
        message = f"Fold {fold_number}: ROC-AUC is NaN because the test fold contains one class"
        print(f"WARNING: {message}")
        logger.warning(message)
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def gain_importance(model: XGBClassifier) -> pd.DataFrame:
    booster_scores = model.get_booster().get_score(importance_type="gain")
    rows = []
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        rows.append({
            "feature": feature_name,
            "gain": float(booster_scores.get(f"f{index}", 0.0)),
        })
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    df = pd.read_csv(DATA_PATH)
    required_columns = [*GROUP_COLUMNS, *FEATURE_COLUMNS, TARGET_COL]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns were not found: {missing_columns}")
    if len(FEATURE_COLUMNS) != 23 or len(set(FEATURE_COLUMNS)) != 23:
        raise AssertionError("Exactly 23 unique model features are required")
    if df[PATIENT_COL].nunique() != 10 or set(df[PATIENT_COL]) != {f"sub-{index:03d}" for index in range(1, 11)}:
        raise ValueError("All ten expected patients sub-001 through sub-010 are required")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain NaN values")
    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite values")
    if df.duplicated(GROUP_COLUMNS).any():
        raise ValueError("Duplicate patient_id + run_id + window_start_time rows found")
    if not df[TARGET_COL].isin([0, 1]).all():
        raise ValueError("Label must contain only binary values 0 and 1")
    if not (df[TARGET_COL] == 1).any():
        raise ValueError("At least one positive label is required")

    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"Baseline positive rate: {df[TARGET_COL].mean():.8f}")
    print("Overall label counts:")
    print(df[TARGET_COL].value_counts().sort_index().to_string())
    print("Label counts per patient:")
    print(df.groupby(PATIENT_COL)[TARGET_COL].value_counts().unstack(fill_value=0).to_string())
    logger.info("Feature columns: %s", FEATURE_COLUMNS)
    logger.info("Baseline positive rate: %.8f", df[TARGET_COL].mean())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL].astype(int)
    groups = df[PATIENT_COL]
    outer_splits = min(5, groups.nunique())
    outer_splitter = grouped_splitter(outer_splits, logger, "outer validation")
    outer_folds = split_indices(outer_splitter, X, y, groups, "outer validation")
    fold_metrics = []
    prediction_frames = []
    best_iterations = []

    for fold_number, (development_indices, test_indices) in enumerate(outer_folds, start=1):
        development_df = df.iloc[development_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        validate_group_separation(development_df, test_df, fold_number, "development/test")

        inner_splitter = grouped_splitter(2, logger, f"fold {fold_number} validation")
        inner_folds = split_indices(
            inner_splitter,
            development_df[FEATURE_COLUMNS],
            development_df[TARGET_COL].astype(int),
            development_df[PATIENT_COL],
            f"fold {fold_number} validation",
        )
        train_indices, validation_indices = inner_folds[0]
        train_df = development_df.iloc[train_indices].reset_index(drop=True)
        validation_df = development_df.iloc[validation_indices].reset_index(drop=True)
        validate_group_separation(train_df, validation_df, fold_number, "training/validation")

        y_train = train_df[TARGET_COL].astype(int)
        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        if positive_count == 0:
            raise ValueError(f"Fold {fold_number} training split has zero positive samples")
        scale_pos_weight = negative_count / positive_count

        model = build_model(scale_pos_weight)
        fit_with_early_stopping(
            model,
            train_df[FEATURE_COLUMNS],
            y_train,
            validation_df[FEATURE_COLUMNS],
            validation_df[TARGET_COL].astype(int),
        )
        validation_probabilities = model.predict_proba(validation_df[FEATURE_COLUMNS])[:, 1]
        selected_threshold, validation_f1, validation_recall = select_threshold(
            validation_df[TARGET_COL].astype(int),
            validation_probabilities,
        )
        test_probabilities = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        y_test = test_df[TARGET_COL].astype(int)
        selected_predictions = (test_probabilities >= selected_threshold).astype(int)
        selected_metrics = classification_metrics(y_test, selected_predictions)
        test_positives = int((y_test == 1).sum())
        test_negatives = int((y_test == 0).sum())
        roc_auc = safe_roc_auc(y_test, test_probabilities, logger, fold_number)
        pr_auc = float(average_precision_score(y_test, test_probabilities))
        fold_metrics.append({
            **selected_metrics,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "test_positive_count": test_positives,
            "test_negative_count": test_negatives,
            "threshold": selected_threshold,
        })
        best_iterations.append(int(getattr(model, "best_iteration", 499)) + 1)

        print(
            f"Fold {fold_number}: selected threshold={selected_threshold:.3f}, "
            f"validation F1={validation_f1:.4f}, validation recall={validation_recall:.4f}, "
            f"test precision={selected_metrics['precision']:.4f}, "
            f"recall={selected_metrics['recall']:.4f}, F1={selected_metrics['f1']:.4f}, "
            f"ROC-AUC={roc_auc}, PR-AUC={pr_auc:.4f}, "
            f"confusion=(TN={selected_metrics['tn']}, FP={selected_metrics['fp']}, "
            f"FN={selected_metrics['fn']}, TP={selected_metrics['tp']}), "
            f"test positives={test_positives}, negatives={test_negatives}"
        )
        logger.info("Fold %d selected metrics: %s", fold_number, fold_metrics[-1])

        print("Fixed thresholds:")
        for threshold in FIXED_THRESHOLDS:
            fixed_predictions = (test_probabilities >= threshold).astype(int)
            fixed_metrics = classification_metrics(y_test, fixed_predictions)
            print(
                f"  threshold={threshold:.2f}: precision={fixed_metrics['precision']:.4f}, "
                f"recall={fixed_metrics['recall']:.4f}, F1={fixed_metrics['f1']:.4f}, "
                f"predicted_positive={fixed_metrics['predicted_positive']}"
            )
            logger.info("Fold %d fixed threshold %.2f: %s", fold_number, threshold, fixed_metrics)

        prediction_frame = test_df[[PATIENT_COL, "run_id", "window_start_time"]].copy()
        prediction_frame["y_true"] = y_test.to_numpy()
        prediction_frame["probability"] = test_probabilities
        prediction_frame["prediction"] = selected_predictions
        prediction_frame["threshold"] = selected_threshold
        prediction_frame["fold"] = fold_number
        prediction_frames.append(prediction_frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != len(df):
        raise AssertionError("OOF prediction row count does not equal dataset row count")
    if predictions.duplicated(GROUP_COLUMNS).any():
        raise AssertionError("OOF predictions contain duplicate patient/run/window keys")
    if set(map(tuple, predictions[GROUP_COLUMNS].to_numpy())) != set(map(tuple, df[GROUP_COLUMNS].to_numpy())):
        raise AssertionError("OOF predictions do not cover every original dataset row exactly once")

    metrics_frame = pd.DataFrame(fold_metrics)
    print(f"Mean F1: {metrics_frame['f1'].mean():.4f}")
    print(f"Mean Precision: {metrics_frame['precision'].mean():.4f}")
    print(f"Mean Recall: {metrics_frame['recall'].mean():.4f}")
    print(f"Mean ROC-AUC: {metrics_frame['roc_auc'].mean():.4f}")
    print(f"Mean PR-AUC: {metrics_frame['pr_auc'].mean():.4f}")
    print(f"Total test positive samples evaluated: {int(metrics_frame['test_positive_count'].sum())}")

    final_estimators = max(1, int(np.median(best_iterations)))
    overall_positive_count = int((y == 1).sum())
    overall_negative_count = int((y == 0).sum())
    final_model = build_model(overall_negative_count / overall_positive_count, final_estimators)
    final_model.fit(X, y, verbose=False)
    joblib.dump(
        {"model": final_model, "feature_columns": FEATURE_COLUMNS},
        MODEL_PATH,
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    importance = gain_importance(final_model)
    gain_total = importance["gain"].sum()
    importance["gain_normalized"] = (
        importance["gain"] / gain_total if gain_total > 0 else 0.0
    )
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print("Feature gain importance:")
    print(importance.to_string(index=False))
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
