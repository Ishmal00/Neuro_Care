import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_PATH = "data/seizeit2_features.csv"
MODEL_PATH = "models/xgb_seizeit2.pkl"
PREDICTIONS_PATH = "data/xgb_predictions.csv"
LOG_PATH = "logs/train_xgb.log"

TARGET_COL = "Label"
PATIENT_COL = "patient_id"
TIME_COL = "window_start_time"
RANDOM_STATE = 42


def main():
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    df = pd.read_csv(DATA_PATH)

    for required_column in (TARGET_COL, PATIENT_COL):
        if required_column not in df.columns:
            raise KeyError(f"Required column '{required_column}' was not found")

    feature_columns = [
        column
        for column in df.columns
        if column not in {TARGET_COL, PATIENT_COL, TIME_COL}
    ]

    X = df[feature_columns]
    y = df[TARGET_COL].astype(int)
    groups = df[PATIENT_COL]

    thresholds = np.arange(0.001, 0.2001, 0.001)
    fold_metrics = []
    last_model = None
    last_predictions = None

    outer_splitter = GroupKFold(n_splits=3)

    for fold_number, (development_indices, test_indices) in enumerate(
        outer_splitter.split(X, y, groups=groups),
        start=1,
    ):
        development_df = df.iloc[development_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        inner_splitter = GroupKFold(n_splits=2)

        train_indices, validation_indices = next(
            inner_splitter.split(
                development_df,
                development_df[TARGET_COL],
                groups=development_df[PATIENT_COL],
            )
        )

        train_df = development_df.iloc[train_indices]
        validation_df = development_df.iloc[validation_indices]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COL].astype(int)

        X_validation = validation_df[feature_columns]
        y_validation = validation_df[TARGET_COL].astype(int)

        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COL].astype(int)

        smote = SMOTE(
            sampling_strategy=0.1,
            k_neighbors=3,
            random_state=RANDOM_STATE,
        )

        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train,
            y_train,
        )

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train_resampled,
        )

        model = XGBClassifier(
            max_depth=3,
            n_estimators=200,
            learning_rate=0.05,
            scale_pos_weight=50,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train_resampled,
            y_train_resampled,
            sample_weight=sample_weights,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )

        validation_probabilities = model.predict_proba(X_validation)[:, 1]

        validation_f1 = [
            f1_score(
                y_validation,
                validation_probabilities >= threshold,
                zero_division=0,
            )
            for threshold in thresholds
        ]

        best_threshold = float(
            thresholds[int(np.argmax(validation_f1))]
        )

        test_probabilities = model.predict_proba(X_test)[:, 1]

        test_predictions = (
            test_probabilities >= best_threshold
        ).astype(int)

        metrics = {
            "f1": f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
            "precision": precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
            "auc": roc_auc_score(
                y_test,
                test_probabilities,
            ),
        }

        fold_metrics.append(metrics)

        test_patient = test_df[PATIENT_COL].iloc[0]

        print(
            f"Fold {fold_number} test patient: {test_patient}, "
            f"threshold: {best_threshold:.3f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"AUC: {metrics['auc']:.4f}"
        )

        logger.info(
            "Fold %d, test patient=%s, threshold=%.3f, metrics=%s",
            fold_number,
            test_patient,
            best_threshold,
            metrics,
        )

        last_model = model

        last_predictions = test_df[[PATIENT_COL]].copy()
        last_predictions["y_true"] = y_test.to_numpy()
        last_predictions["probability"] = test_probabilities
        last_predictions["prediction"] = test_predictions
        last_predictions["best_threshold"] = best_threshold

    metrics_frame = pd.DataFrame(fold_metrics)

    print(f"Mean F1: {metrics_frame['f1'].mean():.4f}")
    print(f"Mean Precision: {metrics_frame['precision'].mean():.4f}")
    print(f"Mean Recall: {metrics_frame['recall'].mean():.4f}")
    print(f"Mean AUC: {metrics_frame['auc'].mean():.4f}")

    joblib.dump(last_model, MODEL_PATH)
    last_predictions.to_csv(PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    main()



    