import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_PATH = "data/seizeit2_features.csv"
MODEL_DIR = "models"
LOG_PATH = "logs/train_xgb.log"
TARGET_COL = "Label"
PATIENT_COL = "patient_id"
TIME_COL = "window_start_time"
RANDOM_STATE = 42
N_SPLITS = 3


def build_model():
    return XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.03,
        scale_pos_weight=30,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def engineer_features(df):
    """Add causal, patient-local features after chronological sorting."""
    df = df.copy()
    patient_groups = df.groupby(PATIENT_COL, sort=False)

    df["EEG_ratio"] = (
        df["EEG_delta"] + df["EEG_theta"]
    ) / (
        df["EEG_alpha"] + df["EEG_beta"] + 1e-6
    )
    df["ACC_jerk"] = patient_groups["ACC_energy"].diff().abs()
    df["EMG_zero_cross"] = patient_groups["EMG_mean"].transform(
        lambda values: values.diff().gt(0).astype(int).rolling(
            10,
            min_periods=1,
        ).sum()
    )
    df["HR_variability"] = patient_groups["HR_mean"].transform(
        lambda values: values.rolling(
            20,
            min_periods=2,
        ).std()
    )

    return df.fillna(0)


def duplicate_sub003_positives(X_train, y_train, patient_id):
    """Add ten copies of sub-003 positives before applying SMOTE."""
    if patient_id != "sub-003" or int(y_train.sum()) >= 10:
        return X_train, y_train

    positive_rows = X_train.loc[y_train == 1]
    positive_labels = y_train.loc[y_train == 1]

    if positive_rows.empty:
        return X_train, y_train

    X_train = pd.concat(
        [X_train] + [positive_rows.copy() for _ in range(10)],
        ignore_index=True,
    )
    y_train = pd.concat(
        [y_train] + [positive_labels.copy() for _ in range(10)],
        ignore_index=True,
    )
    return X_train, y_train


def apply_smote(X_train, y_train, patient_id):
    X_train, y_train = duplicate_sub003_positives(
        X_train,
        y_train,
        patient_id,
    )

    if y_train.nunique() < 2:
        return X_train, y_train

    minority_count = int(y_train.value_counts().min())

    if minority_count < 2:
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy=0.1,
        k_neighbors=min(3, minority_count - 1),
        random_state=RANDOM_STATE,
    )

    return smote.fit_resample(X_train, y_train)


def main():
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    df = pd.read_csv(DATA_PATH)

    for required_column in (TARGET_COL, PATIENT_COL, TIME_COL):
        if required_column not in df.columns:
            raise KeyError(f"Required column '{required_column}' was not found")

    df = df.sort_values(
        [PATIENT_COL, TIME_COL]
    ).reset_index(drop=True)
    df = engineer_features(df)

    feature_columns = [
        column
        for column in df.columns
        if column not in {TARGET_COL, PATIENT_COL, TIME_COL}
    ]

    thresholds = np.arange(0.001, 0.501, 0.001)

    for patient_id, patient_df in df.groupby(
        PATIENT_COL,
        sort=True,
    ):
        patient_df = patient_df.reset_index(drop=True)

        X_patient = patient_df[feature_columns]
        y_patient = patient_df[TARGET_COL].astype(int)

        validation_probabilities = []
        validation_labels = []
        split_metrics = []

        splitter = TimeSeriesSplit(n_splits=N_SPLITS)

        for split_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            splitter.split(X_patient),
            start=1,
        ):
            X_train = X_patient.iloc[train_indices]
            y_train = y_patient.iloc[train_indices]

            X_validation = X_patient.iloc[validation_indices]
            y_validation = y_patient.iloc[validation_indices]

            X_resampled, y_resampled = apply_smote(
                X_train,
                y_train,
                patient_id,
            )

            if y_resampled.nunique() < 2:
                logger.warning(
                    "Skipping patient %s split %d because training "
                    "contains only one class",
                    patient_id,
                    split_number,
                )
                continue

            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_resampled,
            )

            model = build_model()

            model.fit(
                X_resampled,
                y_resampled,
                sample_weight=sample_weights,
                verbose=False,
            )

            probabilities = model.predict_proba(
                X_validation
            )[:, 1]

            validation_probabilities.extend(probabilities)
            validation_labels.extend(
                y_validation.to_numpy()
            )

            split_f1 = max(
                f1_score(
                    y_validation,
                    probabilities >= threshold,
                    zero_division=0,
                )
                for threshold in thresholds
            )

            split_metrics.append(split_f1)

        if not validation_labels:
            raise ValueError(
                f"Patient {patient_id} has no valid "
                "time-series validation split"
            )

        validation_labels = np.asarray(validation_labels)
        validation_probabilities = np.asarray(
            validation_probabilities
        )

        patient_f1_scores = [
            f1_score(
                validation_labels,
                validation_probabilities >= threshold,
                zero_division=0,
            )
            for threshold in thresholds
        ]

        best_threshold = float(
            thresholds[int(np.argmax(patient_f1_scores))]
        )

        validation_predictions = (
            validation_probabilities >= best_threshold
        ).astype(int)

        patient_f1 = f1_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )

        patient_precision = precision_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )

        patient_recall = recall_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )

        print(
            f"Patient {patient_id}: "
            f"F1={patient_f1:.4f}, "
            f"Precision={patient_precision:.4f}, "
            f"Recall={patient_recall:.4f}, "
            f"Threshold={best_threshold:.3f}"
        )

        logger.info(
            "Patient %s: F1=%.4f, precision=%.4f, "
            "recall=%.4f, threshold=%.3f, split_f1=%s",
            patient_id,
            patient_f1,
            patient_precision,
            patient_recall,
            best_threshold,
            split_metrics,
        )

        X_final, y_final = apply_smote(
            X_patient,
            y_patient,
            patient_id,
        )

        final_model = build_model()

        final_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_final,
        )

        final_model.fit(
            X_final,
            y_final,
            sample_weight=final_weights,
            verbose=False,
        )

        model_path = (
            Path(MODEL_DIR)
            / f"xgb_seizeit2_{patient_id}.pkl"
        )

        joblib.dump(
            {
                "model": final_model,
                "threshold": best_threshold,
                "patient_id": patient_id,
                "feature_columns": feature_columns,
            },
            model_path,
        )

        print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()