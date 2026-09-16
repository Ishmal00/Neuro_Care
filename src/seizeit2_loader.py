"""Extract aligned multimodal SeizeIT2 features for the first ten patients."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy import signal


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "ds005873"
OUTPUT_PATH = ROOT_DIR / "data" / "seizeit2_features.csv"
LABELS_PATH = ROOT_DIR / "data" / "seizeit2_labels.csv"
SUMMARY_PATH = ROOT_DIR / "data" / "seizeit2_processing_log.txt"

PATIENTS = [f"sub-{number:03d}" for number in range(1, 11)]
WINDOW_SECONDS = 60.0
STEP_SECONDS = 30.0

EEG_FEATURES = [
    "EEG_delta",
    "EEG_theta",
    "EEG_alpha",
    "EEG_beta",
    "EEG_gamma",
]
ECG_FEATURES = ["ECG_HR_mean", "ECG_RMSSD", "ECG_SDNN"]
EMG_FEATURES = ["EMG_mean", "EMG_std"]
MOV_FEATURES = ["ACC_mean", "ACC_std", "ACC_energy"]
EEG_DERIVED_FEATURES = [
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
FEATURE_COLUMNS = EEG_FEATURES + ECG_FEATURES + EMG_FEATURES + MOV_FEATURES + EEG_DERIVED_FEATURES

EEG_BANDS = {
    "EEG_delta": (1.0, 4.0),
    "EEG_theta": (4.0, 8.0),
    "EEG_alpha": (8.0, 13.0),
    "EEG_beta": (13.0, 30.0),
    "EEG_gamma": (30.0, 45.0),
}


def run_number(path: Path) -> int:
    match = re.search(r"_run-(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not determine run number from {path.name}")
    return int(match.group(1))


def find_recordings(patient_dir: Path, modality: str) -> dict[int, Path]:
    return {
        run_number(path): path
        for path in (patient_dir / "ses-01" / modality).glob("*.edf")
    }


def find_channel(raw: mne.io.BaseRaw, preferred: Iterable[str]) -> str:
    for name in preferred:
        if name in raw.ch_names:
            return name
    raise ValueError(f"None of the expected channels exist: {list(preferred)}")


def window_count(duration: float) -> int:
    if duration < WINDOW_SECONDS:
        return 0
    return int(np.floor((duration - WINDOW_SECONDS) / STEP_SECONDS) + 1)


def band_features(data: np.ndarray, sampling_rate: float) -> dict[str, float]:
    data = np.asarray(data, dtype=float)
    nperseg = min(data.shape[-1], max(8, int(2 * sampling_rate)))
    frequencies, power = signal.welch(data, fs=sampling_rate, nperseg=nperseg)
    frequency_step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    band_powers = {}
    for name, (low, high) in EEG_BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        band_powers[name] = float(np.mean(np.sum(power[..., mask], axis=-1) * frequency_step))
    epsilon = np.finfo(float).eps
    total_power = sum(band_powers.values())
    relative_powers = {
        name: value / (total_power + epsilon)
        for name, value in band_powers.items()
    }
    mean_power_spectrum = np.mean(power, axis=0)
    probability = mean_power_spectrum / (np.sum(mean_power_spectrum) + epsilon)
    spectral_entropy = float(-np.sum(probability * np.log(probability + epsilon)))
    features = {
        **band_powers,
        "EEG_total_power": total_power,
        **{
            f"{name}_relative": relative_powers[name]
            for name in EEG_FEATURES
        },
        "EEG_theta_alpha_ratio": band_powers["EEG_theta"] / (band_powers["EEG_alpha"] + epsilon),
        "EEG_beta_alpha_ratio": band_powers["EEG_beta"] / (band_powers["EEG_alpha"] + epsilon),
        "EEG_theta_beta_ratio": band_powers["EEG_theta"] / (band_powers["EEG_beta"] + epsilon),
        "EEG_spectral_entropy": spectral_entropy,
    }
    return features


def eeg_features(raw: mne.io.BaseRaw, start: float, end: float) -> dict[str, float]:
    channel_names = [
        name for name in ("BTEleft SD", "BTEright SD", "CROSStop SD")
        if name in raw.ch_names
    ]
    if not channel_names:
        raise ValueError("No configured SeizeIT2 EEG channels found")
    data = raw.get_data(picks=channel_names, start=int(start * raw.info["sfreq"]), stop=int(end * raw.info["sfreq"]))
    return band_features(data, float(raw.info["sfreq"]))


def ecg_features(raw: mne.io.BaseRaw, start: float, end: float) -> dict[str, float]:
    channel = find_channel(raw, ("ECG SD",))
    data = raw.get_data(picks=[channel], start=int(start * raw.info["sfreq"]), stop=int(end * raw.info["sfreq"]))[0]
    sampling_rate = float(raw.info["sfreq"])
    centered = data - np.median(data)
    prominence = max(float(np.std(centered)) * 0.25, 1e-12)
    peaks, _ = signal.find_peaks(centered, distance=max(1, int(0.30 * sampling_rate)), prominence=prominence)
    intervals = np.diff(peaks) / sampling_rate
    valid_intervals = intervals[(intervals >= 0.30) & (intervals <= 2.0)]
    if not len(valid_intervals):
        return {name: 0.0 for name in ECG_FEATURES}
    heart_rates = 60.0 / valid_intervals
    return {
        "ECG_HR_mean": float(np.mean(heart_rates)),
        "ECG_RMSSD": float(np.sqrt(np.mean(np.diff(valid_intervals) ** 2))) if len(valid_intervals) > 1 else 0.0,
        "ECG_SDNN": float(np.std(valid_intervals, ddof=1)) if len(valid_intervals) > 1 else 0.0,
    }


def emg_features(raw: mne.io.BaseRaw, start: float, end: float) -> dict[str, float]:
    channel = find_channel(raw, ("EMG SD",))
    data = raw.get_data(picks=[channel], start=int(start * raw.info["sfreq"]), stop=int(end * raw.info["sfreq"]))[0]
    return {"EMG_mean": float(np.mean(data)), "EMG_std": float(np.std(data))}


def mov_features(raw: mne.io.BaseRaw, start: float, end: float) -> dict[str, float]:
    channels = [name for name in ("EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z") if name in raw.ch_names]
    if len(channels) != 3:
        raise ValueError("All three SeizeIT2 movement axes are required")
    data = raw.get_data(picks=channels, start=int(start * raw.info["sfreq"]), stop=int(end * raw.info["sfreq"]))
    magnitude = np.sqrt(np.sum(data**2, axis=0))
    return {
        "ACC_mean": float(np.mean(magnitude)),
        "ACC_std": float(np.std(magnitude)),
        "ACC_energy": float(np.mean(magnitude**2)),
    }


def seizure_intervals(events_path: Path) -> list[tuple[float, float]]:
    if not events_path.exists():
        return []
    events = pd.read_csv(events_path, sep="\t")
    if not {"onset", "duration", "eventType"}.issubset(events.columns):
        return []
    seizure_rows = events[events["eventType"].astype(str).str.startswith("sz")]
    return [(float(row.onset), float(row.onset + row.duration)) for row in seizure_rows.itertuples()]


def overlaps_seizure(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start < seizure_end and end > seizure_start for seizure_start, seizure_end in intervals)


def process_run(patient_id: str, run: int, paths: dict[str, Path], missing: Counter) -> list[dict[str, object]]:
    raws = {}
    try:
        for modality, path in paths.items():
            try:
                raws[modality] = mne.io.read_raw_edf(path, preload=False, verbose=False)
            except Exception as error:
                missing["unreadable_recording"] += 1
                print(
                    f"Skipping {patient_id} run {run}: {modality} EDF {path} could not be opened; "
                    f"{type(error).__name__}: {error}"
                )
                return []
        durations = {modality: float(raw.times[-1]) for modality, raw in raws.items()}
        duration = min(durations.values())
        events_path = paths["eeg"].with_name(paths["eeg"].name.replace("_eeg.edf", "_events.tsv"))
        intervals = seizure_intervals(events_path)
        records = []
        for index in range(window_count(duration)):
            start = index * STEP_SECONDS
            end = start + WINDOW_SECONDS
            row: dict[str, object] = {
                "patient_id": patient_id,
                "run_id": run,
                "window_start_time": start,
            }
            try:
                row.update(eeg_features(raws["eeg"], start, end))
                row.update(ecg_features(raws["ecg"], start, end))
                row.update(emg_features(raws["emg"], start, end))
                row.update(mov_features(raws["mov"], start, end))
            except (ValueError, IndexError, RuntimeError):
                missing["incomplete_window"] += 1
                continue
            row["Label"] = int(overlaps_seizure(start, end, intervals))
            records.append(row)
        return records
    finally:
        for raw in raws.values():
            raw.close()


def validate_dataset(df: pd.DataFrame) -> None:
    expected_columns = ["patient_id", "run_id", "window_start_time", *FEATURE_COLUMNS, "Label"]
    if list(df.columns) != expected_columns:
        raise AssertionError("Unexpected output columns or feature ordering")
    if sorted(df["patient_id"].unique()) != PATIENTS:
        raise AssertionError("Output does not contain exactly the ten requested patients")
    if df.duplicated(["patient_id", "run_id", "window_start_time"]).any():
        raise AssertionError("Duplicate patient/run/window rows found")
    if not set(df["Label"].unique()).issubset({0, 1}):
        raise AssertionError("Labels are not binary")
    feature_frame = df[FEATURE_COLUMNS]
    if feature_frame.isna().any().any() or not np.isfinite(feature_frame.to_numpy()).all():
        raise AssertionError("NaN or infinite feature values found")
    if df[["patient_id", "run_id", "window_start_time"]].duplicated().any():
        raise AssertionError("Duplicate alignment keys found")
    if not df.equals(df.sort_values(["patient_id", "run_id", "window_start_time"]).reset_index(drop=True)):
        raise AssertionError("Output is not sorted")


def write_summary(df: pd.DataFrame, missing: Counter, patient_counts: dict[str, int]) -> None:
    lines = [
        "SeizeIT2 multimodal feature extraction summary",
        f"Patients processed: {', '.join(PATIENTS)}",
        f"Window length seconds: {WINDOW_SECONDS}",
        f"Window step seconds: {STEP_SECONDS}",
        "Windows per patient:",
    ]
    for patient_id in PATIENTS:
        patient_df = df[df["patient_id"] == patient_id]
        lines.append(
            f"  {patient_id}: {patient_counts[patient_id]} total, "
            f"{int(patient_df['Label'].sum())} positive, "
            f"{int((patient_df['Label'] == 0).sum())} negative"
        )
    lines.extend([
        f"Total rows: {len(df)}",
        f"Feature count: {len(FEATURE_COLUMNS)}",
        f"EEG feature count: {len(EEG_FEATURES)}",
        f"ECG feature count: {len(ECG_FEATURES)}",
        f"EMG feature count: {len(EMG_FEATURES)}",
        f"MOV feature count: {len(MOV_FEATURES)}",
        "Missing modality/window counts:",
        f"  missing_recording: {missing['missing_recording']}",
        f"  unreadable_recording: {missing['unreadable_recording']}",
        f"  incomplete_window: {missing['incomplete_window']}",
        f"Class distribution: 0={int((df['Label'] == 0).sum())}, 1={int((df['Label'] == 1).sum())}",
        "Alignment: each row uses the same patient, run, start, and 60-second interval across EEG, ECG, EMG, and MOV.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, object]] = []
    missing: Counter = Counter()
    patient_counts = {}

    for patient_id in PATIENTS:
        patient_dir = DATASET_DIR / patient_id
        recordings = {modality: find_recordings(patient_dir, modality) for modality in ("eeg", "ecg", "emg", "mov")}
        common_runs = set.intersection(*(set(paths) for paths in recordings.values()))
        all_runs = set.union(*(set(paths) for paths in recordings.values()))
        missing["missing_recording"] += sum(run not in common_runs for run in all_runs)
        patient_records = []
        for run in sorted(common_runs):
            patient_records.extend(process_run(patient_id, run, {modality: recordings[modality][run] for modality in recordings}, missing))
        all_records.extend(patient_records)
        patient_counts[patient_id] = len(patient_records)

    df = pd.DataFrame(all_records, columns=["patient_id", "run_id", "window_start_time", *FEATURE_COLUMNS, "Label"])
    df = df.sort_values(["patient_id", "run_id", "window_start_time"]).reset_index(drop=True)
    validate_dataset(df)
    df.to_csv(OUTPUT_PATH, index=False)
    df[["patient_id", "run_id", "window_start_time", "Label"]].to_csv(LABELS_PATH, index=False)
    write_summary(df, missing, patient_counts)

    print("DATASET READY")
    print(f"Shape: {df.shape}")
    print(f"Feature names by modality: EEG={EEG_FEATURES}; ECG={ECG_FEATURES}; EMG={EMG_FEATURES}; MOV={MOV_FEATURES}")
    print(f"Missing counts: {dict(missing)}")
    return df


if __name__ == "__main__":
    main()
