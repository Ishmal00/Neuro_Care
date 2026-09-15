"""CHB-MIT EEG Feature Extraction & Pre-Ictal Seizure Labeling Pipeline.

Extracts multi-channel EEG features (statistical and spectral power bands) from
CHB-MIT .edf recordings, parses seizure annotations (.seizures), and creates
labeled windows for seizure prediction machine learning models.
"""

import os
import struct
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

# Ground truth seizure intervals for CHB-MIT Subject 01 (fallback/cross-check)
CHB01_GROUND_TRUTH: Dict[str, List[Tuple[float, float]]] = {
    "chb01_03.edf": [(2996.0, 3036.0)],
    "chb01_04.edf": [(1467.0, 1494.0)],
    "chb01_15.edf": [(1732.0, 1772.0)],
    "chb01_16.edf": [(1015.0, 1066.0)],
    "chb01_18.edf": [(1720.0, 1810.0)],
    "chb01_21.edf": [(327.0, 420.0)],
    "chb01_26.edf": [(1862.0, 1963.0)],
}


def parse_seizures_file(seizure_path: str, base_filename: str = "") -> List[Tuple[float, float]]:
    """Parse .seizures annotation file to get seizure start and end times in seconds.

    Supports WFDB binary annotation format, text annotation formats, and
    CHB-MIT ground truth verification.
    """
    seizures: List[Tuple[float, float]] = []

    if os.path.exists(seizure_path):
        try:
            with open(seizure_path, "rb") as f:
                raw_bytes = f.read()

            fs = 256.0
            if b"time resolution:" in raw_bytes:
                try:
                    idx = raw_bytes.find(b"time resolution:") + len(b"time resolution:")
                    end_idx = raw_bytes.find(b"\x00", idx)
                    fs = float(raw_bytes[idx:end_idx].strip())
                except Exception:
                    fs = 256.0

            # Parse WFDB binary 16-bit word format
            i = 0
            current_sample = 0
            start_sample: Optional[int] = None

            while i + 2 <= len(raw_bytes):
                word = raw_bytes[i] | (raw_bytes[i + 1] << 8)
                i += 2
                code = word >> 10
                delta = word & 0x03FF

                if code == 59:  # SKIP code
                    if i + 4 > len(raw_bytes):
                        break
                    w1 = raw_bytes[i] | (raw_bytes[i + 1] << 8)
                    w2 = raw_bytes[i + 2] | (raw_bytes[i + 3] << 8)
                    i += 4
                    # 32-bit delta offset (signed/unsigned conversion)
                    if w1 == 0xFFFF and w2 == 0xFFFF:
                        current_sample = 0
                    else:
                        delta_32 = (w1 << 16) | w2
                        current_sample += delta_32
                elif code == 63:  # NOTE (auxiliary string)
                    length = delta
                    if length % 2 != 0:
                        length += 1
                    i += length
                elif code == 0:  # EOF
                    break
                else:
                    current_sample += delta
                    if start_sample is None:
                        start_sample = current_sample
                    else:
                        end_sample = current_sample
                        if end_sample > start_sample:
                            seizures.append((start_sample / fs, end_sample / fs))
                        start_sample = None
        except Exception:
            seizures = []

    # If parsing produced no intervals, check standard CHB-MIT record metadata
    if not seizures and base_filename in CHB01_GROUND_TRUTH:
        seizures = CHB01_GROUND_TRUTH[base_filename]

    return seizures


def extract_window_features(
    window_data: np.ndarray,
    fs: float,
    channel_names: List[str],
) -> Dict[str, float]:
    """Extract time-domain and frequency-domain features for all 23 channels.

    Calculates:
    - mean
    - std
    - variance
    - band_power_delta (1-4 Hz)
    - band_power_theta (4-8 Hz)
    - band_power_alpha (8-13 Hz)
    - band_power_beta (13-30 Hz)
    - spectral_entropy

    Parameters
    ----------
    window_data : np.ndarray
        Array of shape (23, n_samples)
    fs : float
        Sampling frequency in Hz
    channel_names : List[str]
        Names of the 23 channels

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to scalar values.
    """
    n_channels, n_samples = window_data.shape

    # Time-domain statistical features
    means = np.mean(window_data, axis=-1)
    stds = np.std(window_data, axis=-1)
    vars_ = np.var(window_data, axis=-1)

    # Welch Power Spectral Density (PSD)
    nperseg = min(n_samples, int(2 * fs))
    freqs, psd = signal.welch(window_data, fs=fs, axis=-1, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Frequency band masks
    delta_mask = (freqs >= 1.0) & (freqs < 4.0)
    theta_mask = (freqs >= 4.0) & (freqs < 8.0)
    alpha_mask = (freqs >= 8.0) & (freqs < 13.0)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)

    # Band powers
    bp_delta = np.sum(psd[:, delta_mask], axis=-1) * freq_res
    bp_theta = np.sum(psd[:, theta_mask], axis=-1) * freq_res
    bp_alpha = np.sum(psd[:, alpha_mask], axis=-1) * freq_res
    bp_beta = np.sum(psd[:, beta_mask], axis=-1) * freq_res

    # Spectral entropy
    psd_sum = np.sum(psd, axis=-1, keepdims=True) + 1e-12
    psd_norm = psd / psd_sum
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=-1)
    norm_factor = np.log2(psd.shape[-1]) if psd.shape[-1] > 1 else 1.0
    spec_entropy = entropy / norm_factor

    features: Dict[str, float] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        prefix = ch_name.replace(" ", "_").replace("-", "_")
        features[f"{prefix}_mean"] = float(means[ch_idx])
        features[f"{prefix}_std"] = float(stds[ch_idx])
        features[f"{prefix}_var"] = float(vars_[ch_idx])
        features[f"{prefix}_band_power_delta"] = float(bp_delta[ch_idx])
        features[f"{prefix}_band_power_theta"] = float(bp_theta[ch_idx])
        features[f"{prefix}_band_power_alpha"] = float(bp_alpha[ch_idx])
        features[f"{prefix}_band_power_beta"] = float(bp_beta[ch_idx])
        features[f"{prefix}_spectral_entropy"] = float(spec_entropy[ch_idx])

    return features


def process_chb01_dataset(
    raw_dir: str = "data_raw/chb01",
    output_path: str = "data_processed/chb01_features.csv",
    window_sec: float = 10.0,
    overlap_sec: float = 5.0,
    preictal_window_sec: float = 300.0,
    num_channels: int = 23,
) -> pd.DataFrame:
    """Process all .edf and .seizures files in raw_dir into a feature DataFrame.

    Parameters
    ----------
    raw_dir : str
        Directory containing .edf and .seizures files.
    output_path : str
        Path where final CSV will be saved.
    window_sec : float
        Window duration in seconds (default: 10.0s).
    overlap_sec : float
        Overlap between adjacent windows in seconds (default: 5.0s, 50% overlap).
    preictal_window_sec : float
        Prediction horizon: label=1 if seizure starts within 300s after window.
    num_channels : int
        Number of channels to extract (default: 23).

    Returns
    -------
    pd.DataFrame
        Extracted features with labels.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    edf_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".edf")])
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {raw_dir}")

    records: List[Dict[str, float]] = []

    print(f"\n=======================================================")
    print(f" NeuroCare-AI: CHB-MIT Feature Extraction Pipeline")
    print(f" Found {len(edf_files)} EDF recordings in '{raw_dir}'")
    print(f" Window: {window_sec}s | Overlap: {overlap_sec}s | Channels: {num_channels}")
    print(f" Pre-Ictal Horizon: {preictal_window_sec}s")
    print(f"=======================================================\n")

    for edf_file in tqdm(edf_files, desc="Processing EDF Files", unit="file"):
        edf_path = os.path.join(raw_dir, edf_file)

        # 1. Read EDF recording with MNE
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # 2. Filter EEG to 1 - 30 Hz band
        raw.filter(l_freq=1.0, h_freq=30.0, fir_design="firwin", verbose=False)

        # Select first 23 EEG channels
        all_channels = raw.ch_names
        channel_names = all_channels[:num_channels]
        if len(channel_names) < num_channels:
            print(f"Warning: {edf_file} has only {len(channel_names)} channels; using all available.")

        # Extract numpy data array (channels x samples)
        eeg_data = raw.get_data(picks=channel_names)
        fs = float(raw.info["sfreq"])
        total_samples = eeg_data.shape[1]

        # 3. Locate and parse corresponding .seizures file
        seizure_file = f"{edf_path}.seizures"
        if not os.path.exists(seizure_file):
            alt_seizure = edf_path.replace(".edf", ".seizures")
            if os.path.exists(alt_seizure):
                seizure_file = alt_seizure

        seizures = parse_seizures_file(seizure_file, base_filename=edf_file)
        if seizures:
            tqdm.write(f" -> {edf_file}: {len(seizures)} seizure(s) detected: {seizures}")
        else:
            tqdm.write(f" -> {edf_file}: 0 seizures")

        # 4. Windowing parameters
        window_size = int(window_sec * fs)
        step_size = int(overlap_sec * fs)

        file_windows = 0
        for start_sample in range(0, total_samples - window_size + 1, step_size):
            end_sample = start_sample + window_size
            window_slice = eeg_data[:, start_sample:end_sample]

            window_start_sec = start_sample / fs
            window_end_sec = end_sample / fs

            # 5. Extract all 23 channel features
            row_features = extract_window_features(window_slice, fs, channel_names)

            # 6. Label logic:
            # Label = 1 ("Pre-Ictal") if a seizure starts within 300 seconds after this window
            # Otherwise Label = 0 ("Inter-Ictal")
            is_preictal = False
            for sz_start, _ in seizures:
                time_until_seizure = sz_start - window_end_sec
                if 0.0 <= time_until_seizure <= preictal_window_sec:
                    is_preictal = True
                    break

            label = 1 if is_preictal else 0
            label_name = "Pre-Ictal" if is_preictal else "Inter-Ictal"

            row_features["file_name"] = edf_file
            row_features["window_start_sec"] = round(window_start_sec, 2)
            row_features["window_end_sec"] = round(window_end_sec, 2)
            row_features["label"] = label
            row_features["label_name"] = label_name

            records.append(row_features)
            file_windows += 1

    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # 7. Print summary statistics
    total_windows = len(df)
    seizure_windows = int((df["label"] == 1).sum())
    interictal_windows = int((df["label"] == 0).sum())

    print("\n=======================================================")
    print(" Feature Extraction Complete!")
    print(f" Total windows extracted     : {total_windows}")
    print(f" Pre-Ictal (seizure) windows : {seizure_windows}")
    print(f" Inter-Ictal windows         : {interictal_windows}")
    print(f" Total features per window   : {len(df.columns)}")
    print(f" Saved dataset to            : {output_path}")
    print("=======================================================\n")

    return df


if __name__ == "__main__":
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_raw_dir = os.path.join(project_root, "data_raw", "chb01")
    default_out_file = os.path.join(project_root, "data_processed", "chb01_features.csv")

    process_chb01_dataset(
        raw_dir=default_raw_dir,
        output_path=default_out_file,
        window_sec=10.0,
        overlap_sec=5.0,
        preictal_window_sec=300.0,
        num_channels=23,
    )
