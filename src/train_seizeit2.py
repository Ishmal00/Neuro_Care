from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data_processed" / "seizeit2_features.csv"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "seizeit2_1dcnn.pth"


class Seizure1DCNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        if num_features != 12:
            raise ValueError(f"Expected 12 features, received {num_features}.")

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = LOG_DIR / f"train_{timestamp}.log"

    logger = logging.getLogger("seizeit2_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("Starting seizure prediction training pipeline.")
    logger.info("Log file: %s", log_path)
    return logger


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    class_counts = {int(label): int(count) for label, count in zip(unique, counts)}
    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 0)

    if pos_count == 0 or neg_count == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    weights = np.array([1.0, neg_count / pos_count], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, probs),
    }
    return metrics


def select_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def main() -> None:
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded dataset with shape %s", df.shape)

    feature_columns = [col for col in df.columns if col not in {"patient_id", "window_start_time", "Label"}]
    if len(feature_columns) != 12:
        raise ValueError(f"Expected 12 feature columns; got {len(feature_columns)}: {feature_columns}")

    train_df = df[df["patient_id"].isin(["sub-001", "sub-002"])].copy()
    test_df = df[df["patient_id"] == "sub-003"].copy()

    logger.info("Train rows: %s | Test rows: %s", len(train_df), len(test_df))
    logger.info("Train patient IDs: %s", sorted(train_df["patient_id"].unique().tolist()))
    logger.info("Test patient IDs: %s", sorted(test_df["patient_id"].unique().tolist()))

    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_train_raw = train_df["Label"].astype(np.int64).to_numpy()

    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    y_test_raw = test_df["Label"].astype(np.int64).to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train_raw,
        test_size=0.2,
        stratify=y_train_raw,
        random_state=42,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_split.reshape(-1, 1, X_train_split.shape[1]), dtype=torch.float32),
        torch.tensor(y_train_split, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.reshape(-1, 1, X_val.shape[1]), dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.reshape(-1, 1, X_test.shape[1]), dtype=torch.float32),
        torch.tensor(y_test_raw, dtype=torch.long),
    )

    class_weights = compute_class_weights(y_train_split)
    class_weights = class_weights.to(device)
    logger.info("Class weights: %s", class_weights.cpu().numpy().tolist())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Seizure1DCNN(num_features=len(feature_columns)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, 21):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        train_loss = epoch_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)

                val_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss_avg = val_loss / len(val_dataset)
        val_probs_array = np.concatenate(val_probs)
        val_labels_array = np.concatenate(val_labels)
        val_auc = roc_auc_score(val_labels_array, val_probs_array)

        logger.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f - val_auc=%.4f",
            epoch,
            20,
            train_loss,
            val_loss_avg,
            val_auc,
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is None:
        raise RuntimeError("Model was never saved because the validation loss did not improve.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, MODEL_PATH)
    logger.info("Best model saved to %s", MODEL_PATH)

    model.load_state_dict(best_state_dict)
    model.to(device)

    val_probs_array = np.concatenate(val_probs)
    val_labels_array = np.concatenate(val_labels)
    best_threshold, best_val_f1 = select_threshold(val_probs_array, val_labels_array)
    logger.info("Best validation threshold: %.4f (best validation F1: %.4f)", best_threshold, best_val_f1)

    test_predictions = []
    test_targets = []
    test_probabilities = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= best_threshold).astype(int)
            test_predictions.extend(preds.tolist())
            test_targets.extend(labels.cpu().numpy().tolist())
            test_probabilities.extend(probs.tolist())

    test_targets = np.asarray(test_targets)
    test_predictions = np.asarray(test_predictions)
    test_probabilities = np.asarray(test_probabilities)
    test_metrics = {
        "accuracy": accuracy_score(test_targets, test_predictions),
        "precision": precision_score(test_targets, test_predictions, zero_division=0),
        "recall": recall_score(test_targets, test_predictions, zero_division=0),
        "f1": f1_score(test_targets, test_predictions, zero_division=0),
        "auc": roc_auc_score(test_targets, test_probabilities),
    }

    logger.info("Final Test metrics on sub-003 only using threshold %.4f:", best_threshold)
    for metric_name, value in test_metrics.items():
        logger.info("%s: %.4f", metric_name, value)
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()w
