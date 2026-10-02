"""Utility helpers: reproducibility, metrics, early stopping."""
import os
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def set_seed(seed: int):
    """Fix all relevant random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_metrics(y_true, y_pred, average="macro"):
    """Return accuracy/precision/recall/F1 as percentages."""
    return {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "precision": 100.0 * precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": 100.0 * recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": 100.0 * f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def regression_metrics(y_true, y_pred):
    """Return MAE / MSE / RMSE / R2 for visibility forecasting."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }


class EarlyStopping:
    """Stops training when the monitored validation metric stops improving."""

    def __init__(self, patience: int = 15, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state with the latest metric value; returns True if improved."""
        improved = False
        if self.best is None:
            self.best = value
            improved = True
        elif (self.mode == "min" and value < self.best - self.min_delta) or (
            self.mode == "max" and value > self.best + self.min_delta
        ):
            self.best = value
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
