"""Shared utilities for data loading, temporal splitting, and model evaluation.

All model scripts (baselines, ensembles, decision curves) import from here
to keep the evaluation pipeline consistent across experiments.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "HTN_13080" not in df.columns:
        raise ValueError("Target column HTN_13080 not found in dataset.")
    if "cycle" not in df.columns:
        raise ValueError("Column cycle not found for temporal split.")
    return df


def load_panels(panels_path: str) -> Dict[str, List[str]]:
    with open(panels_path, "r") as fh:
        panels = json.load(fh)
    if not panels:
        raise ValueError("Panel definitions are empty.")
    return panels


def temporal_split(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "HTN_13080",
    train_cycle: str = "2015_2016",
    test_cycle: str = "2017_2018",
) -> SplitData:
    """Split by NHANES cycle: train on 2015-2016, test on 2017-2018."""
    cols = [c for c in feature_cols if c in df.columns]
    missing = sorted(set(feature_cols) - set(cols))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    train_mask = df["cycle"] == train_cycle
    test_mask = df["cycle"] == test_cycle
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Temporal split failed; check cycle labels in data.")
    x_train = df.loc[train_mask, cols]
    y_train = df.loc[train_mask, target_col].astype(int)
    x_test = df.loc[test_mask, cols]
    y_test = df.loc[test_mask, target_col].astype(int)
    ensure_no_missing(x_train, "train")
    ensure_no_missing(x_test, "test")
    return SplitData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

def ensure_no_missing(df: pd.DataFrame, label: str) -> None:
    missing = df.isna().sum()
    if missing.any():
        missing_cols = missing[missing > 0].index.tolist()
        raise ValueError(f"Missing values in {label} features: {missing_cols}")


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUC, Brier, sensitivity, specificity, precision, NPV, F1 at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    precision, sensitivity, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    npv = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {
        "auc": auc,
        "brier": brier,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile",
    )
    return mean_pred, frac_pos


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

