"""Decision curve analysis for evaluating clinical utility.

Computes net benefit across a range of threshold probabilities, comparing
model-guided screening against treat-all and treat-none strategies.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from models.baselines.baseline_utils import (
    ensure_dir,
    load_dataset,
    load_panels,
    temporal_split,
)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Net benefit = TP rate minus weighted FP rate at a given threshold."""
    preds = (y_prob >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    n = len(y_true)
    harm = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * harm


def compute_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return np.array([net_benefit(y_true, y_prob, t) for t in thresholds])


def build_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "svm_linear": SVC(kernel="linear", probability=True, random_state=random_state),
    }


def run_decision_curves(
    data_path: str,
    panels_path: str,
    output_dir: str,
) -> None:
    df = load_dataset(data_path)
    panels = load_panels(panels_path)
    thresholds = np.linspace(0.05, 0.5, 10)
    ensure_dir(output_dir)

    for panel_name, panel_cols in panels.items():
        split = temporal_split(df, panel_cols)
        y_true = split.y_test.to_numpy()

        # reference strategies: treat-all and treat-none baselines
        prevalence = y_true.mean()
        treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        treat_none = np.zeros_like(thresholds)

        rows: List[Dict[str, float]] = []
        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, treat_all, label="treat_all", linestyle="--")
        plt.plot(thresholds, treat_none, label="treat_none", linestyle="--")

        for model_name, model in build_models().items():
            x_train = split.x_train.to_numpy()
            x_test = split.x_test.to_numpy()
            model.fit(x_train, split.y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            nb = compute_decision_curve(y_true, y_prob, thresholds)
            plt.plot(thresholds, nb, label=model_name)
            for t, val in zip(thresholds, nb):
                rows.append(
                    {
                        "panel": panel_name,
                        "model": model_name,
                        "threshold": float(t),
                        "net_benefit": float(val),
                    }
                )

        plt.title(f"Decision Curve Analysis: {panel_name}")
        plt.xlabel("Threshold probability")
        plt.ylabel("Net benefit")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"decision_curve_{panel_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        df_out = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"decision_curve_{panel_name}.csv")
        df_out.to_csv(csv_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decision-curve analysis plots for baseline models."
    )
    parser.add_argument(
        "--data",
        default=os.path.join("data", "clean", "processed_dataset.csv"),
        help="Path to processed dataset CSV.",
    )
    parser.add_argument(
        "--panels",
        default=os.path.join("data", "clean", "feature_panels.json"),
        help="Path to feature panels JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("reports", "decision_curves"),
        help="Output directory for decision-curve plots and tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_decision_curves(args.data, args.panels, args.output)
    print(f"Saved decision curves to {args.output}")


if __name__ == "__main__":
    main()

