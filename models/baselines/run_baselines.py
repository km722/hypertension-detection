"""Baseline classifier training and evaluation across biomarker panels.

Trains logistic regression, decision tree, kNN, and linear SVM on the
NHANES 2015-2016 cohort and evaluates on the held-out 2017-2018 cohort.
Supports optional hyperparameter tuning via GridSearchCV.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from baseline_utils import (
    SplitData,
    compute_calibration_curve,
    ensure_dir,
    evaluate_predictions,
    load_dataset,
    load_panels,
    temporal_split,
)


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Instantiate baseline classifiers with sensible defaults.

    LR and SVM are linear models expected to perform well given the
    predominantly linear biomarker-hypertension relationship. DT and kNN
    serve as non-linear references.
    """
    return {
        "logreg": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, random_state=random_state
        ),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "svm_linear": SVC(
            kernel="linear", probability=True, random_state=random_state
        ),
    }


def fit_and_evaluate(
    split: SplitData,
    model_name: str,
    model: object,
    output_dir: str,
    panel_name: str,
    run_label: str,
) -> Tuple[Dict[str, float], object]:
    """Train a model on the training split, predict on test, and return metrics."""
    x_train = split.x_train.to_numpy()
    x_test = split.x_test.to_numpy()
    model.fit(x_train, split.y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(split.y_test.to_numpy(), y_prob)

    # generate calibration plot for visual inspection
    mean_pred, frac_pos = compute_calibration_curve(
        split.y_test.to_numpy(), y_prob
    )
    plot_calibration_curve(
        mean_pred, frac_pos, model_name, panel_name, output_dir, run_label,
    )
    return metrics, model


def plot_calibration_curve(
    mean_pred: np.ndarray,
    frac_pos: np.ndarray,
    model_name: str,
    panel_name: str,
    output_dir: str,
    run_label: str,
) -> None:
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
    plt.title(f"Calibration: {panel_name} - {model_name}_{run_label}")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    filename = f"calibration_{panel_name}_{model_name}_{run_label}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def run_baselines(
    data_path: str,
    panels_path: str,
    output_dir: str,
    tune: bool,
) -> pd.DataFrame:
    df = load_dataset(data_path)
    panels = load_panels(panels_path)
    models = build_models()
    results: List[Dict[str, float]] = []

    # grid search spaces for each model family
    param_grids = {
        "logreg": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        "decision_tree": {
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "min_samples_leaf": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        },
        "svm_linear": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
    }
    run_label = "tuned" if tune else "untuned"

    for panel_name, panel_cols in panels.items():
        split = temporal_split(df, panel_cols)
        for model_name, model in models.items():
            print(f"\n{model_name} | {panel_name} | train={split.x_train.shape[0]}, test={split.x_test.shape[0]}")

            param_grid = param_grids.get(model_name, {})
            best_params = {}
            best_score = np.nan

            # stratified 5-fold CV tuning, optimising AUC
            if tune and param_grid:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring="roc_auc", n_jobs=1, verbose=1,
                )
                grid_search.fit(split.x_train.to_numpy(), split.y_train.to_numpy())
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                print(f"  best params: {best_params}, CV AUC: {best_score:.4f}")
            metrics, trained_model = fit_and_evaluate(
                split=split,
                model_name=model_name,
                model=model,
                output_dir=os.path.join(output_dir, "calibration"),
                panel_name=panel_name,
                run_label=run_label,
            )
            # persist trained model for later SHAP analysis and deployment
            models_dir = os.path.join(output_dir, "models")
            ensure_dir(models_dir)
            model_path = os.path.join(models_dir, f"{panel_name}_{model_name}_{run_label}.joblib")
            joblib.dump(trained_model, model_path)

            row = {
                "panel": panel_name,
                "model": model_name,
                "tuning": run_label,
                "best_params": best_params,
                "cv_auc": best_score,
                **metrics,
            }
            results.append(row)
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline models across biomarker panels."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with stratified cross-validation.",
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
        default=os.path.join("reports", "baselines"),
        help="Output directory for metrics and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)
    results = run_baselines(args.data, args.panels, args.output, args.tune)
    suffix = "tuned" if args.tune else "untuned"
    metrics_path = os.path.join(args.output, f"baseline_metrics_{suffix}.csv")
    results.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

