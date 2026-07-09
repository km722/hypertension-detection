"""Ensemble model training with Bayesian hyperparameter optimisation.

Implements Random Forest, LightGBM, XGBoost, Soft Voting, and Stacking.
Hyperparameters are tuned using Optuna's TPE sampler with 5-fold stratified
CV on the NHANES 2015-2016 training set, maximising AUC.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.svm import SVC

import lightgbm as lgb
import xgboost as xgb

from models.baselines.baseline_utils import (
    SplitData,
    compute_calibration_curve,
    ensure_dir,
    evaluate_predictions,
    load_dataset,
    load_panels,
    temporal_split,
)


def tune_model(model_name, X_train, y_train, n_trials=100, random_state=42):
    """Bayesian optimisation via Optuna TPE for a single model family."""

    def objective(trial):
        # define search space per model family
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            }
            model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)

        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = lgb.LGBMClassifier(**params, random_state=random_state, verbose=-1, n_jobs=-1)

        elif model_name == "xgboost":
            # includes L1/L2 regularisation and gamma (min split loss)
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            }
            model = xgb.XGBClassifier(**params, random_state=random_state, eval_metric="logloss", n_jobs=-1)

        else:
            raise ValueError(f"Unknown model for tuning: {model_name}")

        # 5-fold stratified CV, maximising AUC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best {model_name} params: {study.best_params}")
    print(f"Best {model_name} CV AUC: {study.best_value:.4f}")

    return study.best_params


def build_models(
    random_state: int = 42,
    tuned_params: Dict[str, Dict] | None = None,
) -> Dict[str, object]:
    """Construct all ensemble models with optional Optuna-tuned parameters."""
    if tuned_params is None:
        tuned_params = {}

    # tree-based base models (tuned params override defaults)
    rf_params = tuned_params.get("random_forest", {
        "n_estimators": 250,
        "max_depth": None,
        "min_samples_leaf": 20,
    })
    rf = RandomForestClassifier(
        **rf_params,
        random_state=random_state,
        n_jobs=-1,
    )

    lgb_params = tuned_params.get("lightgbm", {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "num_leaves": 31,
    })
    lgb_model = lgb.LGBMClassifier(
        **lgb_params,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )

    xgb_params = tuned_params.get("xgboost", {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    })
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    # linear models included for diversity in meta-ensembles
    logreg = LogisticRegression(max_iter=1000, random_state=random_state)
    svm = SVC(kernel="linear", probability=True, random_state=random_state)

    # soft voting: average predicted probabilities across diverse base learners
    voting = VotingClassifier(
        estimators=[
            ("logreg", logreg),
            ("svm", svm),
            ("rf", rf),
            ("lgb", lgb_model),
        ],
        voting="soft",
        n_jobs=-1,
    )

    # stacking: out-of-fold predictions fed to an LR meta-learner
    stacking = StackingClassifier(
        estimators=[
            ("logreg", logreg),
            ("svm", svm),
            ("rf", rf),
            ("lgb", lgb_model),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    models: Dict[str, object] = {
        "random_forest": rf,
        "lightgbm": lgb_model,
        "xgboost": xgb_model,
        "voting": voting,
        "stacking": stacking,
    }

    return models


def fit_and_evaluate(
    split: SplitData,
    model_name: str,
    model: object,
    output_dir: str,
    panel_name: str,
    run_label: str = "tuned",
) -> Tuple[Dict[str, float], object]:

    x_train = split.x_train.to_numpy()
    x_test = split.x_test.to_numpy()
    y_train = split.y_train.to_numpy()
    
    # early stopping for boosting: hold out 20% of training data
    if model_name in ["lightgbm", "xgboost"]:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        if model_name == "lightgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                eval_names=["train", "valid"],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            train_scores = model.evals_result_["train"]["auc"]
            val_scores = model.evals_result_["valid"]["auc"]

        else:  # xgboost
            model.set_params(early_stopping_rounds=50)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                verbose=False,
            )
            results = model.evals_result()
            train_scores = results["validation_0"]["logloss"]
            val_scores = results["validation_1"]["logloss"]

        # plot convergence curve (train vs validation by iteration)
        training_curves_dir = os.path.join(output_dir, "training_curves")
        plot_training_curve(
            train_scores=train_scores,
            val_scores=val_scores,
            model_name=f"{model_name}_{run_label}",
            panel_name=panel_name,
            output_dir=training_curves_dir,
            metric_name="AUC" if model_name == "lightgbm" else "Log Loss",
            higher_is_better=(model_name == "lightgbm"),  # AUC: higher better, LogLoss: lower better
        )
    else:
        model.fit(x_train, y_train)

    # evaluate on held-out 2017-2018 test set
    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(split.y_test.to_numpy(), y_prob)

    # calibration plot
    mean_pred, frac_pos = compute_calibration_curve(
        split.y_test.to_numpy(), y_prob
    )
    calibration_dir = os.path.join(output_dir, "calibration")
    plot_calibration_curve(
        mean_pred,
        frac_pos,
        f"{model_name}_{run_label}",
        panel_name,
        calibration_dir,
    )
    
    # persist model for downstream SHAP and deployment
    models_dir = os.path.join(output_dir, "models")
    ensure_dir(models_dir)
    model_path = os.path.join(models_dir, f"{panel_name}_{model_name}_{run_label}.joblib")
    joblib.dump(model, model_path)
    
    return metrics, model   
  

def plot_calibration_curve(
    mean_pred: np.ndarray,
    frac_pos: np.ndarray,
    model_name: str,
    panel_name: str,
    output_dir: str,
) -> None:
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
    plt.title(f"Calibration: {panel_name} - {model_name}")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    filename = f"calibration_{panel_name}_{model_name}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_training_curve(
    train_scores: List[float],
    val_scores: List[float],
    model_name: str,
    panel_name: str,
    output_dir: str,
    metric_name: str = "AUC",
    higher_is_better: bool = True,
) -> None:
    """Plot training vs validation score per iteration."""
    ensure_dir(output_dir)

    plt.figure(figsize=(8, 5))
    iterations = range(1, len(train_scores) + 1)
    plt.plot(iterations, train_scores, label=f"Train {metric_name}", color="blue")
    plt.plot(iterations, val_scores, label=f"Val {metric_name}", color="orange")

    # Mark the early stopping point (best validation score)
    if higher_is_better:
        best_iteration = val_scores.index(max(val_scores)) + 1
    else:
        best_iteration = val_scores.index(min(val_scores)) + 1
    plt.axvline(x=best_iteration, color="red", linestyle="--", label=f"Best iteration: {best_iteration}")

    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title(f"Training Curve: {panel_name} - {model_name}")
    plt.legend()
    plt.tight_layout()

    filename = f"training_curve_{panel_name}_{model_name}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()





def compute_shap_importance(
    model: object,
    x_background: np.ndarray,
    feature_names: List[str],
    output_path: str,
    model_name: str,
    panel_name: str,
) -> None:
    try:
        import shap  # type: ignore
    except ImportError:
        return

    if x_background.shape[0] > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_background.shape[0], size=500, replace=False)
        x_background = x_background[idx]

    explainer = shap.Explainer(model, x_background)
    shap_values = explainer(x_background)
    values = shap_values.values

    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "model": model_name,
            "panel": panel_name,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)


def extract_stacking_coefficients(
    model: StackingClassifier,
    panel_name: str,
    output_dir: str,
) -> pd.DataFrame:
    """Extract meta-learner coefficients to show base model contributions."""
    meta_learner = model.final_estimator_
    base_model_names = [name for name, _ in model.estimators]

    if hasattr(meta_learner, 'coef_'):
        # with predict_proba stacking, each base model contributes P(class=0) and P(class=1)
        coefficients = meta_learner.coef_[0]

        coef_labels = []
        for name in base_model_names:
            coef_labels.extend([f"{name}_prob0", f"{name}_prob1"])

        df = pd.DataFrame({
            "base_model_feature": coef_labels[:len(coefficients)],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "panel": panel_name,
        }).sort_values("abs_coefficient", ascending=False)
        
        # simplified view: aggregate absolute weight per base model
        model_importance = {}
        for name in base_model_names:
            mask = df["base_model_feature"].str.startswith(name)
            model_importance[name] = df.loc[mask, "abs_coefficient"].sum()
        
        df_simple = pd.DataFrame({
            "base_model": list(model_importance.keys()),
            "total_abs_coefficient": list(model_importance.values()),
            "panel": panel_name,
        }).sort_values("total_abs_coefficient", ascending=False)
        
        ensure_dir(output_dir)
        df.to_csv(os.path.join(output_dir, f"stacking_coefficients_detailed_{panel_name}.csv"), index=False)
        df_simple.to_csv(os.path.join(output_dir, f"stacking_coefficients_{panel_name}.csv"), index=False)
        
        print(f"\nStacking meta-learner coefficients for {panel_name}:")
        print(df_simple.to_string(index=False))
        
        return df_simple
    else:
        print(f"Meta-learner has no coefficients (not a linear model)")
        return pd.DataFrame()


def run_ensembles(
    data_path: str,
    panels_path: str,
    output_dir: str,
    model_filter: List[str] | None = None,
    compute_shap: bool = False,
    tune: bool = False,
    n_trials: int = 100,
) -> pd.DataFrame:
    df = load_dataset(data_path)
    panels = load_panels(panels_path)

    # If tuning, run Optuna on training data first
    tuned_params = {}
    if tune:
        # Use first panel's data for tuning (Panel A)
        first_panel = list(panels.keys())[0]
        split = temporal_split(df, panels[first_panel])
        X_train = split.x_train.to_numpy()
        y_train = split.y_train.to_numpy()

        for model_name in ["random_forest", "lightgbm", "xgboost"]:
            print(f"\n=== Tuning {model_name} ===")
            tuned_params[model_name] = tune_model(model_name, X_train, y_train, n_trials=n_trials)
        
        # Save tuned params to JSON for reproducibility
        tuning_dir = os.path.join(output_dir, "tuning")
        ensure_dir(tuning_dir)
        params_path = os.path.join(tuning_dir, "tuned_params.json")
        with open(params_path, "w") as f:
            json.dump(tuned_params, f, indent=2)
        print(f"\nSaved tuned params to {params_path}")

    # Build models (if passing --tune argument)
    models = build_models(tuned_params=tuned_params if tune else None)

    if model_filter:
        models = {k: v for k, v in models.items() if k in model_filter}
        if not models:
            raise ValueError(f"No models matched filter: {model_filter}")

    results: List[Dict[str, float]] = []

    for panel_name, panel_cols in panels.items():
        split = temporal_split(df, panel_cols)

        for model_name, model in models.items():
            print(f"\n=== Training {model_name} on {panel_name} ===")
            metrics, trained_model = fit_and_evaluate(
                split=split,
                model_name=model_name,
                model=model,
                output_dir=output_dir,
                panel_name=panel_name,
                run_label="tuned" if tune else "untuned",
            )
            row = {
                "panel": panel_name,
                "model": model_name,
                "tuning": "tuned" if tune else "untuned",
                **metrics,
            }
            results.append(row)

            if compute_shap and model_name in {"random_forest", "lightgbm", "xgboost"}:
                x_train = split.x_train.to_numpy()
                shap_path = os.path.join(
                    output_dir,
                    "shap",
                    f"shap_{panel_name}_{model_name}.csv",
                )
                compute_shap_importance(
                    model=trained_model,
                    x_background=x_train,
                    feature_names=list(split.x_train.columns),
                    output_path=shap_path,
                    model_name=model_name,
                    panel_name=panel_name,
                )
            
            # Extract stacking meta-learner coefficients
            if model_name == "stacking":
                stacking_dir = os.path.join(output_dir, "stacking")
                extract_stacking_coefficients(
                    model=trained_model,
                    panel_name=panel_name,
                    output_dir=stacking_dir,
                )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble/advanced models across biomarker panels."
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before training.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials per model (default: 100).",
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
        default=os.path.join("reports", "ensembles"),
        help="Output directory for metrics and plots.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run (e.g., random_forest svm_rbf).",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute SHAP importance for tree-based models.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)

    results = run_ensembles(
        args.data,
        args.panels,
        args.output,
        args.models,
        compute_shap=args.shap,
        tune=args.tune,
        n_trials=args.trials,
    )

    metrics_path = os.path.join(args.output, "ensemble_metrics.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        results = pd.concat([existing, results], ignore_index=True)
        results = results.drop_duplicates(subset=["panel", "model"], keep="last")

    results.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()