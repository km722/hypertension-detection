"""
One-off script to extract StandardScaler mean/std from the preprocessing pipeline.
These values are hardcoded into the Streamlit app for portability.
"""
import sys
from pathlib import Path

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from src.preprocessing_pipeline import build_preprocessed_dataset

PANEL_A_NUMERIC = [
    "ACR_mg_g_log",
    "BMXBMI",
    "LBXSAL",
    "LBXSBU",
    "LBXSCR_log",
    "LBXSUA",
    "RIDAGEYR",
    "URXUCR",
    "URXUMA",
    "eGFR",
]

def main():
    print("Running preprocessing pipeline...")
    df_scaled, panels, scaler = build_preprocessed_dataset()
    print(f"Dataset shape: {df_scaled.shape}")
    print(f"Scaler fitted on {len(scaler.feature_names_in_)} features")

    # The scaler was fitted on: numeric_cols + ["ACR_mg_g_log", "LBXSCR_log"]
    feature_names = list(scaler.feature_names_in_)
    means = scaler.mean_
    stds = scaler.scale_

    print("\n--- Panel A Numeric Features: Scaler Parameters ---")
    print("SCALER_PARAMS = {")
    for feat in PANEL_A_NUMERIC:
        idx = feature_names.index(feat)
        print(f'    "{feat}": ({means[idx]:.10f}, {stds[idx]:.10f}),')
    print("}")

    # Save scaler as backup
    out_dir = ROOT / "reports" / "baselines" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = out_dir / "panel_a_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to {scaler_path}")

    # Also print Panel A feature list for verification
    print(f"\nPanel A features: {panels['PanelA_RenalDemo']}")

    # Load the LR model and print coefficients for verification
    model_path = out_dir / "PanelA_RenalDemo_logreg_tuned.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        print("\n--- LR Model Coefficients ---")
        panel_a_features = panels["PanelA_RenalDemo"]
        for feat, coef in zip(panel_a_features, model.coef_[0]):
            print(f"  {feat}: {coef:.6f}")
        print(f"  intercept: {model.intercept_[0]:.6f}")
    else:
        print(f"\nModel not found at {model_path}")

if __name__ == "__main__":
    main()
