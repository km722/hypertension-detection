# Detecting Undiagnosed Hypertension from Routine Biomarkers

Final-year BSc dissertation (Heriot-Watt University, 2026). A large share of hypertensive adults are undiagnosed, but most already get routine blood tests. This project asks whether ensemble models can flag at-risk patients from renal and haematological biomarkers they already have on file.

## Data

NHANES 2015-2018 (public CDC data), n = 10,739 adults after filtering. Raw SAS XPORT files sit in `data/raw/`. The preprocessing pipeline (`src/preprocessing_pipeline.py`) harmonises the two survey cycles, screens implausible measurements, imputes missing values, winsorises extreme tails and produces three feature panels: renal + demographics, CBC, and combined.

## Approach

Baselines are logistic regression and a single decision tree. On top of those I trained random forests (bagging), XGBoost and LightGBM (boosting), soft voting, and stacking ensembles. Validation is temporal: models are trained on the 2015-2016 cycle and evaluated on 2017-2018, to mimic deployment on future patients rather than a random split. Interpretability comes from SHAP on the tree ensembles, cross-checked against logistic regression coefficients (rank concordance rho = 0.89).

## Results

The best model (tuned stacking on the combined panel) reached AUC 0.823 under temporal validation. Full metrics for every model/panel combination are in `reports/master_metrics.csv`, with calibration curves, decision curves and SHAP outputs under `reports/`.

A deliberately parsimonious 3-feature model (age, BMI, eGFR) retains 92.1% of the discriminative ability above chance, which matters clinically: it needs almost no inputs and is fully explainable.

## Screening tool

`app/presentation_app.py` is a small Streamlit demo that turns 11 routine clinical inputs into a calibrated risk estimate with per-feature explanations. It is a thin presentation layer over the trained models; the substance of the project is in `src/`, `models/` and the notebooks.

    pip install -r requirements.txt
    streamlit run app/presentation_app.py

## Layout

    config/       paths configuration
    data/raw/     NHANES XPORT files (CDC, public)
    src/          preprocessing pipeline and EDA utilities
    models/       baseline and ensemble training code
    notebooks/    EDA and figure generation
    reports/      metrics, calibration/decision curves, SHAP outputs
    app/          Streamlit demo

Serialized model files are not committed (size); results are reproducible from the training code with fixed seeds.

This is research code written for a dissertation, not a production system, and the model is a screening aid rather than a diagnostic tool.
