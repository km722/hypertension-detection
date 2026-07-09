"""
Processing pipeline for the NHANES 2015–2018 hypertension project.
Processing stages
-----------------
1. Load raw NHANES cycles (2015–2016, 2017–2018) and harmonise by SEQN.
2. Filter adults (≥20 years), exclude pregnancy records, and exclude rows with uncertain HTN status.
3. Screen for implausible biomarker measurements and mark them as missing.
4. Impute missing values (median for numeric, mode for categorical) and add
"*_imputed" flags wherever ≥10% of rows were missing.
5. Winsorise numeric variables (5th/95th percentiles) to tame extreme tails.
6. Recompute derived fields (mean SBP/DBP, HTN_13080, eGFR, ACR_mg_g) after cleaning.
7. Apply log1p transformations to ACR_mg_g and LBXSCR, then encode demographics.
8. Standardise numeric predictors with z-scores.
9. Output feature panels (renal+demographics, CBC, combined) for modelling.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint
# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------
def read_paths_config(config_path: str = "config/paths.json") -> Dict:
    with open(config_path, "r") as fh:
        return json.load(fh)
def read_xpt(path: str) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")
REQUIRED_KEYS = ["DEMO", "BPQ", "BPX", "CBC", "BIOPRO", "ALB_CR", "BMX"]
def load_cycle_raw(config: Dict, cycle_key: str) -> Dict[str, pd.DataFrame]:
    info = config[cycle_key]
    base = info["folder"]
    files = info["files"]
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        fpath = os.path.join(base, fname)
        dfs[key] = read_xpt(fpath) if os.path.exists(fpath) else None
    return dfs
# -----------------------------------------------------------------------------
# Derived metrics and harmonisation utilities
# -----------------------------------------------------------------------------
def egfr_ckdepi_2021(scr, age, sex_code):
    """CKD-EPI 2021 refit; sex_code: 1=male, 2=female."""
    if pd.isna(scr) or pd.isna(age) or pd.isna(sex_code):
        return np.nan
    k = 0.9 if sex_code == 1 else 0.7
    alpha = -0.302 if sex_code == 1 else -0.241
    min_ratio = min(scr / k, 1) ** alpha
    max_ratio = max(scr / k, 1) ** (-1.200)
    sex_factor = 1 if sex_code == 1 else 1.012
    return 142 * min_ratio * max_ratio * (0.9938 ** age) * sex_factor
def merge_cycle(demo, bpq, bpx, cbc, biopro, albcr, bmx) -> pd.DataFrame:
    keep_demo = [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "WTMEC2YR",
        "RIDEXPRG", # Pregnancy status
    ]
    keep_bpq = ["SEQN", "BPQ020", "BPQ030", "BPQ040A", "BPQ050A"]
    keep_bpx = [
        "SEQN",
        "BPXSY1",
        "BPXSY2",
        "BPXSY3",
        "BPXSY4",
        "BPXDI1",
        "BPXDI2",
        "BPXDI3",
        "BPXDI4",
    ]
    keep_cbc = ["SEQN", "LBXWBCSI", "LBXHGB", "LBXHCT", "LBXPLTSI", "LBXRDW"]
    keep_biopro = ["SEQN", "LBXSCR", "LBXSBU", "LBXSUA", "LBXSAL"]
    keep_alb = ["SEQN", "URXUMA", "URXUCR"]
    keep_bmx = ["SEQN", "BMXBMI"]
    demo = demo[[c for c in keep_demo if c in demo.columns]].copy()
    bpq = bpq[[c for c in keep_bpq if c in bpq.columns]].copy()
    bpx = bpx[[c for c in keep_bpx if c in bpx.columns]].copy()
    cbc = cbc[[c for c in keep_cbc if c in cbc.columns]].copy()
    biopro = biopro[[c for c in keep_biopro if c in biopro.columns]].copy()
    albcr = albcr[[c for c in keep_alb if c in albcr.columns]].copy()
    bmx = bmx[[c for c in keep_bmx if c in bmx.columns]].copy()
    df = (
        demo.merge(bpq, on="SEQN", how="inner")
        .merge(bpx, on="SEQN", how="inner")
        .merge(cbc, on="SEQN", how="inner")
        .merge(biopro, on="SEQN", how="inner")
        .merge(albcr, on="SEQN", how="inner")
        .merge(bmx, on="SEQN", how="inner")
    )
    # Adults only & non-pregnant
    df = df[df["RIDAGEYR"] >= 20].copy()
    if "RIDEXPRG" in df.columns:
        df = df[df["RIDEXPRG"] != 1].copy()
    # Derived averages and labels
    sbp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in df.columns]
    dbp_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]
    df["SBP"] = df[sbp_cols].mean(axis=1)
    df["DBP"] = df[dbp_cols].mean(axis=1)
    df["HTN_self"] = (df.get("BPQ020") == 1).astype(int)
    # Use only BPQ050A (currently taking meds) for accurate current HTN status
    df["HTN_meds"] = (df.get("BPQ050A") == 1).astype(int)
    df["HTN_13080"] = ((df["SBP"] >= 130) | (df["DBP"] >= 80) | (df["HTN_self"] == 1) |
                       (df["HTN_meds"] == 1)).astype(int)
    # Exclude rows with uncertain HTN status (missing BP AND no self-report/meds)
    uncertain_htn = df["SBP"].isnull() & (df["HTN_self"] == 0) & (df["HTN_meds"] == 0)
    df = df[~uncertain_htn].copy()
    df["eGFR"] = df.apply(lambda r: egfr_ckdepi_2021(r.get("LBXSCR"), r.get("RIDAGEYR"),
                                                      r.get("RIAGENDR")), axis=1)
    if "URXUMA" in df.columns and "URXUCR" in df.columns:
        # Convert creatinine from mg/dL to g/L (/100) before computing mg/g ratio
        df["ACR_mg_g"] = (df["URXUMA"] / df["URXUCR"]) * 100
    else:
        df["ACR_mg_g"] = np.nan
    return df
# -----------------------------------------------------------------------------
# Data quality checks & preprocessing steps
# -----------------------------------------------------------------------------
IMPLAUSIBLE_RANGES: Dict[str, Tuple[float | None, float | None]] = {
    "BMXBMI": (10, 80),
    "SBP": (70, 260),
    "DBP": (40, 160),
    "LBXSCR": (0.2, 6),
    "LBXSUA": (0.5, 15),
    "LBXSAL": (2, 6),
    "LBXSBU": (1, 80),
    "LBXHGB": (6, 20),
    "LBXHCT": (20, 60),
    "LBXPLTSI": (50, 900),
    "LBXWBCSI": (1, 30),
    "ACR_mg_g": (0, 500),
}
def screen_implausible(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lower, upper) in IMPLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        mask = pd.Series(True, index=df.index)
        if lower is not None:
            mask &= df[col] >= lower
        if upper is not None:
            mask &= df[col] <= upper
        df.loc[~mask, col] = np.nan
    return df
def winsorize(df: pd.DataFrame, numeric_cols: Iterable[str], lower: float = 0.05, upper:
              float = 0.95) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df
def log_transform(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])
    return df
def recompute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate SBP/DBP, hypertension flags, eGFR, and ACR after
    imputation/winsorisation."""
    df = df.copy()
    sbp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in df.columns]
    dbp_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]
    if sbp_cols:
        df["SBP"] = df[sbp_cols].mean(axis=1)
    if dbp_cols:
        df["DBP"] = df[dbp_cols].mean(axis=1)
    df["HTN_self"] = (df.get("BPQ020") == 1).astype(int)
    df["HTN_meds"] = (df.get("BPQ050A") == 1).astype(int)
    df["HTN_13080"] = (
        (df.get("SBP", np.nan) >= 130)
        | (df.get("DBP", np.nan) >= 80)
        | (df["HTN_self"] == 1)
        | (df["HTN_meds"] == 1)
    ).astype(int)
    df["eGFR"] = df.apply(
        lambda r: egfr_ckdepi_2021(r.get("LBXSCR"), r.get("RIDAGEYR"),
                                    r.get("RIAGENDR")),
        axis=1,
    )
    if "URXUMA" in df.columns and "URXUCR" in df.columns:
        df["ACR_mg_g"] = (df["URXUMA"] / df["URXUCR"]) * 100
    else:
        df["ACR_mg_g"] = np.nan
    return df
def impute_with_flags(df: pd.DataFrame, numeric_cols: Iterable[str], categorical_cols:
                      Iterable[str], threshold: float = 0.10) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)
    # Numeric
    for col in numeric_cols:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        if missing_mask.mean() >= threshold:
            df[f"{col}_imputed"] = missing_mask.astype(int)
        df[col] = df[col].fillna(df[col].median())
    # Categorical
    for col in categorical_cols:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        if missing_mask.mean() >= threshold:
            df[f"{col}_imputed"] = missing_mask.astype(int)
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else 0
        df[col] = df[col].fillna(fill_value)
    return df
def encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RIAGENDR" in df.columns:
        df["sex_female"] = df["RIAGENDR"].map({1: 0, 2: 1})
    if "RIDRETH3" in df.columns:
        race_map = {
            1: "mexican_american",
            2: "other_hispanic",
            3: "non_hispanic_white",
            4: "non_hispanic_black",
            6: "non_hispanic_asian",
            7: "other_race",
        }
        df["RIDRETH3_label"] = df["RIDRETH3"].map(race_map)
        race_dummies = pd.get_dummies(df["RIDRETH3_label"], prefix="race",
                                      dummy_na=False, dtype=int)  # Use int instead of bool
        df = pd.concat([df, race_dummies], axis=1)
    return df
def scale_numeric(df: pd.DataFrame, numeric_cols: Iterable[str]) -> Tuple[pd.DataFrame,
                                                                           StandardScaler]:
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler
def define_feature_panels(df: pd.DataFrame) -> Dict[str, list]:
    renal = ["eGFR", "LBXSCR_log", "LBXSBU", "LBXSUA", "LBXSAL", "URXUMA", "URXUCR",
             "ACR_mg_g_log"]
    cbc = ["LBXRDW", "LBXHGB", "LBXWBCSI", "LBXPLTSI"]  # Removed LBXHCT (highly correlated with LBXHGB)
    demographics = ["RIDAGEYR", "sex_female", "BMXBMI"] + [col for col in df.columns if
                                                           col.startswith("race_")]
    panel_a = sorted(set(renal + demographics))
    panel_b = sorted(set(cbc + demographics))
    panel_c = sorted(set(panel_a + panel_b))
    panels = {
        "PanelA_RenalDemo": [col for col in panel_a if col in df.columns],
        "PanelB_CBC": [col for col in panel_b if col in df.columns],
        "PanelC_Combined": [col for col in panel_c if col in df.columns],
    }
    return panels
# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def build_preprocessed_dataset(
    cycles: Iterable[str] = ("2015_2016", "2017_2018"),
    config_path: str = "config/paths.json",
) -> Tuple[pd.DataFrame, Dict[str, list], StandardScaler]:
    config = read_paths_config(config_path)
    frames = []
    for cycle in cycles:
        dfs = load_cycle_raw(config, cycle)
        missing = [k for k in REQUIRED_KEYS if dfs.get(k) is None]
        if missing:
            raise FileNotFoundError(f"Cycle {cycle} missing files: {missing}")
        frames.append(
            merge_cycle(
                dfs["DEMO"],
                dfs["BPQ"],
                dfs["BPX"],
                dfs["CBC"],
                dfs["BIOPRO"],
                dfs["ALB_CR"],
                dfs["BMX"],
            ).assign(cycle=cycle)
        )
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = screen_implausible(df)
    numeric_cols = [
        "RIDAGEYR",
        "BMXBMI",
        "SBP",
        "DBP",
        "LBXSCR",
        "LBXSUA",
        "LBXSAL",
        "LBXSBU",
        "LBXHGB",
        "LBXHCT",
        "LBXPLTSI",
        "LBXWBCSI",
        "LBXRDW",
        "URXUMA",
        "URXUCR",
        "eGFR",
        "ACR_mg_g",
    ]
    categorical_cols = ["RIAGENDR", "RIDRETH3", "HTN_self", "HTN_meds"]
    df = impute_with_flags(df, numeric_cols, categorical_cols, threshold=0.10)
    df = winsorize(df, numeric_cols)
    df = recompute_derived_fields(df)
    df = log_transform(df, ["ACR_mg_g", "LBXSCR"]) # Transform highly skewed variables
    df = encode_demographics(df)
    df_scaled, scaler = scale_numeric(df, numeric_cols + ["ACR_mg_g_log", "LBXSCR_log"])
    panels = define_feature_panels(df_scaled)
    return df_scaled, panels, scaler
if __name__ == "__main__":
    dataset, panels, scaler = build_preprocessed_dataset()
    print("Processed dataset:", dataset.shape)
