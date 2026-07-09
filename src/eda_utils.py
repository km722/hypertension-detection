import json
import os
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd

REQUIRED_KEYS = ["DEMO", "BPQ", "BPX", "CBC", "BIOPRO", "ALB_CR", "BMX"]

# Read SAS transport file for a given path.
def _read_xpt(path: str) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")

# Load all raw files for a cycle using the paths config.
def _load_cycle_raw(config: Dict, cycle_key: str) -> Dict[str, pd.DataFrame]:
    info = config[cycle_key]
    base = info["folder"]
    files = info["files"]
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        fpath = os.path.join(base, fname)
        dfs[key] = _read_xpt(fpath) if os.path.exists(fpath) else None
    return dfs

# Merge the essential NHANES components and derive helper variables.
def _merge_minimal(demo, bpq, bpx, cbc, biopro, albcr, bmx) -> pd.DataFrame:
    keep_demo = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "WTMEC2YR"]
    keep_bpq = ["SEQN", "BPQ020", "BPQ040A", "BPQ050A"]
    keep_bpx = ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2",
                "BPXDI3", "BPXDI4"]
    keep_cbc = ["SEQN", "LBXWBCSI", "LBXHGB", "LBXHCT", "LBXPLTSI", "LBXRDW"]
    keep_biopro = ["SEQN", "LBXSCR", "LBXSBU", "LBXSUA", "LBXSAL"]
    keep_albcr = ["SEQN", "URXUMA", "URXUCR"]
    keep_bmx = ["SEQN", "BMXBMI"]
    demo = demo[[c for c in keep_demo if c in demo.columns]]
    bpq = bpq[[c for c in keep_bpq if c in bpq.columns]]
    bpx = bpx[[c for c in keep_bpx if c in bpx.columns]]
    cbc = cbc[[c for c in keep_cbc if c in cbc.columns]]
    biopro = biopro[[c for c in keep_biopro if c in biopro.columns]]
    albcr = albcr[[c for c in keep_albcr if c in albcr.columns]]
    bmx = bmx[[c for c in keep_bmx if c in bmx.columns]]
    df = (
        demo.merge(bpq, on="SEQN", how="inner")
        .merge(bpx, on="SEQN", how="inner")
        .merge(cbc, on="SEQN", how="inner")
        .merge(biopro, on="SEQN", how="inner")
        .merge(albcr, on="SEQN", how="inner")
        .merge(bmx, on="SEQN", how="inner")
    )
    if "RIDAGEYR" in df.columns:
        df = df[df["RIDAGEYR"] >= 20].copy()
    sbp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in df.columns]
    dbp_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]
    df["SBP"] = df[sbp_cols].mean(axis=1)
    df["DBP"] = df[dbp_cols].mean(axis=1)
    df["HTN_self"] = (df.get("BPQ020", pd.Series([np.nan] * len(df))) == 1).astype(int)
    # Use only BPQ050A (currently taking meds) for accurate current HTN status
    df["HTN_meds"] = (df.get("BPQ050A", pd.Series([np.nan] * len(df))) == 1).astype(int)
    df["HTN_13080"] = ((df["SBP"] >= 130) | (df["DBP"] >= 80) | (df["HTN_self"] == 1) |
                       (df["HTN_meds"] == 1)).astype(int)
    if all(c in df.columns for c in ["LBXSCR", "RIDAGEYR", "RIAGENDR"]):
        df["eGFR"] = df.apply(lambda r: _egfr_ckdepi_2021(r["LBXSCR"], r["RIDAGEYR"],
                                                          r["RIAGENDR"]), axis=1)
    else:
        df["eGFR"] = np.nan
    if "URXUMA" in df.columns and "URXUCR" in df.columns:
        df["ACR_mg_g"] = (df["URXUMA"] / df["URXUCR"]) * 100
    else:
        df["ACR_mg_g"] = np.nan
    return df

# CKD-EPI 2021 creatinine-only equation; sex_code: 1=male, 2=female.
def _egfr_ckdepi_2021(scr, age, sex_code):
    k = 0.9 if sex_code == 1 else 0.7
    alpha = -0.302 if sex_code == 1 else -0.241
    min_ratio = min(scr / k, 1) ** alpha
    max_ratio = max(scr / k, 1) ** (-1.200)
    sex_factor = 1 if sex_code == 1 else 1.012
    return 142 * min_ratio * max_ratio * (0.9938 ** age) * sex_factor

# Define feature panels (renal + demo, CBC, combined).
def _define_panels() -> Dict[str, Tuple[str, ...]]:
    renal = ("eGFR", "LBXSCR", "LBXSBU", "LBXSUA", "LBXSAL", "URXUMA", "URXUCR",
             "ACR_mg_g")
    cbc = ("LBXRDW", "LBXHGB", "LBXHCT", "LBXWBCSI", "LBXPLTSI")
    demo = ("RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI")
    panel_a = renal + demo
    panel_b = cbc
    panel_c = tuple(sorted(set(panel_a + panel_b)))
    return {
        "PanelA_RenalDemo": panel_a,
        "PanelB_CBC": panel_b,
        "PanelC_Combined": panel_c,
    }

# Read JSON config listing all NHANES file paths.
def _read_paths_config(config_path: str = "config/paths.json") -> Dict:
    with open(config_path, "r") as fh:
        return json.load(fh)

def load_cycle_dataframe(
    cycle_key: str, config_path: str = "config/paths.json", *, add_cycle_flag: bool =
    True
) -> pd.DataFrame:
    config = _read_paths_config(config_path)
    if cycle_key not in config:
        raise KeyError(f"Cycle key '{cycle_key}' is not defined in {config_path}")
    dfs = _load_cycle_raw(config, cycle_key)
    missing = [k for k in REQUIRED_KEYS if dfs.get(k) is None]
    if missing:
        raise FileNotFoundError(f"Missing files for cycle {cycle_key}: {', '.join(missing)}")
    df = _merge_minimal(
        dfs["DEMO"],
        dfs["BPQ"],
        dfs["BPX"],
        dfs["CBC"],
        dfs["BIOPRO"],
        dfs["ALB_CR"],
        dfs["BMX"],
    )
    if add_cycle_flag:
        df = df.assign(cycle=cycle_key)
    return df

# Merge multiple cycles together with consistent schema.
def load_merged_cycles(cycles: Iterable[str], config_path: str = "config/paths.json") -> pd.DataFrame:
    frames = [load_cycle_dataframe(c, config_path=config_path) for c in cycles]
    return pd.concat(frames, ignore_index=True, sort=False)

# Overview table with dtypes, non-null counts, and simple stats.
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "non_null": df.notnull().sum(),
    })
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_stats = df[numeric_cols].agg(["min", "max", "mean",
                                          "median"]).T.reset_index().rename(
        columns={"index": "column"}
    )
    overview = base.merge(numeric_stats, on="column", how="left")
    overview["missing_pct"] = (1 - overview["non_null"] / len(df)) * 100
    return overview.sort_values("column").reset_index(drop=True)

# Missingness sorted by percentage.
def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    table = (
        df.isna()
        .sum()
        .rename("missing")
        .to_frame()
        .assign(missing_pct=lambda x: (x["missing"] / total) * 100)
        .sort_values("missing_pct", ascending=False)
    )
    return table

# Class-balance helper for HTN_13080.
def class_balance(df: pd.DataFrame, target: str = "HTN_13080") -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False).rename("count").to_frame()
    counts["percent"] = (counts["count"] / counts["count"].sum()) * 100
    counts.index.name = target
    return counts.reset_index()

# Descriptive statistics (with skew/kurtosis) for selected numeric columns.
def numeric_descriptives(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats.rename(
        columns={
            "50%": "median",
            "25%": "q1",
            "75%": "q3",
        }
    )
    stats["skew"] = df[cols].skew()
    stats["kurtosis"] = df[cols].kurtosis()
    return stats.reset_index().rename(columns={"index": "column"})

# Correlation matrix (default Spearman) for numeric columns.
def correlation_matrix(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)

# Expose feature panel definitions to notebooks and scripts.
def list_panels() -> Dict[str, Tuple[str, ...]]:
    return _define_panels()
