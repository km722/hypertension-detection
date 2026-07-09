from pathlib import Path

import numpy as np
import pandas as pd

from eda_utils import load_cycle_dataframe, dataset_overview, missingness_table, class_balance, numeric_descriptives, correlation_matrix


def _as_text_table(df: pd.DataFrame, head: int | None = None) -> str:
    sample = df.head(head) if head is not None else df
    return "```\n" + sample.to_string(index=False) + "\n```"


def build_report(cycles, config_path="config/paths.json", output="reports/eda_checklist.md"):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cycle_frames = {}
    errors = {}
    for c in cycles:
        try:
            cycle_frames[c] = load_cycle_dataframe(c, config_path=config_path)
        except Exception as exc:  
            errors[c] = str(exc)

    if not cycle_frames:
        raise RuntimeError(f"Unable to load any cycles. Errors: {errors}")

    merged = pd.concat(cycle_frames.values(), ignore_index=True, sort=False)

    overview = dataset_overview(merged)
    missing = missingness_table(merged)
    balance = class_balance(merged)
    numeric_cols = merged.select_dtypes(include="number").columns.tolist()
    descriptives = numeric_descriptives(merged, numeric_cols)
    corr = correlation_matrix(merged[numeric_cols])
    high_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = (
        high_corr.stack()
        .reset_index()
        .rename(columns={0: "corr", "level_0": "var1", "level_1": "var2"})
        .query("abs(corr) >= 0.8")
    )

    lines = []
    lines.append("# Exploratory Data Analysis Checklist Results")
    lines.append("")
    lines.append("## Dataset overview")
    lines.append(f"- Participants per cycle: { {k: int(v.shape[0]) for k, v in cycle_frames.items()} }")
    lines.append(f"- Total adults (>=20y): {int(merged.shape[0])}")
    lines.append(f"- Unique SEQN: {int(merged['SEQN'].nunique())}")
    lines.append(f"- Duplicate SEQN rows: {int(merged['SEQN'].duplicated().sum())}")
    if errors:
        lines.append("")
        lines.append("## Cycles with load issues")
        for key, msg in errors.items():
            lines.append(f"- {key}: {msg}")
    lines.append("")

    lines.append("## Missingness")
    lines.append(_as_text_table(missing.reset_index().rename(columns={"index": "variable"}), head=10))
    lines.append("")

    lines.append("## Class balance (HTN_13080)")
    lines.append(_as_text_table(balance, head=None))
    lines.append("")

    lines.append("## Numeric descriptives (first 10 variables)")
    lines.append(_as_text_table(descriptives, head=10))
    lines.append("")

    lines.append("## High correlations |ρ| ≥ 0.8")
    if high_corr.empty:
        lines.append("No variable pairs exceeded |ρ| ≥ 0.8.")
    else:
        lines.append(_as_text_table(high_corr, head=None))

    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    build_report(["2015_2016", "2017_2018"])


