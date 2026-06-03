"""
Seahorse OCR Normalization by Hoechst Fluorescence
====================================================
Normalizes Seahorse metabolic parameters by cell count (Hoechst fluorescence).

Input CSVs expected:
  - hoechst.csv  : columns [well, fluorescence]
  - seahorse.csv : columns [well, <parameter1>, <parameter2>, ...]

Output:
  - seahorse_normalized.csv : same structure, values divided by scaling factor

Scaling logic (median-anchored):
  scaling_factor(well) = fluorescence(well) / median(all fluorescence values)
  normalized_value(well) = raw_value(well) / scaling_factor(well)

This is equivalent to your A1-anchored approach but more robust — instead of
  A1/A1=1, A2/A1=1.1 ...
we use:
  A1/median=0.97, A2/median=1.07 ...
so no single well dominates the normalization.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def normalize_seahorse(hoechst_path: str, seahorse_path: str, output_path: str = None):
    # --- Load data ---
    hoechst = pd.read_csv(hoechst_path)
    seahorse = pd.read_csv(seahorse_path)

    # Normalise column names
    hoechst.columns = hoechst.columns.str.strip().str.lower()
    seahorse.columns = seahorse.columns.str.strip()

    # Expect 'well' and 'fluorescence' columns in hoechst
    assert "well" in hoechst.columns, "hoechst.csv must have a 'well' column"
    assert "fluorescence" in hoechst.columns, "hoechst.csv must have a 'fluorescence' column"
    assert "well" in seahorse.columns.str.lower().tolist(), "seahorse.csv must have a 'well' column"

    # Normalise well column name in seahorse too
    seahorse = seahorse.rename(columns={c: c.lower() if c.lower() == "well" else c for c in seahorse.columns})

    # --- Compute scaling factors ---
    plate_median = hoechst["fluorescence"].median()
    print(f"Plate median fluorescence: {plate_median:.2f}")

    hoechst["scaling_factor"] = hoechst["fluorescence"] / plate_median

    # Flag wells that deviate a lot from median (potential edge effects / outliers)
    flag_threshold = 0.3  # >30% deviation
    flagged = hoechst[abs(hoechst["scaling_factor"] - 1) > flag_threshold]
    if not flagged.empty:
        print(f"\n⚠  Wells with >30% deviation from median (check for edge effects / outliers):")
        print(flagged[["well", "fluorescence", "scaling_factor"]].to_string(index=False))

    # --- Merge and normalize ---
    merged = seahorse.merge(hoechst[["well", "scaling_factor"]], on="well", how="left")

    missing = merged["scaling_factor"].isna().sum()
    if missing > 0:
        print(f"\n⚠  {missing} Seahorse wells have no matching Hoechst value — they will not be normalized.")

    # All columns except 'well' and 'scaling_factor' are OCR/ECAR parameters
    param_cols = [c for c in merged.columns if c not in ("well", "scaling_factor")]

    normalized = merged.copy()
    for col in param_cols:
        normalized[col] = merged[col] / merged["scaling_factor"]

    normalized = normalized.drop(columns=["scaling_factor"])

    # --- Save ---
    if output_path is None:
        output_path = Path(seahorse_path).stem + "_normalized.csv"

    normalized.to_csv(output_path, index=False)
    print(f"\n✓ Normalized data saved to: {output_path}")

    # --- Summary stats ---
    print("\nScaling factor summary:")
    print(hoechst["scaling_factor"].describe().round(3).to_string())

    return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Seahorse data by Hoechst fluorescence.")
    parser.add_argument("hoechst", help="Path to hoechst.csv (columns: well, fluorescence)")
    parser.add_argument("seahorse", help="Path to seahorse.csv (columns: well, param1, param2, ...)")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: seahorse_normalized.csv)")
    args = parser.parse_args()

    normalize_seahorse(args.hoechst, args.seahorse, args.output)
