"""
seahorse_collate.py
====================
Two collation functions for aggregating data across N* experiment folders.

collate(directory, mode="normalised")
    Reads analysis/<mode>/parameters.csv from each N* folder and writes
    one CSV per metric into collated/<mode>/parameters/.
    Columns = conditions, rows = replicate values (Prism-ready).

collate_timeseries(directory, mode="normalised")
    Reads OCR_cleaned / ECAR_cleaned (normalised) or
         OCR_raw_cleaned / ECAR_raw_cleaned (raw)
    from the *_prepared.xlsx file in each N* folder.
    Each experiment contributes one column per rep per condition,
    labelled <Condition>_N1, <Condition>_N2, etc.
    All timing differences are ignored — the time column from the
    first valid experiment (N1) is used as the shared index.
    Writes collated/<mode>/timeseries/OCR.csv and ECAR.csv.

Usage:
    from seahorse_collate import collate, collate_timeseries

    collate("path/to/directory")
    collate("path/to/directory", mode="raw")

    collate_timeseries("path/to/directory")
    collate_timeseries("path/to/directory", mode="raw")
"""

import re
import pandas as pd
from pathlib import Path

METRICS = [
    "Basal_OCR",
    "Oligomycin_OCR",
    "FCCP_OCR",
    "RotAA_OCR",
    "Non_mitochondrial_respiration",
    "Basal_respiration",
    "ATP_linked_respiration",
    "Proton_leak",
    "Maximal_respiration",
    "Spare_respiratory_capacity",
    "Coupling_efficiency",
    "RCR_like_metric",
    "Basal_ECAR",
    "ECAR_after_oligomycin",
    "Compensatory_glycolysis",
]


def collate(directory, mode="normalised"):
    base = Path(directory)
    experiment_folders = sorted(base.glob("N*"))

    frames = []
    for folder in experiment_folders:
        params_path = folder / "analysis" / mode / "parameters.csv"
        if not params_path.exists():
            print(f"  Skipping {folder.name}: no parameters.csv found at {params_path}")
            continue
        df = pd.read_csv(params_path)
        frames.append(df)
        print(f"  Loaded: {params_path}")

    if not frames:
        print("No parameters.csv files found.")
        return

    combined = pd.concat(frames, ignore_index=True)

    out_dir = base / "collated" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        if metric not in combined.columns:
            continue
        treatments = combined["Treatment"].unique()
        cols = []
        for treatment in treatments:
            vals = (
                combined.loc[combined["Treatment"] == treatment, metric]
                .dropna()
                .reset_index(drop=True)
            )
            cols.append(vals.rename(treatment))
        export_df = pd.concat(cols, axis=1)
        out_path = out_dir / f"{metric}.csv"
        export_df.to_csv(out_path, index=False)

    print(f"\n✓ Collation complete. CSVs written to: {out_dir}/")


def _get_condition(col_name):
    """Strip _rep<N> suffix to recover the condition name."""
    return re.sub(r"_rep\s*\d+$", "", col_name, flags=re.IGNORECASE).strip()


def _timeseries_for_sheet(wb_path, sheet_name):
    """
    Read a _cleaned sheet and return:
        result  : { condition_name: [Series_rep1, Series_rep2, ...] }
                  each Series has a plain integer index (timing stripped)
        time_values : numpy array of time values, or None
    """
    df = pd.read_excel(wb_path, sheet_name=sheet_name, header=0)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")

    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    rep_cols = [c for c in df.columns if c not in ("Cycle", time_col) and time_col]

    time_values = df[time_col].reset_index(drop=True).values if time_col else None

    # Group rep columns by condition, keeping each rep as a separate Series
    result = {}
    for col in rep_cols:
        cond = _get_condition(col)
        result.setdefault(cond, []).append(
            df[col].reset_index(drop=True)  # drop time index — will assign N1 times later
        )

    return result, time_values


def collate_timeseries(directory, mode="normalised"):
    base = Path(directory)
    experiment_folders = sorted(base.glob("N*"))

    if mode == "normalised":
        ocr_sheet  = "OCR_cleaned"
        ecar_sheet = "ECAR_cleaned"
    else:
        ocr_sheet  = "OCR_raw_cleaned"
        ecar_sheet = "ECAR_raw_cleaned"

    # { condition: { n_label: [Series_rep1, Series_rep2, ...] } }
    ocr_data  = {}
    ecar_data = {}
    n_counter = {}      # tracks experiment number per condition independently
    ref_ocr_time  = None
    ref_ecar_time = None

    for folder in experiment_folders:
        prepared_files = sorted(folder.glob("*_prepared.xlsx"))
        if not prepared_files:
            print(f"  Skipping {folder.name}: no *_prepared.xlsx found")
            continue
        wb_path = prepared_files[0]
        print(f"  Loading: {wb_path}")

        try:
            ocr_conds, ocr_time = _timeseries_for_sheet(wb_path, ocr_sheet)
            if ref_ocr_time is None and ocr_time is not None:
                ref_ocr_time = ocr_time
        except Exception as e:
            print(f"    Could not read {ocr_sheet}: {e}")
            ocr_conds = {}

        try:
            ecar_conds, ecar_time = _timeseries_for_sheet(wb_path, ecar_sheet)
            if ref_ecar_time is None and ecar_time is not None:
                ref_ecar_time = ecar_time
        except Exception as e:
            print(f"    Could not read {ecar_sheet}: {e}")
            ecar_conds = {}

        # Assign _N label per condition (each condition gets its own counter)
        for cond, reps in ocr_conds.items():
            n_counter.setdefault(("ocr", cond), 0)
            n_counter[("ocr", cond)] += 1
            n = n_counter[("ocr", cond)]
            ocr_data.setdefault(cond, {})[f"_N{n}"] = reps

        for cond, reps in ecar_conds.items():
            n_counter.setdefault(("ecar", cond), 0)
            n_counter[("ecar", cond)] += 1
            n = n_counter[("ecar", cond)]
            ecar_data.setdefault(cond, {})[f"_N{n}"] = reps

    if not ocr_data and not ecar_data:
        print("No timeseries data found.")
        return

    out_dir = base / "collated" / mode / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    def build_export(data_dict, ref_time):
        """
        data_dict : { condition: { '_N1': [rep_series, ...], '_N2': [...] } }
        Produces a wide DataFrame with one column per rep, labelled Condition_N<n>,
        using ref_time as the index (Time column).
        """
        cols = []
        for cond, n_dict in data_dict.items():
            for n_label, reps in n_dict.items():
                for rep_series in reps:
                    cols.append(rep_series.rename(f"{cond}{n_label}"))
        export_df = pd.concat(cols, axis=1)
        if ref_time is not None:
            export_df.index = ref_time[:len(export_df)]
        export_df.index.name = "Time (minutes)"
        return export_df

    if ocr_data:
        ocr_export = build_export(ocr_data, ref_ocr_time)
        ocr_export.to_csv(out_dir / "OCR.csv")
        print(f"  Saved: {out_dir}/OCR.csv")

    if ecar_data:
        ecar_export = build_export(ecar_data, ref_ecar_time)
        ecar_export.to_csv(out_dir / "ECAR.csv")
        print(f"  Saved: {out_dir}/ECAR.csv")

    print(f"\n✓ Timeseries collation complete. CSVs written to: {out_dir}/")
