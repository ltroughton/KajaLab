"""
seahorse_analyze.py
====================
Reads OCR_corrected and ECAR_corrected sheets from the prepared workbook
produced by seahorse_prepare.py (after the user has manually cleared any
outlier wells in those sheets).

Outputs:
  - <experiment>_parameters.csv   — one row per replicate, all metrics
  - <experiment>_summary.csv      — mean ± SEM per treatment
  - seahorse_metric_plots/        — one PNG per metric
  - seahorse_metric_csv/          — one CSV per metric (treatments as columns)

Usage:
  python seahorse_analyze.py path/to/experiment_prepared.xlsx

Cycle windows:
  Edit BASELINE_WINDOW, OLIGOMYCIN_WINDOW, FCCP_WINDOW, ROTAA_WINDOW below.
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_treatment_name(name):
    name = str(name).replace("\n", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", name).strip()


def read_corrected_sheet(wb_path, sheet_name):
    """
    Read a corrected sheet (flat headers, Cycle + Time (minutes) + well columns).
    Wells the user deleted (cleared) will read as NaN columns — these are dropped.
    """
    df = pd.read_excel(wb_path, sheet_name=sheet_name, header=0)
    df.columns = [clean_treatment_name(c) for c in df.columns]

    # Drop columns that are entirely NaN (user-deleted outlier wells)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def window_vals(sub, col, w):
    return sub.loc[sub["Cycle"].between(w[0], w[1]), col]


def calculate_ocr_metrics(df, BASELINE_WINDOW, OLIGOMYCIN_WINDOW, FCCP_WINDOW, ROTAA_WINDOW):
    results = []
    rep_cols = [c for c in df.columns if c not in ("Cycle", "Time (minutes)")]

    for col in rep_cols:
        sub = df[["Cycle", col]].dropna()

        basal_ocr  = window_vals(sub, col, BASELINE_WINDOW).mean()
        olig_ocr   = window_vals(sub, col, OLIGOMYCIN_WINDOW).mean()
        fccp_ocr   = window_vals(sub, col, FCCP_WINDOW).max()   # peak, not mean
        rotaa_ocr  = window_vals(sub, col, ROTAA_WINDOW).mean()

        non_mito      = rotaa_ocr
        basal_resp    = basal_ocr  - non_mito
        atp_linked    = basal_ocr  - olig_ocr
        proton_leak   = olig_ocr   - non_mito
        maximal_resp  = fccp_ocr   - non_mito
        spare_cap     = maximal_resp - basal_resp
        coupling_eff  = atp_linked / basal_resp  if basal_resp  != 0 else np.nan
        rcr_like      = maximal_resp / proton_leak if proton_leak != 0 else np.nan

        treatment = clean_treatment_name(col.rsplit("_rep", 1)[0])

        results.append({
            "Treatment": treatment, "Replicate": col,
            "Basal_OCR": basal_ocr, "Oligomycin_OCR": olig_ocr,
            "FCCP_OCR": fccp_ocr,   "RotAA_OCR": rotaa_ocr,
            "Non_mitochondrial_respiration": non_mito,
            "Basal_respiration": basal_resp,
            "ATP_linked_respiration": atp_linked,
            "Proton_leak": proton_leak,
            "Maximal_respiration": maximal_resp,
            "Spare_respiratory_capacity": spare_cap,
            "Coupling_efficiency": coupling_eff,
            "RCR_like_metric": rcr_like,
        })

    return pd.DataFrame(results)


def calculate_ecar_metrics(df, BASELINE_WINDOW, OLIGOMYCIN_WINDOW):
    results = []
    rep_cols = [c for c in df.columns if c not in ("Cycle", "Time (minutes)")]

    for col in rep_cols:
        sub = df[["Cycle", col]].dropna()

        basal_ecar = window_vals(sub, col, BASELINE_WINDOW).mean()
        olig_ecar  = window_vals(sub, col, OLIGOMYCIN_WINDOW).mean()
        comp_glyc  = olig_ecar - basal_ecar

        treatment = clean_treatment_name(col.rsplit("_rep", 1)[0])

        results.append({
            "Treatment": treatment, "Replicate": col,
            "Basal_ECAR": basal_ecar,
            "ECAR_after_oligomycin": olig_ecar,
            "Compensatory_glycolysis": comp_glyc,
        })

    return pd.DataFrame(results)


def get_treatment_order(df):
    order = []
    for col in df.columns:
        if col in ("Cycle", "Time (minutes)"):
            continue
        t = clean_treatment_name(col.rsplit("_rep", 1)[0])
        if t not in order:
            order.append(t)
    return order


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "Basal_OCR":                    ("Basal OCR",                    "OCR under baseline conditions prior to drug additions."),
    "Oligomycin_OCR":               ("Oligomycin OCR",               "Residual OCR after ATP synthase inhibition."),
    "FCCP_OCR":                     ("FCCP OCR",                     "Peak OCR under uncoupled conditions."),
    "RotAA_OCR":                    ("Rotenone/Antimycin A OCR",     "Residual OCR after complete ETC inhibition."),
    "Non_mitochondrial_respiration":("Non-mitochondrial respiration","OCR independent of the mitochondrial ETC."),
    "Basal_respiration":            ("Basal respiration",            "Mitochondrial OCR under basal conditions (Basal OCR − Non-mito)."),
    "ATP_linked_respiration":       ("ATP-linked respiration",       "OCR coupled to ATP production (Basal OCR − Oligomycin OCR)."),
    "Proton_leak":                  ("Proton leak",                  "OCR not coupled to ATP synthesis (Oligomycin OCR − Non-mito)."),
    "Maximal_respiration":          ("Maximal respiration",          "Maximum mitochondrial respiratory capacity (FCCP OCR − Non-mito)."),
    "Spare_respiratory_capacity":   ("Spare respiratory capacity",   "Reserve capacity (Maximal − Basal respiration)."),
    "Coupling_efficiency":          ("Coupling efficiency",          "Fraction of basal respiration used for ATP production."),
    "RCR_like_metric":              ("RCR-like metric",              "Maximal respiration / Proton leak."),
    "Basal_ECAR":                   ("Basal ECAR",                   "Extracellular acidification under baseline conditions."),
    "ECAR_after_oligomycin":        ("ECAR after oligomycin",        "ECAR following ATP synthase inhibition."),
    "Compensatory_glycolysis":      ("Compensatory glycolysis",      "Increase in ECAR after oligomycin."),
}

PLOT_METRICS = list(METRIC_LABELS.keys())


def plot_metric(params, metric, treatment_order, out_dir):
    summary = (
        params.groupby("Treatment")[metric]
        .agg(mean="mean", sem="sem")
        .reindex(treatment_order)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(max(6, len(treatment_order) * 0.7), 5))
    x = np.arange(len(summary))

    ax.bar(x, summary["mean"], yerr=summary["sem"], capsize=5,
           edgecolor="black", facecolor="none", linewidth=1.5)

    for i, treatment in enumerate(treatment_order):
        vals = params.loc[params["Treatment"] == treatment, metric].dropna()
        jitter = np.random.default_rng(42).normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=30, color="black", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(treatment_order, rotation=45, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric.replace("_", " "))

    title, subtitle = METRIC_LABELS[metric]
    ax.set_title(title, fontsize=13, pad=22)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color="#444444")

    plt.tight_layout()
    path = out_dir / f"{metric}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_metric_csvs(params, treatment_order, metrics, out_dir):
    for metric in metrics:
        export_df = pd.DataFrame()
        for treatment in treatment_order:
            vals = params.loc[params["Treatment"] == treatment, metric].reset_index(drop=True)
            export_df[treatment] = vals
        export_df.to_csv(out_dir / f"{metric}.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(input_path, baseline=(1,3), oligomycin=(4,6), fccp=(7,9), rotaa=(10,12)):

# ---------------------------------------------------------------------------
# Edit these windows to match your injection schedule (cycle numbers)
# ---------------------------------------------------------------------------
    BASELINE_WINDOW   = baseline
    OLIGOMYCIN_WINDOW = oligomycin
    FCCP_WINDOW       = fccp
    ROTAA_WINDOW      = rotaa

    OCR_SHEET  = "OCR_cleaned"
    ECAR_SHEET = "ECAR_cleaned"    
    input_path = Path(input_path)
    out_dir = input_path.parent
    stem = input_path.stem

    print(f"Reading: {input_path}")

    df_ocr  = read_corrected_sheet(input_path, OCR_SHEET)
    df_ecar = read_corrected_sheet(input_path, ECAR_SHEET)

    treatment_order = get_treatment_order(df_ocr)
    print(f"Treatments ({len(treatment_order)}): {treatment_order}")

    params_ocr  = calculate_ocr_metrics(df_ocr, BASELINE_WINDOW, OLIGOMYCIN_WINDOW, FCCP_WINDOW, ROTAA_WINDOW)
    params_ecar = calculate_ecar_metrics(df_ecar, BASELINE_WINDOW, OLIGOMYCIN_WINDOW)
    
    params = pd.merge(params_ocr, params_ecar, on=["Treatment", "Replicate"], how="outer")

    # Save per-replicate parameters
    params_path = out_dir / f"{stem}_parameters.csv"
    params.to_csv(params_path, index=False)
    print(f"Saved: {params_path}")

    # Save summary
    agg_dict = {m: ["mean", "sem"] for m in PLOT_METRICS if m in params.columns}
    summary = params.groupby("Treatment", as_index=False).agg(agg_dict).reindex(
        pd.MultiIndex.from_product([["Treatment"], [""]])
        if False else range(len(params.groupby("Treatment")))
    )
    summary = params.groupby("Treatment")[list(agg_dict.keys())].agg(["mean", "sem"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reindex(treatment_order)
    summary_path = out_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    # Plots
    plot_dir = out_dir / "seahorse_metric_plots"
    plot_dir.mkdir(exist_ok=True)
    for metric in PLOT_METRICS:
        if metric in params.columns:
            plot_metric(params, metric, treatment_order, plot_dir)
    print(f"Plots saved to: {plot_dir}/")

    # Per-metric CSVs
    csv_dir = out_dir / "seahorse_metric_csv"
    csv_dir.mkdir(exist_ok=True)
    export_metric_csvs(params, treatment_order, PLOT_METRICS, csv_dir)
    print(f"CSVs saved to: {csv_dir}/")

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python seahorse_analyze.py path/to/experiment_prepared.xlsx")
        sys.exit(1)
    analyze(sys.argv[1])
