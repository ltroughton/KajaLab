"""
seahorse_analyze.py
====================
Reads from a prepared workbook produced by seahorse_prepare.py.
Runs analysis on two sets of cleaned sheets:
  - OCR_cleaned / ECAR_cleaned         -> Hoechst-normalised data
  - OCR_raw_cleaned / ECAR_raw_cleaned -> unnormalised data (fallback)

Outputs (in subdirectories alongside the input file):
  normalised/
    <stem>_parameters.csv
    <stem>_summary.csv
    seahorse_metric_plots/
    seahorse_metric_csv/
  raw/
    <stem>_parameters.csv
    <stem>_summary.csv
    seahorse_metric_plots/
    seahorse_metric_csv/

Basal OCR and ECAR are taken from the second-to-last cycle of the baseline
window (iloc[-2]) to avoid the first-cycle equilibration artefact and to
provide one cycle of buffer from the oligomycin injection.
All other windows (oligomycin, FCCP, rotenone/AA) continue to use the
full-window mean (or max for FCCP) as these phases are pharmacologically
clamped and do not exhibit the same drift.
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def clean_treatment_name(name):
    name = str(name).replace("\n", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", name).strip()


def read_cleaned_sheet(wb_path, sheet_name):
    df = pd.read_excel(wb_path, sheet_name=sheet_name, header=0)
    df.columns = [clean_treatment_name(c) for c in df.columns]
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

        # Use second-to-last cycle of baseline window as the pre-injection
        # reference point, avoiding early equilibration and injection artefact.
        basal_ocr = window_vals(sub, col, BASELINE_WINDOW).iloc[-2]
        olig_ocr  = window_vals(sub, col, OLIGOMYCIN_WINDOW).mean()
        fccp_ocr  = window_vals(sub, col, FCCP_WINDOW).max()
        rotaa_ocr = window_vals(sub, col, ROTAA_WINDOW).mean()

        non_mito     = rotaa_ocr
        basal_resp   = basal_ocr - non_mito
        atp_linked   = basal_ocr - olig_ocr
        proton_leak  = olig_ocr  - non_mito
        maximal_resp = fccp_ocr  - non_mito
        spare_cap    = maximal_resp - basal_resp
        coupling_eff = atp_linked / basal_resp   if basal_resp  != 0 else np.nan
        rcr_like     = maximal_resp / proton_leak if proton_leak != 0 else np.nan

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

        # Use second-to-last cycle of baseline window as the pre-injection
        # reference point, avoiding early equilibration and injection artefact.
        basal_ecar = window_vals(sub, col, BASELINE_WINDOW).iloc[-2]
        olig_ecar  = window_vals(sub, col, OLIGOMYCIN_WINDOW).mean()

        treatment = clean_treatment_name(col.rsplit("_rep", 1)[0])
        results.append({
            "Treatment": treatment, "Replicate": col,
            "Basal_ECAR": basal_ecar,
            "ECAR_after_oligomycin": olig_ecar,
            "Compensatory_glycolysis": olig_ecar - basal_ecar,
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


METRIC_LABELS = {
    "Basal_OCR":                     ("Basal OCR",                     "OCR at second-to-last baseline cycle (pre-injection reference)."),
    "Oligomycin_OCR":                ("Oligomycin OCR",                "Residual OCR after ATP synthase inhibition."),
    "FCCP_OCR":                      ("FCCP OCR",                      "Peak OCR under uncoupled conditions."),
    "RotAA_OCR":                     ("Rotenone/Antimycin A OCR",      "Residual OCR after complete ETC inhibition."),
    "Non_mitochondrial_respiration": ("Non-mitochondrial respiration",  "OCR independent of the mitochondrial ETC."),
    "Basal_respiration":             ("Basal respiration",              "Mitochondrial OCR under basal conditions (Basal OCR − Non-mito)."),
    "ATP_linked_respiration":        ("ATP-linked respiration",         "OCR coupled to ATP production (Basal OCR − Oligomycin OCR)."),
    "Proton_leak":                   ("Proton leak",                    "OCR not coupled to ATP synthesis (Oligomycin OCR − Non-mito)."),
    "Maximal_respiration":           ("Maximal respiration",            "Maximum mitochondrial respiratory capacity (FCCP OCR − Non-mito)."),
    "Spare_respiratory_capacity":    ("Spare respiratory capacity",     "Reserve capacity (Maximal − Basal respiration)."),
    "Coupling_efficiency":           ("Coupling efficiency",            "Fraction of basal respiration used for ATP production."),
    "RCR_like_metric":               ("RCR-like metric",                "Maximal respiration / Proton leak."),
    "Basal_ECAR":                    ("Basal ECAR",                     "ECAR at second-to-last baseline cycle (pre-injection reference)."),
    "ECAR_after_oligomycin":         ("ECAR after oligomycin",          "ECAR following ATP synthase inhibition."),
    "Compensatory_glycolysis":       ("Compensatory glycolysis",        "Increase in ECAR after oligomycin."),
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
    plt.savefig(out_dir / f"{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_metric_csvs(params, treatment_order, metrics, out_dir):
    for metric in metrics:
        cols = []
        for treatment in treatment_order:
            vals = params.loc[params["Treatment"] == treatment, metric].reset_index(drop=True)
            cols.append(vals.rename(treatment))
        export_df = pd.concat(cols, axis=1)
        export_df.to_csv(out_dir / f"{metric}.csv", index=False)


def run_analysis(df_ocr, df_ecar, treatment_order, out_dir,
                 BASELINE_WINDOW, OLIGOMYCIN_WINDOW, FCCP_WINDOW, ROTAA_WINDOW):
    """Core analysis — shared by both normalised and raw pipelines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir.parent.name  # experiment name from parent folder

    params_ocr  = calculate_ocr_metrics(df_ocr,  BASELINE_WINDOW, OLIGOMYCIN_WINDOW, FCCP_WINDOW, ROTAA_WINDOW)
    params_ecar = calculate_ecar_metrics(df_ecar, BASELINE_WINDOW, OLIGOMYCIN_WINDOW)
    params = pd.merge(params_ocr, params_ecar, on=["Treatment", "Replicate"], how="outer")

    params.to_csv(out_dir / "parameters.csv", index=False)
    print(f"  Saved: {out_dir}/parameters.csv")

    agg_dict = {m: ["mean", "sem"] for m in PLOT_METRICS if m in params.columns}
    summary = params.groupby("Treatment")[list(agg_dict.keys())].agg(["mean", "sem"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reindex(treatment_order)
    summary.to_csv(out_dir / "summary.csv")
    print(f"  Saved: {out_dir}/summary.csv")

    plot_dir = out_dir / "seahorse_metric_plots"
    plot_dir.mkdir(exist_ok=True)
    for metric in PLOT_METRICS:
        if metric in params.columns:
            plot_metric(params, metric, treatment_order, plot_dir)
    print(f"  Plots: {plot_dir}/")

    csv_dir = out_dir / "seahorse_metric_csv"
    csv_dir.mkdir(exist_ok=True)
    export_metric_csvs(params, treatment_order, PLOT_METRICS, csv_dir)
    print(f"  CSVs:  {csv_dir}/")

    return params


def analyze(input_path, baseline=(1,3), oligomycin=(4,6), fccp=(7,9), rotaa=(10,12)):
    BASELINE_WINDOW   = baseline
    OLIGOMYCIN_WINDOW = oligomycin
    FCCP_WINDOW       = fccp
    ROTAA_WINDOW      = rotaa

    input_path = Path(input_path)
    stem       = input_path.stem
    base_dir   = input_path.parent

    print(f"Reading: {input_path}\n")

    # ── Normalised analysis ───────────────────────────────────────────────────
    print("── Normalised (Hoechst-corrected) ──")
    df_ocr  = read_cleaned_sheet(input_path, "OCR_cleaned")
    df_ecar = read_cleaned_sheet(input_path, "ECAR_cleaned")
    treatment_order = get_treatment_order(df_ocr)
    print(f"Treatments: {treatment_order}")

    run_analysis(
        df_ocr, df_ecar, treatment_order,
        out_dir=base_dir / "analysis/normalised",
        BASELINE_WINDOW=BASELINE_WINDOW, OLIGOMYCIN_WINDOW=OLIGOMYCIN_WINDOW,
        FCCP_WINDOW=FCCP_WINDOW, ROTAA_WINDOW=ROTAA_WINDOW,
    )

    # ── Raw analysis ──────────────────────────────────────────────────────────
    print("\n── Raw (unnormalised) ──")
    df_ocr_raw  = read_cleaned_sheet(input_path, "OCR_raw_cleaned")
    df_ecar_raw = read_cleaned_sheet(input_path, "ECAR_raw_cleaned")
    treatment_order_raw = get_treatment_order(df_ocr_raw)

    run_analysis(
        df_ocr_raw, df_ecar_raw, treatment_order_raw,
        out_dir=base_dir / "analysis/raw",
        BASELINE_WINDOW=BASELINE_WINDOW, OLIGOMYCIN_WINDOW=OLIGOMYCIN_WINDOW,
        FCCP_WINDOW=FCCP_WINDOW, ROTAA_WINDOW=ROTAA_WINDOW,
    )

    print("\n✓ Analysis complete.")
    print(f"  Normalised outputs: {base_dir}/analysis/normalised/")
    print(f"  Raw outputs:        {base_dir}/analysis/raw/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python seahorse_analyze.py path/to/experiment_prepared.xlsx")
        sys.exit(1)
    analyze(sys.argv[1])
