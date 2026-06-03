"""
dextran_analysis.py
====================
Image analysis module for Dextran Uptake / Nuclei Counting experiments.

Functions
---------
1. test_thresholding(file_path, stack_index=0)
        Compare four auto-threshold methods visually; returns the chosen threshold value.

2. test_nuclei_counting(file_path, threshold_method="otsu", threshold_value=None,
                         min_area=150, max_area=20000, stack_index=0)
        Run the full segmentation pipeline on one image with QC plots;
        returns the nuclei count.

3. run_analysis(parent_directory, threshold_method="otsu", threshold_value=None,
                min_area=150, max_area=20000,
                show_image=False, save_image=False)
        Loop over all sub-folders, segment every TIFF, pool by well and condition,
        save CSV / HDF5 results, and generate bar plots.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import (threshold_li, threshold_otsu, threshold_triangle,
                              threshold_yen)
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from tqdm import tqdm

# ── Internal helpers ──────────────────────────────────────────────────────────

_THRESHOLD_FUNCS = {
    "otsu":     threshold_otsu,
    "triangle": threshold_triangle,
    "yen":      threshold_yen,
    "li":       threshold_li,
}


def _resolve_threshold(img, method="otsu", value=None):
    """Return a numeric threshold.

    Priority:
      1. If *value* is given (not None), use it directly.
      2. Otherwise compute it with *method* (one of: otsu, triangle, yen, li).
    """
    if value is not None:
        return float(value)
    func = _THRESHOLD_FUNCS.get(method.lower())
    if func is None:
        raise ValueError(
            f"Unknown threshold method '{method}'. "
            f"Choose from: {list(_THRESHOLD_FUNCS)}"
        )
    return func(img)


def _segment_nuclei(nuclei_img, thresh_val, min_area, max_area):
    """Core segmentation pipeline. Returns (filled, processed_binary, filtered_regions)."""
    binary = nuclei_img > thresh_val
    filled = ndimage.binary_fill_holes(binary)

    distance = ndimage.distance_transform_edt(filled)
    coords = peak_local_max(distance, labels=filled, footprint=np.ones((40, 40)))
    local_maxi = np.zeros_like(distance, dtype=bool)
    if coords.size:
        local_maxi[tuple(coords.T)] = True
    markers = label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=filled)

    regions = regionprops(labels_ws)
    valid_labels = [r.label for r in regions if min_area <= r.area <= max_area]
    filtered_mask = np.isin(labels_ws, valid_labels)
    processed_binary = (filtered_mask.astype(np.uint8) * 255)
    filtered_regions = [r for r in regions if r.label in valid_labels]

    return filled, processed_binary, filtered_regions


def _extract_well(filename):
    """Extract well ID (e.g. 'A1', 'H12') from a filename. Returns None if not found."""
    match = re.search(r'_([A-H][0-9]{1,2})_', filename)
    return match.group(1) if match else None


# ── Public API ────────────────────────────────────────────────────────────────

def test_thresholding(file_path, stack_index=0):
    """Compare four auto-threshold methods on a single image.

    Displays a side-by-side figure: raw image + one binary mask per method.

    Parameters
    ----------
    file_path   : str   Path to a multi-slice TIFF file.
    stack_index : int   Which slice to use as the nuclei channel (default 0).

    Returns
    -------
    dict
        ``{"Otsu": val, "Triangle": val, "Yen": val, "Li": val}``
        so you can pass the value you like directly into the next functions.

    Example
    -------
    thresholds = test_thresholding("path/to/image.tif")
    chosen_value = thresholds["Otsu"]          # then pass to test_nuclei_counting
    """
    img_stack = tiff.imread(file_path)
    nuclei_img = img_stack[stack_index]

    methods = {
        "Otsu":     threshold_otsu,
        "Triangle": threshold_triangle,
        "Yen":      threshold_yen,
        "Li":       threshold_li,
    }

    results = {}
    binaries = {}
    for name, func in methods.items():
        val = func(nuclei_img)
        results[name] = val
        binaries[name] = nuclei_img > val
        print(f"  {name:10s} threshold value: {val:.1f}")

    fig, axes = plt.subplots(1, len(binaries) + 1, figsize=(20, 6))
    axes[0].imshow(nuclei_img, cmap='gray',
                   vmin=np.percentile(nuclei_img, 1),
                   vmax=np.percentile(nuclei_img, 99))
    axes[0].set_title("Raw (16-bit)")
    axes[0].axis('off')

    for ax, (name, binary) in zip(axes[1:], binaries.items()):
        ax.imshow(binary, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nReturn value: dict with threshold values for each method.")
    print("Pass your chosen value to test_nuclei_counting() or run_analysis() via 'threshold_value='.")
    return results


def test_nuclei_counting(file_path,
                         threshold_method="otsu",
                         threshold_value=None,
                         min_area=150,
                         max_area=20000,
                         stack_index=0):
    """Run the full segmentation pipeline on a single image with QC plots.

    Parameters
    ----------
    file_path          : str    Path to a multi-slice TIFF file.
    threshold_method   : str    Auto-threshold method to use if threshold_value
                                is None ('otsu', 'triangle', 'yen', 'li').
    threshold_value    : float  Fixed threshold value. If provided, overrides
                                threshold_method (use the value from
                                test_thresholding()).
    min_area           : int    Minimum nucleus area in pixels (default 150).
    max_area           : int    Maximum nucleus area in pixels (default 20000).
    stack_index        : int    Nuclei channel slice index (default 0).

    Returns
    -------
    int   Number of nuclei detected.

    Example
    -------
    # Using auto-threshold:
    count = test_nuclei_counting("path/to/image.tif")

    # Using value picked from test_thresholding():
    thresholds = test_thresholding("path/to/image.tif")
    count = test_nuclei_counting("path/to/image.tif",
                                  threshold_value=thresholds["Otsu"])
    """
    img_stack   = tiff.imread(file_path)
    nuclei_img  = img_stack[stack_index]

    thresh_val = _resolve_threshold(nuclei_img, threshold_method, threshold_value)
    print(f"Using threshold value: {thresh_val:.1f}")

    filled, processed_binary, filtered_regions = _segment_nuclei(
        nuclei_img, thresh_val, min_area, max_area
    )

    # Build overlay
    overlay = cv2.cvtColor(processed_binary, cv2.COLOR_GRAY2BGR)
    for i, region in enumerate(filtered_regions):
        y, x = region.centroid
        cv2.putText(overlay, str(i + 1), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(nuclei_img, cmap='gray',
                   vmin=np.percentile(nuclei_img, 1),
                   vmax=np.percentile(nuclei_img, 99))
    axes[0].set_title("Raw (16-bit)")

    axes[1].imshow(filled, cmap='gray')
    label_str = (f"Fixed threshold ({thresh_val:.0f})"
                 if threshold_value is not None
                 else f"{threshold_method.capitalize()} Threshold")
    axes[1].set_title(label_str)

    axes[2].imshow(processed_binary, cmap='gray')
    axes[2].set_title("Processed (size-filtered)")

    axes[3].imshow(overlay)
    axes[3].set_title(f"Nuclei Count: {len(filtered_regions)}")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\nDetected nuclei: {len(filtered_regions)}")
    return len(filtered_regions)


def run_analysis(parent_directory,
                 threshold_method="otsu",
                 threshold_value=None,
                 min_area=150,
                 max_area=20000,
                 show_image=False,
                 save_image=False):
    """Loop over all sub-folders, segment every TIFF, and save results.

    For each sub-folder that contains TIFFs the function creates a
    ``<subfolder>_analyses/`` directory containing:
      - ``Raw_data.csv / .h5``     per-image metrics
      - ``FOV_pooled.csv / .h5``   per-well medians (images with > 300 nuclei excluded)
      - ``Final_data.csv / .h5``   per-condition medians (requires an
                                   ``*_Experiment_Plan.xlsx`` with a
                                   ``Table_view`` sheet having ``Well ID``
                                   and ``Condition`` columns)
      - ``*_plot.png``             bar plots per condition

    Parameters
    ----------
    parent_directory   : str    Top-level folder containing experiment sub-folders.
    threshold_method   : str    Auto-threshold method ('otsu', 'triangle', 'yen', 'li').
    threshold_value    : float  Fixed threshold value. Overrides threshold_method
                                when provided (pass thresholds["Otsu"] from
                                test_thresholding()).
    min_area           : int    Minimum nucleus area in pixels (default 150).
    max_area           : int    Maximum nucleus area in pixels (default 20000).
    show_image         : bool   Display QC figures interactively (default False).
    save_image         : bool   Save QC figures as PNG (default False).

    Returns
    -------
    None

    Example
    -------
    # Default (Otsu auto-threshold):
    run_analysis("/Volumes/Kaja/Emily/Flebogamma/")

    # Pass the threshold picked from test_thresholding():
    thresholds = test_thresholding("path/to/image.tif")
    run_analysis("/Volumes/Kaja/Emily/Flebogamma/",
                 threshold_value=thresholds["Otsu"])
    """
    sns.set(style="whitegrid")

    cols_to_pool = [
        "Nuclei count",
        "Texas red mean intensity",
        "Normalized mean intensity",
        "Texas red sum intensity",
        "Normalized sum intensity",
    ]

    for subfolder_path, _dirnames, filenames in os.walk(parent_directory):
        # Skip the top-level parent
        if os.path.abspath(subfolder_path) == os.path.abspath(parent_directory):
            continue

        subfolder = os.path.basename(subfolder_path)

        if subfolder.endswith("_analyses"):
            print(f"Skipping analyses folder: {subfolder_path}")
            continue

        print(f"\n=== Processing subfolder: {subfolder_path} ===")

        image_files = [
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith((".tif", ".tiff"))
            and not f.startswith("._")
            and os.path.isfile(os.path.join(subfolder_path, f))
        ]
        image_files.sort()
        total_files = len(image_files)
        print(f"Found {total_files} image files after filtering.\n")

        if total_files == 0:
            print(f"No TIFF images found in {subfolder_path}, skipping.")
            continue

        save_directory = os.path.join(subfolder_path, f"{subfolder}_analyses")
        os.makedirs(save_directory, exist_ok=True)

        Data = pd.DataFrame(columns=[
            "File name", "Well", "Nuclei count",
            "Texas red mean intensity", "Normalized mean intensity",
            "Texas red sum intensity", "Normalized sum intensity",
        ])
        skipped = []

        for filename in tqdm(image_files, desc=f"Processing {subfolder}"):
            filepath = os.path.join(subfolder_path, filename)

            # Signature check
            try:
                with open(filepath, "rb") as fh:
                    sig = fh.read(4)
                if sig not in (b"II*\x00", b"MM\x00*"):
                    skipped.append((filename, "Not a TIFF signature"))
                    continue
            except Exception as e:
                skipped.append((filename, f"Signature read error: {e}"))
                continue

            # Read
            try:
                img_stack = tiff.imread(filepath)
            except Exception as e:
                skipped.append((filename, f"tifffile read error: {e}"))
                continue

            if getattr(img_stack, "ndim", 0) < 3 or img_stack.shape[0] != 2:
                skipped.append((filename, f"Unexpected stack shape: {getattr(img_stack, 'shape', None)}"))
                continue

            nuclei_img = img_stack[0]
            cyto_img   = img_stack[1]

            # Threshold
            thresh_val = _resolve_threshold(nuclei_img, threshold_method, threshold_value)

            # Segmentation
            filled, processed_binary, filtered_regions = _segment_nuclei(
                nuclei_img, thresh_val, min_area, max_area
            )
            nuclei_count = len(filtered_regions)

            # Intensity metrics
            cyto_mean = float(np.mean(cyto_img))
            cyto_sum  = float(np.sum(cyto_img))
            norm_mean = cyto_mean / nuclei_count if nuclei_count > 0 else np.nan
            norm_sum  = cyto_sum  / nuclei_count if nuclei_count > 0 else np.nan

            Data.loc[len(Data)] = [
                filename, _extract_well(filename),
                nuclei_count, cyto_mean, norm_mean, cyto_sum, norm_sum,
            ]

            # QC figure
            overlay = cv2.cvtColor(processed_binary, cv2.COLOR_GRAY2BGR)
            for i, region in enumerate(filtered_regions):
                y, x = region.centroid
                cv2.putText(overlay, str(i + 1), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(nuclei_img, cmap='gray',
                           vmin=np.percentile(nuclei_img, 1),
                           vmax=np.percentile(nuclei_img, 99))
            axes[0].set_title("Raw Nuclei (16-bit)")
            axes[1].imshow(filled, cmap='gray')
            axes[1].set_title("Threshold")
            axes[2].imshow(processed_binary, cmap='gray')
            axes[2].set_title("Processed Mask (size-filtered)")
            axes[3].imshow(overlay)
            axes[3].set_title(f"Nuclei Count: {nuclei_count}")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()

            if show_image:
                plt.show()
            if save_image:
                save_path = os.path.join(
                    save_directory,
                    filename.replace(".tif", "_validation.png")
                             .replace(".tiff", "_validation.png")
                )
                fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        # ── Per-subfolder reporting ──────────────────────────────────────────
        print(f"Finished processing images in: {subfolder_path}")
        if skipped:
            print(f"Skipped {len(skipped)} files:")
            for name, reason in skipped[:10]:
                print(f"  - {name}: {reason}")
            if len(skipped) > 10:
                print("  ...")

        # Save raw data
        Data.to_csv(os.path.join(save_directory, "Raw_data.csv"), index=False)
        Data.to_hdf(os.path.join(save_directory, "Raw_data.h5"), key="data", mode="w")

        # ── FOV pooling ──────────────────────────────────────────────────────
        initial_rows = len(Data)
        Data_pool = Data[Data["Nuclei count"] <= 300].copy()
        print(f"Excluded {initial_rows - len(Data_pool)} images with Nuclei count > 300")

        if Data_pool.empty or Data_pool["Well"].isnull().all():
            print(f"No valid wells after nuclei filter; writing empty FOV_pooled.")
            Data_Median_FOV = pd.DataFrame(columns=["Well"] + cols_to_pool)
            Data_Median_FOV.to_csv(os.path.join(save_directory, "FOV_pooled.csv"), index=False)
            Data_Median_FOV.to_hdf(os.path.join(save_directory, "FOV_pooled.h5"), key="data", mode="w")
            continue

        if Data_pool["Well"].isnull().any():
            print("Warning: some filenames could not be parsed for well information.")

        Data_Median_FOV = Data_pool.groupby("Well")[cols_to_pool].median().reset_index()
        Data_Median_FOV.to_csv(os.path.join(save_directory, "FOV_pooled.csv"), index=False)
        Data_Median_FOV.to_hdf(os.path.join(save_directory, "FOV_pooled.h5"), key="data", mode="w")
        print(f"Saved FOV_pooled for '{subfolder_path}'")

        try:
            display(Data_Median_FOV.head(12))  # noqa: F821  (Jupyter built-in)
        except NameError:
            print(Data_Median_FOV.head(12))

        # ── Condition-level pooling ──────────────────────────────────────────
        plan_files = [
            f for f in os.listdir(subfolder_path)
            if f.endswith("_Experiment_Plan.xlsx")
            and not f.startswith("~$")
            and os.path.isfile(os.path.join(subfolder_path, f))
        ]

        if not plan_files:
            print(f"No '*_Experiment_Plan.xlsx' found; skipping condition-level pooling.")
            continue
        if len(plan_files) > 1:
            print(f"Warning: multiple plan files found; using '{plan_files[0]}'.")

        plan_path = os.path.join(subfolder_path, plan_files[0])
        try:
            plan_df = pd.read_excel(plan_path, sheet_name="Table_view")
        except Exception as e:
            print(f"Error reading experiment plan: {e}")
            continue

        plan_df.columns = [str(c).strip() for c in plan_df.columns]
        required_cols = {"Well ID", "Condition"}
        if not required_cols.issubset(set(plan_df.columns)):
            print(f"Experiment plan missing required columns {required_cols}; found {list(plan_df.columns)}.")
            continue

        well_to_condition = dict(
            zip(plan_df["Well ID"].astype(str).str.strip(),
                plan_df["Condition"].astype(str).str.strip())
        )

        # Annotate raw and FOV tables with Condition and overwrite
        Data["Condition"]           = Data["Well"].map(well_to_condition)
        Data_Median_FOV["Condition"] = Data_Median_FOV["Well"].map(well_to_condition)

        Data.to_csv(os.path.join(save_directory, "Raw_data.csv"), index=False)
        Data.to_hdf(os.path.join(save_directory, "Raw_data.h5"), key="data", mode="w")
        Data_Median_FOV.to_csv(os.path.join(save_directory, "FOV_pooled.csv"), index=False)
        Data_Median_FOV.to_hdf(os.path.join(save_directory, "FOV_pooled.h5"), key="data", mode="w")

        Data_Median_FOV_cond = Data_Median_FOV.dropna(subset=["Condition"]).copy()
        if Data_Median_FOV_cond.empty:
            print("No wells matched the experiment plan; skipping condition-level summary.")
            continue

        Data_Median_Conditions = (
            Data_Median_FOV_cond.groupby("Condition")[cols_to_pool].median().reset_index()
        )
        Data_Median_Conditions.to_csv(os.path.join(save_directory, "Final_data.csv"), index=False)
        Data_Median_Conditions.to_hdf(os.path.join(save_directory, "Final_data.h5"), key="data", mode="w")
        print(f"Saved Final_data for '{subfolder_path}'")

        try:
            display(Data_Median_Conditions.head(12))  # noqa: F821
        except NameError:
            print(Data_Median_Conditions.head(12))

        # ── Bar plots ────────────────────────────────────────────────────────
        try:
            df_plot = pd.read_csv(os.path.join(save_directory, "Final_data.csv"))
        except Exception as e:
            print(f"Error re-loading Final_data.csv for plotting: {e}")
            continue

        for var in cols_to_pool:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_plot, x="Condition", y=var)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"{var} per Condition")
            plt.tight_layout()
            plot_path = os.path.join(
                save_directory,
                f"{var.replace(' ', '_')}_plot.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved plot: {plot_path}")

    print("\nAll subfolders processed.")
