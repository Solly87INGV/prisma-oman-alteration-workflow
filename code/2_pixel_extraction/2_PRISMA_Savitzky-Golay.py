#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply Savitzky-Golay smoothing to PRISMA spectra stored as TXT/CSV files.

CONFIG-MODE VERSION
-------------------
You do NOT need command-line arguments.

Just edit the CONFIGURATION BLOCK below:

1) Set OUTPUT_DIR
2) Add your spectra paths in INPUT_SPECTRUM_FILES
3) Optionally add folders in INPUT_SPECTRUM_DIRS
4) Run the script from VS Code / PyCharm / terminal

Input structure supported:
- preferred structure:
    sample_id    wavelength    reflectance

- also supported:
    wavelength   reflectance

Output structure:
- TXT tab-delimited file with the same simple structure:

    sample_id    wavelength    reflectance

where "reflectance" is the Savitzky-Golay smoothed reflectance.

For each input file, the script writes:
    <original_name>_savgol.txt

Optional plot:
    <original_name>_savgol_plot.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# CONFIGURATION BLOCK - EDIT ONLY THIS SECTION FOR NORMAL USE
# ============================================================

# Output folder where filtered spectra and plots will be saved.
OUTPUT_DIR = r"path/to/ASD_spectra"

# List of individual spectrum files to process.
# Leave commented the ones you do not want to process.
INPUT_SPECTRUM_FILES = [
    r"path/to/SOIA_1_PRISMA.txt",
    r"path/to/SOIA_2_PRISMA.txt",
    r"path/to/SOIA_13_PRISMA.txt",
]

# Optional: process all .txt/.csv files inside these folders.
# Leave empty if you only want to use INPUT_SPECTRUM_FILES.
INPUT_SPECTRUM_DIRS = [
    # r"path/to/spectra",
]

# Savitzky-Golay parameters.
# For PRISMA spectra, a first conservative test can be:
# window = 7, polyorder = 2
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2

# Save comparison plot original vs filtered.
SAVE_PLOTS = True

# Output file suffix.
OUTPUT_SUFFIX = "_savgol"

# Output numeric formatting.
# None keeps pandas default. Recommended for clean TXT:
FLOAT_FORMAT = "%.10f"

# If True, also saves an extended diagnostic file with original + smoothed reflectance.
# Main requested output remains the simple TXT:
# sample_id, wavelength, reflectance
SAVE_DIAGNOSTIC_FILE = False


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def detect_separator(file_path: Path) -> Optional[str]:
    """
    Guess the separator used by the input file.

    Supported:
    - tab
    - semicolon
    - comma
    - generic whitespace
    """
    sample = file_path.read_text(encoding="utf-8", errors="ignore")[:5000]

    if "\t" in sample:
        return "\t"
    if ";" in sample:
        return ";"
    if "," in sample:
        return ","

    return r"\s+"


def read_spectrum(file_path: Path) -> pd.DataFrame:
    """
    Read a spectrum file into a pandas DataFrame.

    The function supports .txt and .csv files with common separators.
    Comment lines starting with # are ignored.
    """
    sep = detect_separator(file_path)

    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            engine="python",
            comment="#"
        )
    except Exception as e:
        raise ValueError(f"Unable to read file '{file_path.name}': {e}") from e

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if df.empty:
        raise ValueError(f"File '{file_path.name}' is empty after parsing.")

    return df


def is_numeric_series(series: pd.Series) -> bool:
    """Return True if a pandas Series can be converted to numeric."""
    try:
        pd.to_numeric(series, errors="raise")
        return True
    except Exception:
        return False


def normalize_column_name(name: object) -> str:
    """Normalize column names for robust matching."""
    return str(name).strip().lower().replace(" ", "_")


def find_columns(df: pd.DataFrame) -> Tuple[Optional[str], str, str]:
    """
    Detect sample_id, wavelength, and reflectance columns.

    Returns:
        sample_id_col, wavelength_col, reflectance_col

    sample_id_col can be None if no sample column exists.
    """
    normalized = {normalize_column_name(col): col for col in df.columns}

    sample_id_col = None
    wavelength_col = None
    reflectance_col = None

    sample_candidates = [
        "sample_id", "sample", "sampleid", "id", "name", "spectrum_id"
    ]

    wavelength_candidates = [
        "wavelength", "wl", "lambda", "nm", "micron", "microns", "um", "µm"
    ]

    reflectance_candidates = [
        "reflectance", "refl", "ref", "value", "intensity", "spectrum", "y"
    ]

    for key, original in normalized.items():
        if any(candidate == key or candidate in key for candidate in sample_candidates):
            sample_id_col = original
            break

    for key, original in normalized.items():
        if any(candidate == key or candidate in key for candidate in wavelength_candidates):
            wavelength_col = original
            break

    for key, original in normalized.items():
        if any(candidate == key or candidate in key for candidate in reflectance_candidates):
            reflectance_col = original
            break

    # Fallback for files with no clean headers:
    # use first two numeric columns as wavelength and reflectance.
    numeric_cols = [col for col in df.columns if is_numeric_series(df[col])]

    if wavelength_col is None or reflectance_col is None:
        if len(numeric_cols) < 2:
            raise ValueError(
                "Could not detect wavelength and reflectance columns. "
                "Expected columns like 'wavelength' and 'reflectance', "
                "or at least two numeric columns."
            )

        wavelength_col = wavelength_col or numeric_cols[0]
        reflectance_col = reflectance_col or numeric_cols[1]

    if wavelength_col == reflectance_col:
        raise ValueError(
            "Wavelength and reflectance columns were detected as the same column. "
            "Please check the input file header."
        )

    return sample_id_col, wavelength_col, reflectance_col


def prepare_savgol_window(n_points: int, requested_window: int, polyorder: int) -> int:
    """
    Make Savitzky-Golay window valid.

    Requirements:
    - odd integer
    - <= number of points
    - > polyorder
    """
    if requested_window < 3:
        raise ValueError("SAVGOL_WINDOW must be at least 3.")

    if polyorder < 1:
        raise ValueError("SAVGOL_POLYORDER must be at least 1.")

    window = requested_window

    # Window must be odd.
    if window % 2 == 0:
        window += 1

    # Window cannot exceed number of points.
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1

    # Window must be greater than polyorder.
    min_valid = polyorder + 2
    if min_valid % 2 == 0:
        min_valid += 1

    if window < min_valid:
        window = min_valid

    if window > n_points:
        raise ValueError(
            f"Not enough valid points ({n_points}) for polyorder={polyorder}. "
            f"Need at least {min_valid} points."
        )

    return window


def smooth_spectrum(reflectance: np.ndarray, window: int, polyorder: int) -> Tuple[np.ndarray, int]:
    """
    Apply Savitzky-Golay smoothing.

    Returns:
        smoothed_reflectance, valid_window_used
    """
    valid_window = prepare_savgol_window(
        n_points=len(reflectance),
        requested_window=window,
        polyorder=polyorder
    )

    smoothed = savgol_filter(
        reflectance,
        window_length=valid_window,
        polyorder=polyorder,
        mode="interp"
    )

    return smoothed, valid_window


def save_plot(
    wavelength: np.ndarray,
    original: np.ndarray,
    smoothed: np.ndarray,
    title: str,
    out_path: Path
) -> None:
    """Save original vs smoothed comparison plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength, original, label="Original")
    plt.plot(wavelength, smoothed, label="Savitzky-Golay")
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_sample_id_series(
    work_df: pd.DataFrame,
    sample_id_col: Optional[str],
    fallback_sample_id: str
) -> pd.Series:
    """
    Return sample_id values for output.

    If the input has a sample_id column, preserve it.
    If not, use the input filename stem for all rows.
    """
    if sample_id_col is not None and sample_id_col in work_df.columns:
        sample_ids = work_df[sample_id_col].astype(str)
        sample_ids = sample_ids.replace({"nan": fallback_sample_id, "None": fallback_sample_id})
        return sample_ids

    return pd.Series([fallback_sample_id] * len(work_df), index=work_df.index)


def process_file(
    file_path: Path,
    output_dir: Path,
    window: int,
    polyorder: int,
    save_plots: bool
) -> None:
    """
    Read, smooth, and export one spectrum file.

    Main output:
        sample_id    wavelength    reflectance

    where reflectance is the smoothed reflectance.
    """
    df = read_spectrum(file_path)
    sample_id_col, wavelength_col, reflectance_col = find_columns(df)

    work_df = df.copy()

    work_df[wavelength_col] = pd.to_numeric(work_df[wavelength_col], errors="coerce")
    work_df[reflectance_col] = pd.to_numeric(work_df[reflectance_col], errors="coerce")
    work_df = work_df.dropna(subset=[wavelength_col, reflectance_col])

    if work_df.empty:
        raise ValueError(f"No valid numeric data found in '{file_path.name}'.")

    # Sort by wavelength to avoid problems if input rows are unordered.
    work_df = work_df.sort_values(by=wavelength_col).reset_index(drop=True)

    sample_ids = build_sample_id_series(
        work_df=work_df,
        sample_id_col=sample_id_col,
        fallback_sample_id=file_path.stem
    )

    wavelength = work_df[wavelength_col].to_numpy(dtype=float)
    reflectance = work_df[reflectance_col].to_numpy(dtype=float)

    smoothed, valid_window = smooth_spectrum(
        reflectance=reflectance,
        window=window,
        polyorder=polyorder
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Requested output format:
    # sample_id    wavelength    reflectance
    out_df = pd.DataFrame({
        "sample_id": sample_ids.to_numpy(),
        "wavelength": wavelength,
        "reflectance": smoothed
    })

    out_file = output_dir / f"{file_path.stem}{OUTPUT_SUFFIX}.txt"

    out_df.to_csv(
        out_file,
        sep="\t",
        index=False,
        float_format=FLOAT_FORMAT
    )

    if SAVE_DIAGNOSTIC_FILE:
        diagnostic_df = pd.DataFrame({
            "sample_id": sample_ids.to_numpy(),
            "wavelength": wavelength,
            "reflectance_original": reflectance,
            "reflectance_savgol": smoothed
        })

        diagnostic_file = output_dir / f"{file_path.stem}{OUTPUT_SUFFIX}_diagnostic.txt"

        diagnostic_df.to_csv(
            diagnostic_file,
            sep="\t",
            index=False,
            float_format=FLOAT_FORMAT
        )

    if save_plots:
        plot_file = output_dir / f"{file_path.stem}{OUTPUT_SUFFIX}_plot.png"
        save_plot(
            wavelength=wavelength,
            original=reflectance,
            smoothed=smoothed,
            title=f"{file_path.stem} | SG window={valid_window}, polyorder={polyorder}",
            out_path=plot_file
        )

    print(f"[OK] {file_path.name} -> {out_file.name} | window used = {valid_window}")


def collect_input_files_from_list(paths: List[str]) -> List[Path]:
    """Collect explicit input files from INPUT_SPECTRUM_FILES."""
    files: List[Path] = []

    for raw_path in paths:
        if not raw_path:
            continue

        path = Path(raw_path)

        if path.is_file():
            files.append(path)
        else:
            print(f"[WARNING] File not found, skipped: {path}")

    return files


def collect_input_files_from_dirs(paths: List[str]) -> List[Path]:
    """Collect .txt and .csv files from INPUT_SPECTRUM_DIRS."""
    files: List[Path] = []

    for raw_path in paths:
        if not raw_path:
            continue

        folder = Path(raw_path)

        if not folder.is_dir():
            print(f"[WARNING] Folder not found, skipped: {folder}")
            continue

        for ext in ("*.txt", "*.csv"):
            files.extend(sorted(folder.glob(ext)))

    return files


def collect_all_input_files() -> List[Path]:
    """
    Collect input files from both:
    - INPUT_SPECTRUM_FILES
    - INPUT_SPECTRUM_DIRS

    Duplicates are removed while preserving order.
    """
    files = []
    files.extend(collect_input_files_from_list(INPUT_SPECTRUM_FILES))
    files.extend(collect_input_files_from_dirs(INPUT_SPECTRUM_DIRS))

    unique_files: List[Path] = []
    seen = set()

    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            unique_files.append(file_path)
            seen.add(resolved)

    return unique_files


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    files = collect_all_input_files()

    if not files:
        print("[ERROR] No input files found.")
        print("Check INPUT_SPECTRUM_FILES or INPUT_SPECTRUM_DIRS in the configuration block.")
        return

    print("Savitzky-Golay smoothing for PRISMA spectra")
    print("------------------------------------------------")
    print(f"Output folder: {output_dir}")
    print(f"Files found: {len(files)}")
    print(f"Requested window: {SAVGOL_WINDOW}")
    print(f"Polynomial order: {SAVGOL_POLYORDER}")
    print(f"TXT output format: sample_id | wavelength | reflectance")
    print("------------------------------------------------")

    failures = []

    for file_path in files:
        try:
            process_file(
                file_path=file_path,
                output_dir=output_dir,
                window=SAVGOL_WINDOW,
                polyorder=SAVGOL_POLYORDER,
                save_plots=SAVE_PLOTS
            )
        except Exception as e:
            failures.append((file_path.name, str(e)))
            print(f"[FAIL] {file_path.name}: {e}")

    print("\nDone.")

    if failures:
        print("\nFiles with errors:")
        for name, err in failures:
            print(f" - {name}: {err}")


if __name__ == "__main__":
    main()
