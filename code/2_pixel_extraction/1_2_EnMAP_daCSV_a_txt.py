#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert ENMAP spectra_long.csv into one TXT spectrum file per sample
with the same structure used for PRISMA / ASD / USGS comparison.

INPUT:
- spectra_long.csv produced by extract_enmap_spectra.py

OUTPUT:
- one TXT file per sample_id, e.g.:
  SOIA_1_ENMAP.txt
  SOIA_12_ENMAP.txt
  ...

Format:
sample_id    wavelength    reflectance
SOIA_1_ENMAP 418.416       0.1234
...
"""

import os
import sys
import traceback
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================
SPECTRA_LONG_CSV = r"path/to/spectra_long.csv"
OUTPUT_DIR = r"path/to/EnMAP"

# keep only this extraction mode
MODE_TO_EXPORT = "single_pixel"

# suffix added to sample_id in exported txt
OUTPUT_SUFFIX = "_ENMAP"

# if True, remove rows where reflectance is NaN
DROP_NAN_REFLECTANCE = True
# =============================================================================


REQUIRED_COLUMNS = ["sample_id", "mode", "wavelength", "reflectance"]


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    bad_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    out = str(name)
    for ch in bad_chars:
        out = out.replace(ch, "_")
    return out.strip()


def main():
    print("\n=== ENMAP TXT EXPORT FROM spectra_long.csv ===\n")

    if not os.path.exists(SPECTRA_LONG_CSV):
        raise FileNotFoundError(f"File non trovato:\n{SPECTRA_LONG_CSV}")

    ensure_output_dir(OUTPUT_DIR)

    print("Leggo spectra_long.csv ...")
    df = pd.read_csv(SPECTRA_LONG_CSV, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Mancano colonne obbligatorie nel CSV: {missing}\n"
            f"Colonne trovate: {list(df.columns)}"
        )

    # tieni solo la modalità desiderata
    df = df[df["mode"] == MODE_TO_EXPORT].copy()

    if df.empty:
        raise ValueError(
            f"Nessuna riga trovata con mode = '{MODE_TO_EXPORT}'."
        )

    # pulizia tipi
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["reflectance"] = pd.to_numeric(df["reflectance"], errors="coerce")

    if DROP_NAN_REFLECTANCE:
        df = df[df["reflectance"].notna()].copy()

    if df.empty:
        raise ValueError("Dopo la pulizia dei NaN non rimane nessuna riga da esportare.")

    exported = 0

    for sample_id, g in df.groupby("sample_id"):
        out_sample_id = f"{sample_id}{OUTPUT_SUFFIX}"

        out_df = g[["wavelength", "reflectance"]].copy()
        out_df = out_df.sort_values("wavelength").reset_index(drop=True)
        out_df.insert(0, "sample_id", out_sample_id)

        out_name = sanitize_filename(out_sample_id) + ".txt"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        out_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
        exported += 1
        print(f"[OK] {out_name}")

    print("\nFATTO.")
    print(f"File esportati: {exported}")
    print(f"Cartella output: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERRORE:")
        print(str(e))
        print("\nTraceback completo:\n")
        traceback.print_exc()
        sys.exit(1)