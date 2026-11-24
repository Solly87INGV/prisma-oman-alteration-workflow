#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd


# =============================================================================
# USER SETTINGS - MODIFICA SOLO QUESTE RIGHE
# =============================================================================
INPUT_DIR = Path(r"path/to/EnMAP")
OUTPUT_DIR = Path(r"path/to/EnMAP_masked_overlap")

# Intervallo da mascherare (in nm)
OVERLAP_MIN_NM = 900.0
OVERLAP_MAX_NM = 1000.0

# Separatore dei file txt
SEP = "\t"
# =============================================================================


REQUIRED_COLUMNS = ["sample_id", "wavelength", "reflectance"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_spectrum(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(txt_path, sep=SEP, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{txt_path.name}: mancano le colonne obbligatorie {missing}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["reflectance"] = pd.to_numeric(df["reflectance"], errors="coerce")

    df = df.dropna(subset=["wavelength"]).copy()
    df = df.sort_values("wavelength").reset_index(drop=True)

    return df


def mask_overlap(df: pd.DataFrame, wl_min: float, wl_max: float) -> pd.DataFrame:
    out = df.copy()
    mask = out["wavelength"].between(wl_min, wl_max, inclusive="both")
    out.loc[mask, "reflectance"] = pd.NA
    return out


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nessun file .txt trovato in: {INPUT_DIR}")

    print(f"Trovati {len(txt_files)} file TXT.")
    print(f"Intervallo overlap da mascherare: {OVERLAP_MIN_NM:.1f}–{OVERLAP_MAX_NM:.1f} nm\n")

    for txt_file in txt_files:
        df = load_spectrum(txt_file)

        n_total = len(df)
        overlap_mask = df["wavelength"].between(OVERLAP_MIN_NM, OVERLAP_MAX_NM, inclusive="both")
        n_overlap = int(overlap_mask.sum())

        df_masked = mask_overlap(df, OVERLAP_MIN_NM, OVERLAP_MAX_NM)

        out_path = OUTPUT_DIR / txt_file.name
        df_masked.to_csv(out_path, sep=SEP, index=False, encoding="utf-8-sig")

        print(
            f"[OK] {txt_file.name} | "
            f"punti totali: {n_total} | "
            f"punti mascherati: {n_overlap} | "
            f"output -> {out_path.name}"
        )

    print("\nFatto.")
    print(f"Output salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()