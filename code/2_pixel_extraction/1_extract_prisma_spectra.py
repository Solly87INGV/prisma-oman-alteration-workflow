#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRISMA spectra extraction from sample points
- Single pixel spectrum
- Mean 3x3 spectrum

USO:
1) Salva questo file come: extract_prisma_spectra.py
2) Modifica SOLO la sezione USER SETTINGS qui sotto
3) Lancialo in Visual Studio Code

OUTPUT:
- extraction_summary.csv
- spectra_long.csv
- spectra_single_pixel_wide.csv
- spectra_mean_3x3_wide.csv
- band_wavelength_lookup.csv
- run_info.txt
"""

import os
import re
import sys
import traceback
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer


# =============================================================================
# USER SETTINGS - MODIFICA SOLO QUESTE RIGHE
# =============================================================================
CUBE_PATH = r"path/to/VNIR_SWIR_latlon_219_ROIclip.tif"
HDR_PATH = r"path/to/VNIR_SWIR_latlon_219.hdr"
POINTS_CSV = r"path/to/soia_selected_samples.xlsx"
OUTPUT_DIR = r"path/to/output_selected_samples"

# CRS dei punti nel file tabellare
POINTS_CRS = "EPSG:4326"

# CRS del raster/cubo PRISMA
RASTER_CRS = "EPSG:32640"

# Opzionale: file CSV/XLSX con una colonna chiamata wavelength oppure wavelength_nm
# Se non lo hai, lascia None
WAVELENGTHS_CSV = None
# =============================================================================


REQUIRED_COLUMNS = ["sample_id", "longitude", "latitude"]
OPTIONAL_COLUMNS = ["asd_spot", "group", "note"]


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_table_robust(table_path: str) -> pd.DataFrame:
    ext = os.path.splitext(table_path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        print(f"Lettura file Excel: {table_path}")
        return pd.read_excel(table_path)

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(table_path, encoding=enc)
            print(f"CSV letto con encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(
        "Impossibile leggere il file tabellare con gli encoding provati.\n"
        f"File: {table_path}\n"
        f"Encoding provati: {encodings_to_try}\n"
        f"Ultimo errore: {last_error}"
    )


def load_points(table_path: str) -> pd.DataFrame:
    df = read_table_robust(table_path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Nel file dei punti mancano queste colonne obbligatorie: {missing}\n"
            f"Colonne trovate: {list(df.columns)}"
        )

    # aggiunge eventuali colonne opzionali mancanti come stringa vuota
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    if df["longitude"].isna().any() or df["latitude"].isna().any():
        bad = df[df["longitude"].isna() | df["latitude"].isna()]
        raise ValueError(
            "Coordinate non valide trovate nel file dei punti.\n"
            f"Righe problematiche:\n{bad.to_string(index=False)}"
        )

    return df


def load_wavelengths_from_table(table_path: str, band_count: int):
    df = read_table_robust(table_path)
    df.columns = [str(c).strip() for c in df.columns]

    possible_cols = ["wavelength", "wavelength_nm", "wl", "lambda_nm"]
    col = next((c for c in possible_cols if c in df.columns), None)

    if col is None:
        raise ValueError(
            f"Il file wavelengths deve avere una colonna tra: {possible_cols}\n"
            f"Colonne trovate: {list(df.columns)}"
        )

    vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()

    if len(vals) != band_count:
        raise ValueError(
            f"Numero wavelengths ({len(vals)}) diverso dal numero bande raster ({band_count})."
        )

    return [float(v) for v in vals], f"table:{table_path}"


def parse_envi_hdr_wavelengths(hdr_path: str, band_count: int):
    if not hdr_path or not os.path.exists(hdr_path):
        return None

    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    match = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    parts = [p.strip() for p in raw.replace("\n", " ").split(",")]

    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass

    if len(vals) != band_count:
        print(
            f"ATTENZIONE: HDR trovato ma numero wavelengths ({len(vals)}) "
            f"diverso da numero bande raster ({band_count})."
        )
        return None

    return vals


def parse_band_descriptions(dataset, band_count: int):
    if not dataset.descriptions:
        return None

    vals = []
    for i in range(band_count):
        desc = dataset.descriptions[i]
        if not desc:
            return None

        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(desc))
        if not nums:
            return None

        vals.append(float(nums[0]))

    if len(vals) != band_count:
        return None

    return vals


def load_wavelengths(dataset, cube_path: str, hdr_path: str, wavelengths_csv):
    band_count = dataset.count

    if hdr_path:
        wl = parse_envi_hdr_wavelengths(hdr_path, band_count)
        if wl is not None:
            print(f"Wavelengths lette da HDR esplicito: {hdr_path}")
            return wl, f"hdr:{hdr_path}"

    if wavelengths_csv:
        wl, src = load_wavelengths_from_table(wavelengths_csv, band_count)
        print(f"Wavelengths lette da tabella esplicita: {wavelengths_csv}")
        return wl, src

    hdr1 = cube_path + ".hdr"
    wl = parse_envi_hdr_wavelengths(hdr1, band_count)
    if wl is not None:
        print("Wavelengths lette da file HDR associato.")
        return wl, f"hdr:{hdr1}"

    base, _ = os.path.splitext(cube_path)
    hdr2 = base + ".hdr"
    wl = parse_envi_hdr_wavelengths(hdr2, band_count)
    if wl is not None:
        print("Wavelengths lette da file HDR con stesso basename.")
        return wl, f"hdr:{hdr2}"

    wl = parse_band_descriptions(dataset, band_count)
    if wl is not None:
        print("Wavelengths lette dalle band descriptions.")
        return wl, "band_descriptions"

    print("Wavelengths non trovate: uso indice banda come fallback.")
    return [float(i) for i in range(1, band_count + 1)], "band_index_fallback"


def format_wavelength_for_colname(wl: float) -> str:
    """
    Restituisce un nome pulito per colonna wide.
    Esempi:
    401.0 -> 401
    401.25 -> 401_25
    """
    if float(wl).is_integer():
        return str(int(round(wl)))
    s = f"{wl:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


def transform_points_safe(xs, ys, src_crs, dst_crs):
    """
    Trasforma i punti da src_crs a dst_crs usando pyproj.
    always_xy=True => input/output nel formato lon, lat / easting, northing
    """
    if str(src_crs).strip().upper() == str(dst_crs).strip().upper():
        return xs, ys, "same_crs"

    try:
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        tx, ty = transformer.transform(xs, ys)
        return list(tx), list(ty), "transformed"
    except Exception as e:
        raise ValueError(
            f"Errore nella trasformazione coordinate da {src_crs} a {dst_crs}: {e}"
        )


def inside_raster(ds, row, col):
    return 0 <= row < ds.height and 0 <= col < ds.width


def read_single_pixel(ds, row, col):
    arr = ds.read(window=Window(col, row, 1, 1)).astype(np.float64)
    return arr[:, 0, 0]


def read_mean_3x3(ds, row, col, nodata_value):
    row_start = max(0, row - 1)
    row_end = min(ds.height - 1, row + 1)
    col_start = max(0, col - 1)
    col_end = min(ds.width - 1, col + 1)

    h = row_end - row_start + 1
    w = col_end - col_start + 1

    cube = ds.read(window=Window(col_start, row_start, w, h)).astype(np.float64)

    if nodata_value is not None:
        invalid = np.any(cube == nodata_value, axis=0)
    else:
        invalid = np.zeros((cube.shape[1], cube.shape[2]), dtype=bool)

    invalid = invalid | np.any(~np.isfinite(cube), axis=0)

    valid_pixel_count = int(np.sum(~invalid))
    mean_spec = np.full(cube.shape[0], np.nan, dtype=np.float64)

    if valid_pixel_count == 0:
        return mean_spec, 0

    for b in range(cube.shape[0]):
        band = cube[b]
        vals = band[~invalid]
        if vals.size > 0:
            mean_spec[b] = np.mean(vals)

    return mean_spec, valid_pixel_count


def spectrum_ok(spec):
    return spec is not None and np.any(np.isfinite(spec))


def spectrum_to_wide_row(meta, spectrum, wavelengths, prefix="band"):
    out = meta.copy()
    for i, wl in enumerate(wavelengths):
        wl_label = format_wavelength_for_colname(wl)
        out[f"{prefix}_{wl_label}"] = spectrum[i]
    return out


def main():
    print("\n=== PRISMA SPECTRA EXTRACTION ===\n")

    for path_name, path_value in [
        ("CUBE_PATH", CUBE_PATH),
        ("POINTS_CSV", POINTS_CSV),
        ("OUTPUT_DIR", OUTPUT_DIR),
    ]:
        if not path_value or "PATH\\TO\\YOUR" in str(path_value):
            raise ValueError(f"Devi compilare correttamente {path_name} nello script.")

    if not os.path.exists(CUBE_PATH):
        raise FileNotFoundError(f"Cubo non trovato:\n{CUBE_PATH}")

    if HDR_PATH is not None and not os.path.exists(HDR_PATH):
        raise FileNotFoundError(f"HDR non trovato:\n{HDR_PATH}")

    if not os.path.exists(POINTS_CSV):
        raise FileNotFoundError(f"File punti non trovato:\n{POINTS_CSV}")

    if WAVELENGTHS_CSV is not None and not os.path.exists(WAVELENGTHS_CSV):
        raise FileNotFoundError(f"File wavelengths non trovato:\n{WAVELENGTHS_CSV}")

    ensure_output_dir(OUTPUT_DIR)

    print("Leggo i punti...")
    points_df = load_points(POINTS_CSV)
    print(f"Punti letti: {len(points_df)}")

    summary_rows = []
    long_rows = []
    single_wide_rows = []
    mean3_wide_rows = []

    print("Apro il cubo PRISMA...")
    with rasterio.open(CUBE_PATH) as ds:
        nodata_value = ds.nodata
        wavelengths, wavelengths_source = load_wavelengths(ds, CUBE_PATH, HDR_PATH, WAVELENGTHS_CSV)

        print(f"Bande raster: {ds.count}")
        print(f"Dimensioni raster: {ds.width} x {ds.height}")
        print(f"CRS raster letto dal file: {ds.crs}")
        print(f"CRS raster usato per la trasformazione: {RASTER_CRS}")
        print(f"NoData: {nodata_value}")
        print(f"Sorgente wavelengths: {wavelengths_source}")

        xs = points_df["longitude"].tolist()
        ys = points_df["latitude"].tolist()
        tx, ty, transform_mode = transform_points_safe(xs, ys, POINTS_CRS, RASTER_CRS)

        print(f"Modalità trasformazione coordinate: {transform_mode}")
        print("\nEstraggo gli spettri...\n")

        for idx, row in points_df.iterrows():
            sample_id = row["sample_id"]
            asd_spot = row["asd_spot"]
            group = row["group"]
            note = row["note"]

            x_proj = tx[idx]
            y_proj = ty[idx]

            try:
                pixel_row, pixel_col = ds.index(x_proj, y_proj)
            except Exception:
                pixel_row, pixel_col = -1, -1

            is_inside = inside_raster(ds, pixel_row, pixel_col)

            single_spec = np.full(ds.count, np.nan, dtype=np.float64)
            mean3_spec = np.full(ds.count, np.nan, dtype=np.float64)

            single_ok = False
            mean3_ok = False
            valid_3x3_pixels = 0
            qc_note = ""

            if is_inside:
                try:
                    single_spec = read_single_pixel(ds, pixel_row, pixel_col)
                    if nodata_value is not None:
                        single_spec = np.where(single_spec == nodata_value, np.nan, single_spec)
                    single_ok = spectrum_ok(single_spec)
                except Exception as e:
                    qc_note += f"single_error={str(e)};"

                try:
                    mean3_spec, valid_3x3_pixels = read_mean_3x3(
                        ds, pixel_row, pixel_col, nodata_value
                    )
                    mean3_ok = spectrum_ok(mean3_spec) and valid_3x3_pixels > 0
                except Exception as e:
                    qc_note += f"mean3x3_error={str(e)};"
            else:
                qc_note += "point_outside_raster;"

            meta = {
                "sample_id": sample_id,
                "asd_spot": asd_spot,
                "group": group,
                "note": note,
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "x_proj": x_proj,
                "y_proj": y_proj,
                "pixel_row": pixel_row,
                "pixel_col": pixel_col,
            }

            summary_rows.append({
                **meta,
                "inside_raster": is_inside,
                "single_pixel_ok": single_ok,
                "mean_3x3_ok": mean3_ok,
                "valid_3x3_pixels": valid_3x3_pixels,
                "nodata_value": nodata_value,
                "transform_mode": transform_mode,
                "wavelengths_source": wavelengths_source,
                "qc_note": qc_note.strip(";"),
            })

            for b in range(ds.count):
                long_rows.append({
                    **meta,
                    "mode": "single_pixel",
                    "band_index": b + 1,
                    "wavelength": wavelengths[b],
                    "reflectance": single_spec[b],
                })
                long_rows.append({
                    **meta,
                    "mode": "mean_3x3",
                    "band_index": b + 1,
                    "wavelength": wavelengths[b],
                    "reflectance": mean3_spec[b],
                })

            single_wide_rows.append(
                spectrum_to_wide_row(meta, single_spec, wavelengths, prefix="band")
            )
            mean3_wide_rows.append(
                spectrum_to_wide_row(meta, mean3_spec, wavelengths, prefix="band")
            )

            print(
                f"{sample_id}: inside={is_inside} | "
                f"single_ok={single_ok} | mean3_ok={mean3_ok} | "
                f"valid_3x3_pixels={valid_3x3_pixels}"
            )

    print("\nSalvo gli output...")

    summary_df = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)
    single_wide_df = pd.DataFrame(single_wide_rows)
    mean3_wide_df = pd.DataFrame(mean3_wide_rows)

    summary_path = os.path.join(OUTPUT_DIR, "extraction_summary.csv")
    long_path = os.path.join(OUTPUT_DIR, "spectra_long.csv")
    single_wide_path = os.path.join(OUTPUT_DIR, "spectra_single_pixel_wide.csv")
    mean3_wide_path = os.path.join(OUTPUT_DIR, "spectra_mean_3x3_wide.csv")
    lookup_path = os.path.join(OUTPUT_DIR, "band_wavelength_lookup.csv")
    info_path = os.path.join(OUTPUT_DIR, "run_info.txt")

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    single_wide_df.to_csv(single_wide_path, index=False, encoding="utf-8-sig")
    mean3_wide_df.to_csv(mean3_wide_path, index=False, encoding="utf-8-sig")

    lookup_df = pd.DataFrame({
        "band_index": list(range(1, len(wavelengths) + 1)),
        "wavelength": wavelengths,
        "wide_column_name": [f"band_{format_wavelength_for_colname(w)}" for w in wavelengths],
        "wavelengths_source": [wavelengths_source] * len(wavelengths),
    })
    lookup_df.to_csv(lookup_path, index=False, encoding="utf-8-sig")

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("PRISMA spectra extraction completed.\n")
        f.write(f"Cube: {CUBE_PATH}\n")
        f.write(f"HDR path: {HDR_PATH}\n")
        f.write(f"Points file: {POINTS_CSV}\n")
        f.write(f"Output dir: {OUTPUT_DIR}\n")
        f.write(f"Points CRS: {POINTS_CRS}\n")
        f.write(f"Raster CRS: {RASTER_CRS}\n")
        f.write(f"Wavelengths file: {WAVELENGTHS_CSV}\n")
        f.write(f"Wavelengths source: {wavelengths_source}\n")
        f.write(f"Transform mode: {transform_mode}\n")

    print("\nFATTO.")
    print(f"\nOutput salvati in:\n{OUTPUT_DIR}")
    print("\nFile creati:")
    print(f"- {os.path.basename(summary_path)}")
    print(f"- {os.path.basename(long_path)}")
    print(f"- {os.path.basename(single_wide_path)}")
    print(f"- {os.path.basename(mean3_wide_path)}")
    print(f"- {os.path.basename(lookup_path)}")
    print(f"- {os.path.basename(info_path)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERRORE:")
        print(str(e))
        print("\nTraceback completo:\n")
        traceback.print_exc()
        sys.exit(1)