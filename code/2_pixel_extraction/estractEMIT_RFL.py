#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMIT L2A NetCDF inspector + layer extractor

Pensato per file tipo:
- EMIT_L2A_RFL*.nc
- EMIT_L2A_RFLUNCERT*.nc
- EMIT_L2A_MASK*.nc

Fa:
1) Ispezione completa del NetCDF:
   - group
   - variabili
   - dimensioni
   - shape
   - dtype
2) Salvataggio inventario in CSV e JSON
3) Ricerca automatica di lat/lon nel group "location" (o fallback)
4) Export GeoTIFF delle variabili numeriche 2D
5) Export opzionale delle variabili numeriche 3D:
   - tutte le bande
   - solo bande selezionate
   - nessuna

PATCH MEMORIA:
- non carica più in RAM l'intero cubo 3D
- legge le variabili 3D una banda per volta
- evita conversioni inutili a float64 per i cubi
"""

import os
import json
import glob
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

# =============================================================================
# PATCH AMBIENTE GDAL / PROJ
# =============================================================================
BASE_PY = r"path/to/python_installation"

os.environ["GDAL_DATA"] = os.path.join(
    BASE_PY, "Lib", "site-packages", "rasterio", "gdal_data"
)
os.environ["PROJ_LIB"] = os.path.join(
    BASE_PY, "Lib", "site-packages", "rasterio", "proj_data"
)
os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]

import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# USER SETTINGS - MODIFICA SOLO QUESTE RIGHE
# =============================================================================

INPUT_FILES = [
    r"path/to/EMIT_L2A_MASK_001_20260216T085347_2604706_019.nc",
    r"path/to/EMIT_L2A_RFL_001_20260216T085347_2604706_019.nc",
    r"path/to/EMIT_L2A_RFLUNCERT_001_20260216T085347_2604706_019.nc",
]

INPUT_GLOB_PATTERN = None

OUTPUT_DIR = r"path/to/EMIT"

EXPORT_2D_VARS = True

# "none" | "selected_bands" | "all_bands"
EXPORT_3D_MODE = "selected_bands"

# indici 0-based
SELECTED_BAND_INDICES = [10, 50, 100, 150, 200]

SAVE_WAVELENGTHS_CSV = True
CROP_TO_VALID = True

DO_REPROJECT_UTM = False
OUTPUT_REPROJECT_CRS = "EPSG:32640"

ABS_INVALID_THRESHOLD = 1e20
DEFAULT_RES_DEG = 0.0005

# =============================================================================
# FUNZIONI
# =============================================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = str(name)
    for b in bad:
        out = out.replace(b, "_")
    return out


def sanitize_array(arr):
    """
    Sanitizza array numerici preservando il dtype float32 quando possibile.
    Non forza tutto a float64.
    """
    arr = np.array(arr)
    arr = np.squeeze(arr)

    if np.issubdtype(arr.dtype, np.number):
        if arr.dtype.kind in ("i", "u"):
            arr = arr.astype(np.float32, copy=False)
        elif arr.dtype.kind == "f" and arr.dtype.itemsize > 4:
            arr = arr.astype(np.float32, copy=False)

        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > ABS_INVALID_THRESHOLD] = np.nan

    return arr


def list_groups_and_variables(nc_path: str):
    root = Dataset(nc_path, "r")
    rows = []

    def walk(group, prefix=""):
        group_path = prefix if prefix else "/"
        for vname, var in group.variables.items():
            rows.append({
                "group_path": group_path,
                "variable": vname,
                "dimensions": ",".join(var.dimensions),
                "shape": tuple(var.shape),
                "dtype": str(var.dtype),
                "ndim": len(var.shape),
            })

        for gname, subgroup in group.groups.items():
            new_prefix = f"{prefix}/{gname}" if prefix else f"/{gname}"
            walk(subgroup, new_prefix)

    walk(root)
    root.close()
    return rows


def open_group_xr(nc_path: str, group_path: str):
    if group_path in ("/", ""):
        return xr.open_dataset(nc_path, engine="netcdf4")
    return xr.open_dataset(nc_path, engine="netcdf4", group=group_path.lstrip("/"))


def get_all_group_paths(nc_path: str) -> List[str]:
    root = Dataset(nc_path, "r")
    paths = []

    def walk(group, prefix=""):
        paths.append(prefix if prefix else "/")
        for gname, subgroup in group.groups.items():
            new_prefix = f"{prefix}/{gname}" if prefix else f"/{gname}"
            walk(subgroup, new_prefix)

    walk(root)
    root.close()
    return paths


def find_location_group(nc_path: str) -> Optional[str]:
    paths = get_all_group_paths(nc_path)
    for p in paths:
        if p.lower().endswith("/location") or p == "/location":
            return p
    return None


def find_lat_lon_in_group(ds) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
    lat_candidates = ["lat", "Lat", "latitude", "Latitude"]
    lon_candidates = ["lon", "Lon", "longitude", "Longitude"]

    lat_name = None
    lon_name = None

    for c in lat_candidates:
        if c in ds.variables:
            lat_name = c
            break
    for c in lon_candidates:
        if c in ds.variables:
            lon_name = c
            break

    lat = sanitize_array(ds[lat_name].values) if lat_name else None
    lon = sanitize_array(ds[lon_name].values) if lon_name else None

    return lat, lon, lat_name, lon_name


def find_lat_lon_anywhere(nc_path: str):
    loc_group = find_location_group(nc_path)
    if loc_group:
        ds = open_group_xr(nc_path, loc_group)
        lat, lon, lat_name, lon_name = find_lat_lon_in_group(ds)
        ds.close()
        if lat is not None and lon is not None:
            return lat, lon, loc_group, lat_name, lon_name

    for gp in get_all_group_paths(nc_path):
        ds = open_group_xr(nc_path, gp)
        lat, lon, lat_name, lon_name = find_lat_lon_in_group(ds)
        ds.close()
        if lat is not None and lon is not None:
            return lat, lon, gp, lat_name, lon_name

    return None, None, None, None, None


def estimate_resolution(lat: np.ndarray, lon: np.ndarray) -> Tuple[float, float]:
    dlon = np.nan
    dlat = np.nan

    if lat is not None and lon is not None:
        try:
            if lon.ndim == 2 and lon.shape[1] > 1:
                dlon = np.nanmedian(np.abs(np.diff(lon, axis=1)))
            if lat.ndim == 2 and lat.shape[0] > 1:
                dlat = np.nanmedian(np.abs(np.diff(lat, axis=0)))
        except Exception:
            pass

    if not np.isfinite(dlon) or dlon == 0:
        dlon = DEFAULT_RES_DEG
    if not np.isfinite(dlat) or dlat == 0:
        dlat = DEFAULT_RES_DEG

    return float(dlat), float(dlon)


def build_regular_grid(lat: np.ndarray, lon: np.ndarray, data_2d: np.ndarray):
    lat = sanitize_array(lat)
    lon = sanitize_array(lon)
    data = sanitize_array(data_2d)

    if lat is None or lon is None:
        raise ValueError("lat/lon non trovati.")
    if lat.shape != lon.shape:
        raise ValueError(f"lat e lon hanno shape diverse: {lat.shape} vs {lon.shape}")
    if data.shape != lat.shape:
        raise ValueError(f"La variabile 2D non ha la stessa shape di lat/lon: {data.shape} vs {lat.shape}")

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(data)
    if not np.any(valid):
        raise ValueError("Nessun pixel valido trovato per questa variabile.")

    lat_valid = lat[valid]
    lon_valid = lon[valid]
    data_valid = data[valid]

    dlat, dlon = estimate_resolution(lat, lon)

    xmin = np.nanmin(lon_valid)
    xmax = np.nanmax(lon_valid)
    ymin = np.nanmin(lat_valid)
    ymax = np.nanmax(lat_valid)

    width = int(np.ceil((xmax - xmin) / dlon)) + 1
    height = int(np.ceil((ymax - ymin) / dlat)) + 1

    width = max(1, min(width, 50000))
    height = max(1, min(height, 50000))

    grid = np.full((height, width), np.nan, dtype=np.float32)

    cols = np.floor((lon_valid - xmin) / dlon).astype(int)
    rows = np.floor((ymax - lat_valid) / dlat).astype(int)

    inside = (
        (rows >= 0) & (rows < height) &
        (cols >= 0) & (cols < width)
    )

    rows = rows[inside]
    cols = cols[inside]
    vals = data_valid[inside]

    grid[rows, cols] = vals.astype(np.float32)
    transform = from_origin(xmin, ymax, dlon, dlat)
    return grid, transform


def crop_to_valid(data: np.ndarray, transform):
    valid = np.isfinite(data)
    if not np.any(valid):
        return data, transform

    rows, cols = np.where(valid)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    cropped = data[rmin:rmax + 1, cmin:cmax + 1]

    new_transform = rasterio.transform.Affine(
        transform.a,
        transform.b,
        transform.c + cmin * transform.a,
        transform.d,
        transform.e,
        transform.f + rmin * transform.e
    )

    return cropped, new_transform


def write_geotiff(out_path: str, data: np.ndarray, transform, crs="EPSG:4326", nodata=np.nan):
    ensure_dir(os.path.dirname(out_path))

    if isinstance(crs, str) and crs.upper().startswith("EPSG:"):
        crs_obj = CRS.from_epsg(int(crs.split(":")[1]))
    else:
        crs_obj = CRS.from_user_input(crs)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=crs_obj,
        transform=transform,
        nodata=nodata,
        compress="deflate"
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def reproject_geotiff(src_path: str, dst_path: str, dst_crs: str):
    if isinstance(dst_crs, str) and dst_crs.upper().startswith("EPSG:"):
        dst_crs_obj = CRS.from_epsg(int(dst_crs.split(":")[1]))
    else:
        dst_crs_obj = CRS.from_user_input(dst_crs)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs_obj, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs_obj,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": "deflate"
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs_obj,
                resampling=Resampling.nearest
            )


def export_2d_array(lat, lon, arr_2d, out_base: str):
    grid, transform = build_regular_grid(lat, lon, arr_2d)
    if CROP_TO_VALID:
        grid, transform = crop_to_valid(grid, transform)

    out_4326 = f"{out_base}_EPSG4326.tif"
    write_geotiff(out_4326, grid, transform, crs="EPSG:4326")

    if DO_REPROJECT_UTM:
        out_reproj = f"{out_base}_{OUTPUT_REPROJECT_CRS.replace(':', '')}.tif"
        reproject_geotiff(out_4326, out_reproj, OUTPUT_REPROJECT_CRS)


def numeric_dtype(dt) -> bool:
    try:
        return np.issubdtype(np.dtype(dt), np.number)
    except Exception:
        return False


def detect_spatial_axes(arr_shape: Tuple[int, ...], latlon_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if len(arr_shape) != 3:
        return None

    target = tuple(latlon_shape)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if (arr_shape[i], arr_shape[j]) == target:
                return i, j
    return None


def move_spatial_to_rows_cols(arr_shape: Tuple[int, int, int], spatial_axes: Tuple[int, int]) -> Tuple[int, int, int]:
    axes = {0, 1, 2}
    band_axis = list(axes - set(spatial_axes))[0]
    return band_axis, spatial_axes[0], spatial_axes[1]


def find_wavelength_candidate(ds) -> Optional[np.ndarray]:
    candidates = ["wavelengths", "wavelength", "bands", "band_centers", "radiance_wl"]
    for c in candidates:
        if c in ds.variables:
            arr = np.array(ds[c].values)
            arr = np.ravel(arr)
            if np.issubdtype(arr.dtype, np.number) and arr.size > 2:
                return arr.astype(float)

    # prova anche tra le coordinate
    for c in ds.coords:
        arr = np.array(ds.coords[c].values)
        arr = np.ravel(arr)
        if np.issubdtype(arr.dtype, np.number) and arr.size > 10:
            name = str(c).lower()
            if "wave" in name or "band" in name:
                return arr.astype(float)

    return None


def export_variable_2d(ds, var_name: str, lat, lon, out_dir: str, file_stub: str):
    arr = sanitize_array(ds[var_name].values)

    if arr.ndim != 2:
        return False, f"skip {var_name}: ndim={arr.ndim}, non 2D"
    if lat is None or lon is None:
        return False, f"skip {var_name}: lat/lon non disponibili"
    if arr.shape != lat.shape:
        return False, f"skip {var_name}: shape {arr.shape} != lat/lon {lat.shape}"

    out_base = os.path.join(out_dir, f"{file_stub}__{sanitize_filename(var_name)}")
    export_2d_array(lat, lon, arr, out_base)
    return True, f"OK 2D export {var_name}"


def export_variable_3d(ds, var_name: str, lat, lon, out_dir: str, file_stub: str, wavelengths=None):
    """
    Export memory-safe: legge una banda per volta dal DataArray xarray.
    """
    da = ds[var_name]

    if da.ndim != 3:
        return False, f"skip {var_name}: ndim={da.ndim}, non 3D"
    if lat is None or lon is None:
        return False, f"skip {var_name}: lat/lon non disponibili"

    arr_shape = tuple(da.shape)
    spatial_axes = detect_spatial_axes(arr_shape, lat.shape)
    if spatial_axes is None:
        return False, f"skip {var_name}: impossibile associare assi spaziali a lat/lon"

    band_axis, row_axis, col_axis = move_spatial_to_rows_cols(arr_shape, spatial_axes)
    n_bands = arr_shape[band_axis]

    if EXPORT_3D_MODE == "none":
        return False, f"skip {var_name}: export 3D disattivato"
    if EXPORT_3D_MODE == "selected_bands":
        band_indices = [b for b in SELECTED_BAND_INDICES if 0 <= b < n_bands]
    elif EXPORT_3D_MODE == "all_bands":
        band_indices = list(range(n_bands))
    else:
        return False, f"skip {var_name}: EXPORT_3D_MODE non valido ({EXPORT_3D_MODE})"

    if not band_indices:
        return False, f"skip {var_name}: nessuna banda valida da esportare"

    cube_dir = os.path.join(out_dir, sanitize_filename(var_name))
    ensure_dir(cube_dir)

    rows_meta = []

    for b in band_indices:
        # Lettura lazy di UNA sola banda
        band_da = da.isel({da.dims[band_axis]: b})
        band2d = sanitize_array(band_da.values)

        # Riordina se necessario in (rows, cols)
        if band2d.shape != lat.shape:
            try:
                # band_da ora ha 2 dimensioni; allineiamo usando i due assi spaziali attesi
                remaining_dims = list(band_da.dims)

                # se shape coincide dopo transpose, prova
                for dims_try in [remaining_dims, remaining_dims[::-1]]:
                    try_arr = band_da.transpose(*dims_try).values
                    try_arr = sanitize_array(try_arr)
                    if try_arr.shape == lat.shape:
                        band2d = try_arr
                        break
                else:
                    band2d = np.reshape(band2d, lat.shape)
            except Exception:
                band2d = np.reshape(band2d, lat.shape)

        suffix = f"band_{b:04d}"
        if wavelengths is not None and b < len(wavelengths):
            wl = wavelengths[b]
            suffix += f"__wl_{wl:.6f}".replace(".", "_")
        else:
            wl = np.nan

        out_base = os.path.join(
            cube_dir,
            f"{file_stub}__{sanitize_filename(var_name)}__{suffix}"
        )
        export_2d_array(lat, lon, band2d, out_base)

        rows_meta.append({
            "variable": var_name,
            "band_index": int(b),
            "wavelength": float(wl) if np.isfinite(wl) else np.nan,
            "output_base": out_base
        })

    if rows_meta:
        meta_csv = os.path.join(cube_dir, f"{sanitize_filename(var_name)}__exported_bands.csv")
        pd.DataFrame(rows_meta).to_csv(meta_csv, index=False, encoding="utf-8-sig")

    return True, f"OK 3D export {var_name}: {len(band_indices)} bande"


def process_nc_file(nc_path: str, output_root: str):
    ensure_dir(output_root)

    file_name = os.path.basename(nc_path)
    file_stub = os.path.splitext(file_name)[0]
    file_out_dir = os.path.join(output_root, file_stub)
    ensure_dir(file_out_dir)

    print("\n" + "=" * 80)
    print(f"[PROCESS] {file_name}")
    print("=" * 80)

    inventory_rows = list_groups_and_variables(nc_path)
    inv_df = pd.DataFrame(inventory_rows)

    inv_csv = os.path.join(file_out_dir, f"{file_stub}__inventory.csv")
    inv_json = os.path.join(file_out_dir, f"{file_stub}__inventory.json")

    inv_df.to_csv(inv_csv, index=False, encoding="utf-8-sig")
    with open(inv_json, "w", encoding="utf-8") as f:
        json.dump(inventory_rows, f, indent=2, ensure_ascii=False, default=str)

    print(f"[OK] Inventario CSV:  {inv_csv}")
    print(f"[OK] Inventario JSON: {inv_json}")

    lat, lon, latlon_group, lat_name, lon_name = find_lat_lon_anywhere(nc_path)
    if lat is None or lon is None:
        print("[WARN] lat/lon non trovati. Export GeoTIFF impossibile per questo file.")
    else:
        print(f"[INFO] lat/lon trovati in group: {latlon_group}")
        print(f"[INFO] lat var: {lat_name} | lon var: {lon_name}")
        print(f"[INFO] lat/lon shape: {lat.shape}")

    summary_rows = []
    wavelength_rows = []

    group_paths = get_all_group_paths(nc_path)

    for gp in group_paths:
        ds = open_group_xr(nc_path, gp)

        gp_label = gp.strip("/").replace("/", "__") if gp != "/" else "root"
        gp_out_dir = os.path.join(file_out_dir, gp_label)
        ensure_dir(gp_out_dir)

        wavelengths = find_wavelength_candidate(ds)
        if wavelengths is not None and SAVE_WAVELENGTHS_CSV:
            wl_csv = os.path.join(gp_out_dir, f"{gp_label}__wavelengths.csv")
            pd.DataFrame({
                "band_index": np.arange(len(wavelengths), dtype=int),
                "wavelength": wavelengths
            }).to_csv(wl_csv, index=False, encoding="utf-8-sig")
            wavelength_rows.append({
                "group_path": gp,
                "wavelength_csv": wl_csv,
                "n_wavelengths": int(len(wavelengths))
            })

        vars_to_scan = list(ds.variables.keys())

        for var_name in vars_to_scan:
            var = ds[var_name]
            row = {
                "group_path": gp,
                "variable": var_name,
                "shape": tuple(var.shape),
                "dtype": str(var.dtype),
                "ndim": len(var.shape),
                "export_status": "not_processed",
                "message": ""
            }

            try:
                if not numeric_dtype(var.dtype):
                    row["export_status"] = "skipped"
                    row["message"] = "non numeric"
                    summary_rows.append(row)
                    continue

                ndim = len(var.shape)

                if ndim == 2 and EXPORT_2D_VARS:
                    ok, msg = export_variable_2d(ds, var_name, lat, lon, gp_out_dir, file_stub)
                    row["export_status"] = "exported" if ok else "skipped"
                    row["message"] = msg

                elif ndim == 3:
                    ok, msg = export_variable_3d(ds, var_name, lat, lon, gp_out_dir, file_stub, wavelengths=wavelengths)
                    row["export_status"] = "exported" if ok else "skipped"
                    row["message"] = msg

                else:
                    row["export_status"] = "skipped"
                    row["message"] = f"ndim={ndim} non gestito"

            except Exception as e:
                row["export_status"] = "error"
                row["message"] = str(e)

            summary_rows.append(row)

        ds.close()

    summary_csv = os.path.join(file_out_dir, f"{file_stub}__export_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Summary export: {summary_csv}")

    if wavelength_rows:
        wl_summary_csv = os.path.join(file_out_dir, f"{file_stub}__wavelength_sources.csv")
        pd.DataFrame(wavelength_rows).to_csv(wl_summary_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Summary wavelengths: {wl_summary_csv}")

    print(f"[DONE] File completato: {file_name}")


def build_input_list():
    files = []

    if INPUT_FILES:
        files.extend(INPUT_FILES)
    if INPUT_GLOB_PATTERN:
        files.extend(sorted(glob.glob(INPUT_GLOB_PATTERN)))

    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def main():
    ensure_dir(OUTPUT_DIR)

    files = build_input_list()
    if not files:
        raise FileNotFoundError("Nessun file input trovato. Controlla INPUT_FILES o INPUT_GLOB_PATTERN.")

    print("=" * 80)
    print("EMIT L2A NetCDF -> inventory + GeoTIFF extractor")
    print("=" * 80)
    print(f"[OUTPUT_DIR] {OUTPUT_DIR}")
    print(f"[EXPORT_2D_VARS] {EXPORT_2D_VARS}")
    print(f"[EXPORT_3D_MODE] {EXPORT_3D_MODE}")
    print(f"[SELECTED_BAND_INDICES] {SELECTED_BAND_INDICES}")
    print(f"[DO_REPROJECT_UTM] {DO_REPROJECT_UTM}")
    if DO_REPROJECT_UTM:
        print(f"[OUTPUT_REPROJECT_CRS] {OUTPUT_REPROJECT_CRS}")
    print("=" * 80)

    for nc_path in files:
        if not os.path.exists(nc_path):
            print(f"[WARN] File non trovato, salto: {nc_path}")
            continue

        try:
            process_nc_file(nc_path, OUTPUT_DIR)
        except Exception as e:
            print(f"[ERROR] Fallito su file: {nc_path}")
            print(str(e))

    print("\n[FINE] Tutto completato.")


if __name__ == "__main__":
    print("[TEST CRS] EPSG:4326 ->", CRS.from_epsg(4326))
    if DO_REPROJECT_UTM:
        print(f"[TEST CRS] {OUTPUT_REPROJECT_CRS} ->", CRS.from_user_input(OUTPUT_REPROJECT_CRS))
    main()