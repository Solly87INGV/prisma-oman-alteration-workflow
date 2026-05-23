#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRISMA → SAM — versione directory-driven
- La libreria SAM può essere caricata tutta dalla directory, senza filtri sui nomi
- Possibilità di far calcolare la "subset mask" su:
    1) tutti i file trovati in directory
    2) solo su un sottoinsieme riconosciuto da chiavi nel nome file
- Calcolo angolo SAM globale e subset-mask nello stesso loop
- Soglia subset-mask = percentile basso (angoli piccoli) + conf minima
- Prospectivity ribilanciata
"""

import os, re, math, json, logging, csv
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape as shp_shape
import fiona
from scipy.ndimage import median_filter, binary_dilation, binary_erosion

# ==================== CONFIG ====================
INPUT_ENVI = r"D:\INGV\1_Human Mobility\PRISMA\PRS_L2D_STD_20250904064907_20250904064911_0001\VNIR_SWIR_latlon_907_CLEAN"
SPECTRAL_DIR = r"E:\1_Human Mobility\MASPAG_2025\SAM_Hyperspectral\Hyperspectral\selected_minerals_NEW"
OUTPUT_DIR   = r"D:\INGV\1_Human Mobility\OUTPUT_SAPM\907"

GEOLOGY_SHP   = None   # es: r"D:\...\geology.shp"
GEOLOGY_FIELD = None   # es: "UNIT"

# ==========================================================
# LIBRERIA: usa TUTTI i .txt trovati in directory
# ==========================================================
USE_ALL_TXT_IN_DIRECTORY = True

# Come costruire la mask "subset" (ex copper mask)
# - "all"       -> usa TUTTI i minerali trovati in directory
# - "name_keys" -> usa solo quelli il cui nome file contiene una delle chiavi sotto
#
# Se scegli "all", la mask non è più una vera "copper mask" in senso stretto:
# diventa una mask dei migliori match spettrali relativi all'intera libreria.
SUBSET_MASK_MODE = "all"   # "all" oppure "name_keys"

# Queste chiavi servono solo se SUBSET_MASK_MODE = "name_keys"
SUBSET_NAME_KEYS = [
    "malachite", "azurite", "cuprite", "chalcopyrite", "bornite",
    "chrysocolla", "tennantite", "covellite"
]

# --- Soglie/adattività ---
BASE_ANGLE_THR_RAD  = 0.35     # soglia QA globale (~20°)
SUBSET_PERCENTILE   = 0.10     # percentile basso su ang_subset (top 10% migliori)
MIN_CONF_SUBSET     = 0.70     # confidence minima per dichiarare la subset-mask

# --- Pesi prospectivity (0–1) ---
W_SUB  = 0.60
W_ALT  = 0.25
W_CONF = 0.15

# --- Certainty tiers (su confidence globale) ---
CERT_HIGH = 0.80
CERT_MED  = 0.60
CERT_LOW  = 0.45

# --- Pulizia spaziale ---
DO_MEDIAN_3x3 = True
DILATE_ITERS  = 1
ERODE_ITERS   = 0

# ------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(OUTPUT_DIR, "processing_log_directory_driven.txt"), encoding="utf-8"),
              logging.StreamHandler()]
)

def read_wavelengths_from_hdr(hdr_path):
    try:
        txt = open(hdr_path, 'r', encoding='utf-8', errors='ignore').read()
        m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.I | re.S)
        if not m:
            return None
        vals = [v.strip() for v in m.group(1).replace("\n", " ").split(",")]
        wls = []
        for v in vals:
            try:
                wls.append(float(v))
            except:
                pass
        wls = np.array(wls, dtype=float)
        if np.nanmedian(wls) > 100:
            wls = wls / 1000.0
        return wls
    except Exception as e:
        logging.warning(f"WL HDR: {e}")
        return None

def mask_good_bands(wl_um):
    wl = np.array(wl_um)
    return (wl >= 0.45) & (wl <= 2.45) & ~(((wl >= 1.35) & (wl <= 1.45)) | ((wl >= 1.80) & (wl <= 1.95)))

def parse_usgs_txt(path):
    xs, ys = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("splib", "Version", "Description", "#", ";")):
                continue
            parts = line.replace(",", " ").split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except:
                    pass
            if len(nums) >= 2:
                xs.append(nums[0]); ys.append(nums[1])
            elif len(nums) == 1:
                ys.append(nums[0])
    if len(xs) >= 5 and len(ys) >= 5 and len(xs) == len(ys):
        wl = np.array(xs, float); refl = np.array(ys, float)
    elif len(ys) >= 20:
        wl = np.linspace(0.35, 2.5, len(ys)); refl = np.array(ys, float)
    else:
        return None, None
    if np.nanmax(refl) > 1.5:
        refl = refl / 100.0
    return np.clip(wl, 0, 5), np.clip(refl, 0, 1.2)

def continuum_removal(y):
    x = np.arange(len(y))
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = x[i], y[i]
            if (y2 - y1) * (x3 - x2) >= (y3 - y2) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((x[i], y[i]))
    hx = np.array([p[0] for p in hull]); hy = np.array([p[1] for p in hull])
    cont = np.interp(x, hx, hy)
    cont[cont == 0] = 1e-6
    return np.clip(y / cont, 0, 2.0)

def sam_angle(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return math.pi / 2
    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return math.acos(cosang)

def open_prisma(envi_base):
    data_file = envi_base
    if not os.path.exists(data_file):
        if os.path.exists(envi_base + ".dat"):
            data_file = envi_base + ".dat"
        else:
            raise FileNotFoundError(f"File dati non trovato: {envi_base}")
    with rasterio.open(data_file) as ds:
        cube = ds.read()  # (bands,rows,cols)
        transform, crs = ds.transform, ds.crs
    return cube, transform, crs

def save_tif(path, arr, transform, crs, dtype=None):
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=dtype, crs=crs, transform=transform, compress='DEFLATE'
    ) as dst:
        dst.write(arr.astype(dtype), 1)

def stats(arr, msk=None):
    if msk is not None:
        arr = arr[msk > 0]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "min": float(np.nanmin(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p90": float(np.nanpercentile(arr, 90)),
        "max": float(np.nanmax(arr))
    }

def main():
    logging.info("=== PRISMA SAM (directory-driven) ===")

    cube, transform, crs = open_prisma(INPUT_ENVI)
    n_b, r, c = cube.shape
    img = np.transpose(cube, (1, 2, 0))

    hdr = INPUT_ENVI if INPUT_ENVI.endswith(".hdr") else INPUT_ENVI + ".hdr"
    wl = read_wavelengths_from_hdr(hdr)
    if wl is None or len(wl) != n_b:
        logging.warning("Wavelengths mancanti/mismatch: uso 0.4–2.5 linspace")
        wl = np.linspace(0.4, 2.5, n_b)

    good = mask_good_bands(wl)
    gidx = np.where(good)[0]
    logging.info(f"Bande buone: {good.sum()}/{n_b}")

    # Preprocess PRISMA
    data = img.reshape(-1, n_b)
    data[~np.isfinite(data)] = 0
    data[data < 0] = 0
    dg = data[:, gidx]
    norms = np.linalg.norm(dg, axis=1, keepdims=True) + 1e-9
    dg = dg / norms
    for i in range(dg.shape[0]):
        dg[i, :] = continuum_removal(dg[i, :])
    img_g = dg.reshape(r, c, -1)

    # Carica libreria da directory
    files = sorted([f for f in os.listdir(SPECTRAL_DIR) if f.lower().endswith(".txt")])
    if not files:
        raise RuntimeError("Nessun file .txt trovato in SPECTRAL_DIR")

    lib = {}
    for fname in files:
        fpath = os.path.join(SPECTRAL_DIR, fname)
        wl_u, refl_u = parse_usgs_txt(fpath)
        if wl_u is None:
            logging.warning(f"LIB non valida: {fname}")
            continue
        interp = np.interp(wl[good], wl_u, refl_u, left=0, right=0)
        interp = interp / (np.linalg.norm(interp) + 1e-9)
        interp = continuum_removal(interp)
        mineral_name = os.path.splitext(fname)[0]
        lib[mineral_name] = interp

    if not lib:
        raise RuntimeError("Nessuno spettro valido caricato dalla directory.")

    mineral_list = list(lib.keys())
    json.dump({i + 1: m for i, m in enumerate(mineral_list)},
              open(os.path.join(OUTPUT_DIR, "legend_minerals.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logging.info(f"Minerali usati ({len(mineral_list)}): {mineral_list}")

    # Definizione subset per la mask
    if SUBSET_MASK_MODE == "all":
        subset_set = set(mineral_list)
    elif SUBSET_MASK_MODE == "name_keys":
        subset_set = set([m for m in mineral_list if any(k in m.lower() for k in SUBSET_NAME_KEYS)])
    else:
        raise ValueError("SUBSET_MASK_MODE deve essere 'all' oppure 'name_keys'")

    if not subset_set:
        logging.warning("Subset-set vuoto: la subset mask sarà tutta zero/NaN.")
    else:
        logging.info(f"Minerali usati per subset-mask ({len(subset_set)}): {sorted(subset_set)}")

    # SAM loop
    best_idx = np.zeros((r, c), np.uint16)
    best_ang = np.full((r, c), math.pi / 2, np.float32)
    ang_subset = np.full((r, c), math.pi / 2, np.float32)

    for i in range(r):
        if i % 50 == 0:
            logging.info(f"Riga {i + 1}/{r}")
        row = img_g[i, :, :]
        for j in range(c):
            s = row[j, :]
            if not np.any(s):
                continue
            ba_all = math.pi / 2
            bidx = 0
            ba_sub = math.pi / 2
            for k, m in enumerate(mineral_list):
                ang = sam_angle(lib[m], s)
                if ang < ba_all:
                    ba_all = ang
                    bidx = k + 1
                if m in subset_set and ang < ba_sub:
                    ba_sub = ang

            best_idx[i, j] = bidx
            best_ang[i, j] = ba_all
            ang_subset[i, j] = ba_sub

    confidence = 1.0 - (best_ang / (math.pi / 2))
    subset_conf = 1.0 - (ang_subset / (math.pi / 2))
    subset_conf = np.clip(subset_conf, 0, 1)

    # QA globale
    mask_ok = best_ang <= BASE_ANGLE_THR_RAD

    valid_sub = np.isfinite(ang_subset)
    if not valid_sub.any():
        logging.warning("ang_subset vuoto: subset-mask sarà tutta 0.")
        thr_sub = math.pi / 2
    else:
        thr_sub = np.quantile(ang_subset[valid_sub], SUBSET_PERCENTILE)
    thr_sub_deg = float(np.degrees(thr_sub))

    subset_mask = ((ang_subset <= thr_sub) & (confidence >= MIN_CONF_SUBSET)).astype(np.uint8)

    # alteration mask non vincolata a nomi
    alteration_mask = mask_ok.astype(np.uint8)

    # Pulizia spaziale
    if DO_MEDIAN_3x3:
        subset_mask = median_filter(subset_mask, size=3)
        alteration_mask = median_filter(alteration_mask, size=3)
    if DILATE_ITERS > 0:
        for _ in range(DILATE_ITERS):
            subset_mask = binary_dilation(subset_mask).astype(np.uint8)
    if ERODE_ITERS > 0:
        for _ in range(ERODE_ITERS):
            subset_mask = binary_erosion(subset_mask).astype(np.uint8)

    alter_dil = binary_dilation(alteration_mask, structure=np.ones((3, 3))).astype(np.uint8)
    prospectivity = (W_SUB * subset_conf + W_ALT * alter_dil + W_CONF * confidence).astype(np.float32)
    prospectivity = np.clip(prospectivity, 0, 1)

    certainty_map = np.zeros((r, c), np.uint8)
    certainty_map[(confidence >= CERT_LOW) & (confidence < CERT_MED)] = 1
    certainty_map[(confidence >= CERT_MED) & (confidence < CERT_HIGH)] = 2
    certainty_map[(confidence >= 0.90)] = 3

    base = os.path.splitext(os.path.basename(INPUT_ENVI))[0]
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_mineral_map.tif"), best_idx, transform, crs, 'uint16')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_angle_deg.tif"), np.degrees(best_ang).astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_confidence.tif"), confidence.astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_angsubset_deg.tif"), np.degrees(ang_subset).astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_subset_conf.tif"), subset_conf.astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_subset_mask.tif"), (subset_mask * 255).astype(np.uint8), transform, crs, 'uint8')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_alteration_mask.tif"), (alteration_mask * 255).astype(np.uint8), transform, crs, 'uint8')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_prospectivity.tif"), prospectivity.astype(np.float32), transform, crs, 'float32')
    save_tif(os.path.join(OUTPUT_DIR, f"{base}_certainty_map.tif"), certainty_map, transform, crs, 'uint8')

    def pct(n):
        return 100.0 * n / (r * c)

    rep = {
        "pixels_total": int(r * c),
        "subset_pixels": int(subset_mask.sum()),
        "subset_pct": pct(subset_mask.sum()),
        "alter_pixels": int(alteration_mask.sum()),
        "alter_pct": pct(alteration_mask.sum()),
        "angle_deg_stats": stats(np.degrees(best_ang)),
        "angsubset_deg_stats": stats(np.degrees(ang_subset)),
        "confidence_stats": stats(confidence),
        "subset_conf_stats": stats(subset_conf, subset_mask),
        "prospectivity_stats": stats(prospectivity),
        "thresholds": {
            "global_QA_deg": float(np.degrees(BASE_ANGLE_THR_RAD)),
            "subset_percentile": SUBSET_PERCENTILE,
            "subset_thr_deg": thr_sub_deg,
            "min_conf_subset": MIN_CONF_SUBSET
        },
        "subset_mask_mode": SUBSET_MASK_MODE,
        "subset_name_keys": SUBSET_NAME_KEYS,
        "legend": {i + 1: m for i, m in enumerate(mineral_list)}
    }
    json.dump(rep, open(os.path.join(OUTPUT_DIR, "report_directory_driven.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    if GEOLOGY_SHP and GEOLOGY_FIELD:
        with fiona.open(GEOLOGY_SHP, 'r') as src:
            geoms = []
            classes = []
            for f in src:
                geoms.append(shp_shape(f["geometry"]).__geo_interface__)
                classes.append(f["properties"][GEOLOGY_FIELD])

        class_set = {v: i + 1 for i, v in enumerate(sorted(set(classes)))}
        shapes = [(g, class_set[c]) for g, c in zip(geoms, classes)]
        class_ras = rasterize(shapes, out_shape=(r, c), transform=transform, fill=0, dtype='uint16')

        rows = [["class", "pixels", "angsubset_p50", "angsubset_p90", "conf_p90", "subsetconf_p90", "pros_p90", "subset_pct"]]
        for name, code in class_set.items():
            m = (class_ras == code)
            if m.sum() == 0:
                continue

            def P(arr, p):
                a = arr[m]
                a = a[np.isfinite(a)]
                return float(np.nanpercentile(a, p)) if a.size else float("nan")

            rows.append([
                name, int(m.sum()),
                P(np.degrees(ang_subset), 50), P(np.degrees(ang_subset), 90),
                P(confidence, 90), P(subset_conf, 90), P(prospectivity, 90),
                pct((subset_mask & m).sum())
            ])

        with open(os.path.join(OUTPUT_DIR, "zonal_geology_directory_driven.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    logging.info("Fatto. Controlla report_directory_driven.json e gli output TIFF.")

if __name__ == "__main__":
    main()
