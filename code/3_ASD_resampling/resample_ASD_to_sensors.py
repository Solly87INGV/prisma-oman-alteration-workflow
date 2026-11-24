# -*- coding: utf-8 -*-
"""
resample_ASD_to_sensors.py

Ricampiona gli spettri di laboratorio ASD FieldSpec 4 (passo 1 nm, 350-2500 nm)
sulle lunghezze d'onda centrali delle bande di EnMAP ed EMIT (e, opzionalmente,
PRISMA per verifica), con lo stesso metodo descritto nel par. 2.5.2:
    interpolazione lineare a tratti sui centri banda del sensore,
    SENZA convoluzione con la Spectral Response Function (SRF).

Input ASD supportati (riconosciuti automaticamente):
  1) export testuale ViewSpec (*.mn.txt / *.asd.txt): header ASD, poi riga
     "Wavelength<TAB>nome_spettro" e valori con virgola decimale;
     se il file contiene piu colonne, ogni colonna e trattata come uno spettro.
  2) formato long tab-separato: sample_id<TAB>wavelength<TAB>reflectance
     (anche con piu sample_id nello stesso file).

Sorgenti delle lunghezze d'onda dei sensori (riconosciute dall'estensione):
  - .csv : tabella con una colonna 'wavelength' (es. band_wavelength_lookup.csv)
  - .xml : METADATA.XML EnMAP (bandID / wavelengthCenterOfBand)
  - .nc  : file EMIT L2A RFL (gruppo sensor_band_parameters / wavelengths)
  - .hdr : header ENVI (campo wavelength = {...})
  - .txt : spettro gia estratto dal sensore (colonna 'wavelength', es. SOIA_1_ENMAP.txt)
Le unita (um o nm) sono riconosciute automaticamente: se max(wl) < 10 -> um -> x1000.

Output, per ogni sensore attivo:
  <OUTPUT_DIR>/<SENSORE>/<sample>_ASD_<SENSORE>.txt   (sample_id, wavelength, reflectance)
  <OUTPUT_DIR>/<SENSORE>/ASD_<SENSORE>_wide.csv       (tutte le bande x tutti i campioni)
  <OUTPUT_DIR>/<SENSORE>/ASD_<SENSORE>_bands_used.csv (centri banda usati / scartati)
  <OUTPUT_DIR>/run_info.txt
"""

import os
import re
import glob
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# =============================================================================
# USER SETTINGS - MODIFICA SOLO QUESTE RIGHE
# =============================================================================

# Cartella con gli spettri ASD a 1 nm
ASD_INPUT_DIR = r"path/to/ASD_spectra"

# Pattern dei file ASD da leggere nella cartella
ASD_GLOB_PATTERNS = ["*.mn.txt", "*.txt"]

# Elabora solo questi file (nomi senza percorso). Lista vuota = tutti i file trovati.
# Esempio: ["SOIA_1_1.txt", "SOIA_1_2.txt", "SOIA_2_mean.mn.txt", "SOIA_13_mean.mn.txt"]
ASD_ONLY_FILES = []

# Sorgenti delle lunghezze d'onda. Metti enabled=False per saltare un sensore.
SENSORS = {
    "EnMAP": {
        "enabled": True,
        "source": r"path/to/band_wavelength_lookup.csv",
    },
    "EMIT": {
        "enabled": True,
        "source": r"path/to/EMIT_L2A_RFL_001_20260216T085347_2604706_019.nc",
    },
    "PRISMA": {
        "enabled": True,
        "source": r"path/to/VNIR_SWIR_latlon_219.hdr",
    },
}

OUTPUT_DIR = r"path/to/ASD_resampled"

# Fattore di scala della riflettanza in output.
# 1.0 = riflettanza 0-1 come ASD; 10000 = stessa scala degli spettri EnMAP estratti.
OUTPUT_SCALE = 1.0

# Bande del sensore fuori dall'intervallo ASD (350-2500 nm) vengono sempre scartate.

# =============================================================================
# LETTURA SPETTRI ASD
# =============================================================================


def _to_float(s):
    s = str(s).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_asd_viewspec(path):
    """Export ViewSpec: header libero, poi 'Wavelength<TAB>nome...'. Ritorna dict nome -> (wl, refl)."""
    with open(path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("wavelength"):
            start = i
            break
    if start is None:
        return {}
    names = [n.strip() for n in lines[start].split("\t")[1:] if n.strip()]
    rows = []
    for ln in lines[start + 1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        vals = [_to_float(p) for p in parts]
        rows.append(vals[: 1 + len(names)])
    arr = np.array([r + [np.nan] * (1 + len(names) - len(r)) for r in rows], dtype=float)
    out = {}
    for j, name in enumerate(names, start=1):
        name = re.sub(r"\.(mn|asd)$", "", name, flags=re.IGNORECASE)
        out[name] = (arr[:, 0], arr[:, j])
    return out


def read_asd_long(path):
    """Formato long: sample_id, wavelength, reflectance (tab)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df["wavelength"] = df["wavelength"].map(_to_float)
    df["reflectance"] = df["reflectance"].map(_to_float)
    out = {}
    for sid, g in df.groupby("sample_id", sort=False):
        out[str(sid).strip()] = (g["wavelength"].to_numpy(), g["reflectance"].to_numpy())
    return out


def read_asd_file(path):
    with open(path, "r", encoding="latin-1") as f:
        first = f.readline().strip().lower()
    if first.startswith("sample_id"):
        return read_asd_long(path)
    return read_asd_viewspec(path)


def collect_asd_spectra():
    files = []
    for pat in ASD_GLOB_PATTERNS:
        files.extend(glob.glob(os.path.join(ASD_INPUT_DIR, pat)))
    files = sorted(set(files))
    if ASD_ONLY_FILES:
        wanted = {f.lower() for f in ASD_ONLY_FILES}
        files = [f for f in files if os.path.basename(f).lower() in wanted]
    spectra = {}
    for fp in files:
        try:
            d = read_asd_file(fp)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(fp)}: {e}")
            continue
        if not d:
            print(f"[SKIP] {os.path.basename(fp)}: formato non riconosciuto")
            continue
        for name, (wl, r) in d.items():
            ok = np.isfinite(wl) & np.isfinite(r)
            wl, r = wl[ok], r[ok]
            order = np.argsort(wl)
            wl, r = wl[order], r[order]
            if wl.size < 100:
                print(f"[SKIP] {name}: solo {wl.size} punti validi")
                continue
            if name in spectra:
                print(f"[WARN] {name} duplicato ({os.path.basename(fp)}): sovrascritto")
            spectra[name] = {"wl": wl, "refl": r, "file": fp}
            print(f"[ASD] {name:<25s} {wl.min():.0f}-{wl.max():.0f} nm, {wl.size} punti  ({os.path.basename(fp)})")
    return spectra


# =============================================================================
# LETTURA LUNGHEZZE D'ONDA DEI SENSORI
# =============================================================================


def wl_from_csv_or_txt(path):
    sep = "\t" if path.lower().endswith(".txt") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    col = next((c for c in df.columns if c.strip().lower() == "wavelength"), None)
    if col is None:
        col = next((c for c in df.columns if "wavelength" in c.lower() and "source" not in c.lower()), None)
    if col is None:
        raise ValueError(f"nessuna colonna 'wavelength' in {path}")
    wl = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    return np.unique(wl)  # unique: un file spettro long puo ripetere le stesse bande


def wl_from_enmap_xml(path):
    root = ET.parse(path).getroot()
    strip = lambda t: t.split("}")[-1]
    wls = []
    for elem in root.iter():
        if strip(elem.tag) == "bandID":
            for ch in elem:
                if strip(ch.tag) == "wavelengthCenterOfBand":
                    wls.append(float(ch.text))
    if not wls:
        raise ValueError("wavelengthCenterOfBand non trovato nel METADATA.XML")
    return np.array(wls, dtype=float)


def wl_from_emit_nc(path):
    from netCDF4 import Dataset
    with Dataset(path, "r") as ds:
        if "sensor_band_parameters" in ds.groups:
            g = ds.groups["sensor_band_parameters"]
            if "wavelengths" in g.variables:
                return np.array(g.variables["wavelengths"][:], dtype=float).ravel()
        for name in ("wavelengths", "wavelength"):
            if name in ds.variables:
                return np.array(ds.variables[name][:], dtype=float).ravel()
    raise ValueError("wavelengths non trovate nel file EMIT")


def wl_from_envi_hdr(path):
    with open(path, "r", encoding="latin-1") as f:
        txt = f.read()
    m = re.search(r"^\s*wavelength\s*=\s*\{([^}]*)\}", txt, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        raise ValueError("campo 'wavelength = {...}' non trovato nell'hdr")
    return np.array([float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()], dtype=float)


def read_sensor_wavelengths(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nc":
        wl = wl_from_emit_nc(path)
    elif ext == ".xml":
        wl = wl_from_enmap_xml(path)
    elif ext == ".hdr":
        wl = wl_from_envi_hdr(path)
    elif ext in (".csv", ".txt"):
        wl = wl_from_csv_or_txt(path)
    else:
        raise ValueError(f"estensione non gestita: {ext}")
    wl = wl[np.isfinite(wl) & (wl > 0)]
    unit = "nm"
    if wl.max() < 10:  # micrometri
        wl = wl * 1000.0
        unit = "um (convertite in nm)"
    return np.sort(wl), unit


# =============================================================================
# RICAMPIONAMENTO
# =============================================================================


def resample_linear(asd_wl, asd_refl, target_wl):
    """Interpolazione lineare a tratti sui centri banda; NaN fuori dall'intervallo ASD."""
    return np.interp(target_wl, asd_wl, asd_refl, left=np.nan, right=np.nan)


def check_paths():
    """Controlla tutti i percorsi prima di partire e li elenca."""
    print("=" * 70)
    print("Controllo percorsi")
    print("=" * 70)
    ok = True
    items = [("ASD_INPUT_DIR", ASD_INPUT_DIR)]
    items += [(f"{k} source", v["source"]) for k, v in SENSORS.items() if v.get("enabled", False)]
    for label, path in items:
        exists = os.path.exists(path)
        print(f"[{'OK' if exists else 'NON TROVATO'}] {label}: {path}")
        if label == "ASD_INPUT_DIR" and not exists:
            ok = False
    out_parent = os.path.dirname(os.path.abspath(OUTPUT_DIR))
    exists = os.path.isdir(out_parent)
    print(f"[{'OK' if exists else 'NON TROVATO'}] cartella che conterra OUTPUT_DIR: {out_parent}")
    if not exists:
        ok = False
    if not ok:
        raise SystemExit("Correggi i percorsi NON TROVATO in USER SETTINGS e rilancia.")


def main():
    check_paths()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Lettura spettri ASD")
    print("=" * 70)
    spectra = collect_asd_spectra()
    if not spectra:
        raise SystemExit("Nessuno spettro ASD letto: controlla ASD_INPUT_DIR / ASD_GLOB_PATTERNS / ASD_ONLY_FILES")

    asd_min = max(s["wl"].min() for s in spectra.values())
    asd_max = min(s["wl"].max() for s in spectra.values())

    info = [
        f"Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "Metodo: interpolazione lineare a tratti (numpy.interp) sui centri banda del sensore; nessuna convoluzione SRF.",
        f"ASD input dir: {ASD_INPUT_DIR}",
        f"Spettri ASD ({len(spectra)}): " + ", ".join(spectra.keys()),
        f"Intervallo ASD comune: {asd_min:.1f}-{asd_max:.1f} nm",
        f"OUTPUT_SCALE: {OUTPUT_SCALE}",
        "",
    ]

    for sensor, cfg in SENSORS.items():
        if not cfg.get("enabled", False):
            continue
        print("=" * 70)
        print(f"Sensore: {sensor}")
        print("=" * 70)
        src = cfg["source"]
        if not os.path.exists(src):
            print(f"[ERRORE] sorgente non trovata: {src}")
            info.append(f"[{sensor}] ERRORE: sorgente non trovata: {src}")
            continue
        wl_all, unit = read_sensor_wavelengths(src)
        inside = (wl_all >= asd_min) & (wl_all <= asd_max)
        wl = wl_all[inside]
        print(f"Bande sensore: {wl_all.size} ({unit}), {wl_all.min():.2f}-{wl_all.max():.2f} nm")
        print(f"Bande dentro l'intervallo ASD: {wl.size}; scartate: {(~inside).sum()}")

        out_dir = os.path.join(OUTPUT_DIR, sensor)
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame({"wavelength_nm": wl_all, "used": inside}).to_csv(
            os.path.join(out_dir, f"ASD_{sensor}_bands_used.csv"), index=False)

        wide = pd.DataFrame({"wavelength": wl})
        for name, s in spectra.items():
            r = resample_linear(s["wl"], s["refl"], wl) * OUTPUT_SCALE
            sid = f"{name}_ASD_{sensor}"
            pd.DataFrame({"sample_id": sid, "wavelength": wl, "reflectance": r}).to_csv(
                os.path.join(out_dir, f"{sid}.txt"), sep="\t", index=False, float_format="%.6f")
            wide[name] = r
            print(f"  [OK] {sid}.txt")
        wide.to_csv(os.path.join(out_dir, f"ASD_{sensor}_wide.csv"), index=False, float_format="%.6f")

        info.append(f"[{sensor}] sorgente wavelengths: {src}")
        info.append(f"[{sensor}] bande sensore: {wl_all.size} ({unit}), {wl_all.min():.3f}-{wl_all.max():.3f} nm")
        info.append(f"[{sensor}] bande ricampionate (dentro intervallo ASD): {wl.size}; scartate: {(~inside).sum()}")
        info.append("")

    with open(os.path.join(OUTPUT_DIR, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info))
    print("=" * 70)
    print(f"Fatto. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()