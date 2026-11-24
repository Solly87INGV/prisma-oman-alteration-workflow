#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =============================================================================
# USER SETTINGS - MODIFICA SOLO QUESTE RIGHE
# =============================================================================

# Directory di fallback: usata solo se INPUT_SPECTRUM_FILES è vuota

# ------------------------------------------------------------------
# INPUT A BLOCCHI
# ------------------------------------------------------------------
# Ogni blocco rappresenta un campione / confronto multi-sensore.
# Dentro ogni blocco puoi mettere 4 file, per esempio:
#   1) ASD / laboratorio
#   2) EnMAP
#   3) PRISMA Savitzky-Golay
#   4) EMIT
#
# Logica degli offset:
# - OFFSET_STEP controlla la distanza tra gli spettri DENTRO lo stesso blocco.
# - GROUP_OFFSET_STEP controlla la distanza tra un blocco e il blocco successivo.
#
# Gli output FULL e MASKED restano invariati: vengono solo plottati con offset a blocchi.

# ------------------------------------------------------------------
# FIG. 10 - CONFIGURAZIONE (modifica solo qui)
# ------------------------------------------------------------------
# R  = cartella dell'ASD ricampionato (sottocartelle PRISMA/, EnMAP/, EMIT/)
# P  = cartella dei file pixel (SOIA_*_PRISMA_savgol.txt, SOIA_*_ENMAP_ovrlp.txt, SOIA_*_EMIT.txt)
# OUT_BASE = cartella in cui viene creata una sottocartella per ogni pannello
R = "path/to/ASD_resampled"
P = "path/to/all"
OUT_BASE = "path/to/Fig10"

# Pannelli: (campione ASD, pixel). AS 1_1 e AS 1_2 usano lo stesso pixel SOIA_1.
# Per lanciarne solo alcuni, commenta le righe che non servono.
PANELS = [
    ("SOIA_13",  "SOIA_13"),   # (a) AS 13
    ("SOIA_2",   "SOIA_2"),    # (b) AS 2
    ("SOIA_1_1", "SOIA_1"),    # (c) AS 1_1
    ("SOIA_1_2", "SOIA_1"),    # (d) AS 1_2
]

SAMPLE_COLORS = {"SOIA_13": "#16A34A", "SOIA_2": "#0EA5E9", "SOIA_1_1": "#7010BE", "SOIA_1_2": "#D97706"}
SENSOR_PIXEL_COLORS = {"PRISMA": "#A16207", "EnMAP": "#BE1075", "EMIT": "#4B494D"}


def make_groups(S: str, PX: str) -> List[dict]:
    """Tre coppie per pannello: ASD ricampionato sul sensore + pixel dello stesso sensore."""
    return [
        {"group_label": "PRISMA", "files": [f"{R}/PRISMA/{S}_ASD_PRISMA.txt", f"{P}/{PX}_PRISMA_savgol.txt"]},
        {"group_label": "EnMAP",  "files": [f"{R}/EnMAP/{S}_ASD_EnMAP.txt",   f"{P}/{PX}_ENMAP_ovrlp.txt"]},
        {"group_label": "EMIT",   "files": [f"{R}/EMIT/{S}_ASD_EMIT.txt",     f"{P}/{PX}_EMIT.txt"]},
    ]


def make_colors(S: str, PX: str) -> Dict[str, str]:
    """ASD ricampionato = colore del campione; pixel = colore del sensore."""
    c = SAMPLE_COLORS.get(S, "#000000")
    return {
        f"{S}_ASD_PRISMA": c, f"{S}_ASD_EnMAP": c, f"{S}_ASD_EMIT": c,
        f"{PX}_PRISMA_savgol": SENSOR_PIXEL_COLORS["PRISMA"],
        f"{PX}_ENMAP_ovrlp":   SENSOR_PIXEL_COLORS["EnMAP"],
        f"{PX}_EMIT":          SENSOR_PIXEL_COLORS["EMIT"],
    }


def as_name(code: str) -> str:
    """SOIA_13 -> AS 13 ; SOIA_1_2 -> AS 1_2"""
    return code.replace("SOIA_", "AS ", 1) if code.startswith("SOIA_") else code


def make_labels(S: str, PX: str) -> Dict[str, str]:
    """Etichette mostrate nel grafico (chiave = nome file senza estensione)."""
    a, ap = as_name(S), as_name(PX)
    return {
        f"{S}_ASD_PRISMA": f"{a} ASD (PRISMA)",
        f"{S}_ASD_EnMAP":  f"{a} ASD (EnMAP)",
        f"{S}_ASD_EMIT":   f"{a} ASD (EMIT)",
        f"{PX}_PRISMA_savgol": f"{ap} PRISMA pixel",
        f"{PX}_ENMAP_ovrlp":   f"{ap} EnMAP pixel",
        f"{PX}_EMIT":          f"{ap} EMIT pixel",
    }


# Valori iniziali (primo pannello); main_all() li reimposta per ogni pannello.
SPECTRUM_GROUPS = make_groups(*PANELS[0])
DISPLAY_LABELS = make_labels(*PANELS[0])
PANEL_NAME = as_name(PANELS[0][0]).replace(" ", "_")   # es. AS_13 (nomi di cartelle e file)
ASD_TXT_DIR = R  # non usata: gli input sono espliciti


def build_spectrum_input_entries(groups: List[dict]) -> List[dict]:
    """
    Converte SPECTRUM_GROUPS in una lista ordinata di input con metadati espliciti.

    IMPORTANTE:
    Non uso più una lookup basata sul path per ricostruire il gruppo.
    In questo modo l'ordine e gli offset sono sempre quelli definiti in SPECTRUM_GROUPS:

        SOIA_13 -> ASD, EnMAP, PRISMA, EMIT
        SOIA_2  -> ASD, EnMAP, PRISMA, EMIT
        ecc.

    Questo evita il problema in cui, per differenze di path Windows/Path,
    gli spettri venivano trattati come gruppi separati e quindi impilati male.
    """
    entries: List[dict] = []

    for group_index, group in enumerate(groups):
        group_label = str(group.get("group_label", f"group_{group_index + 1}"))
        files = group.get("files", [])

        for within_group_index, file_str in enumerate(files):
            entries.append({
                "file": file_str,
                "group_label": group_label,
                "group_index": group_index,
                "within_group_index": within_group_index,
            })

    return entries


# Lista di input generata automaticamente dai blocchi.
# Lascia così se usi SPECTRUM_GROUPS.
SPECTRUM_INPUT_ENTRIES = build_spectrum_input_entries(SPECTRUM_GROUPS)
INPUT_SPECTRUM_FILES = [entry["file"] for entry in SPECTRUM_INPUT_ENTRIES]

# Modalità alternativa vecchia:
# se vuoi tornare alla lista piatta manuale, svuota SPECTRUM_GROUPS = []
# e inserisci i path direttamente qui sotto. In quel caso verrà usata la modalità piatta.
# INPUT_SPECTRUM_FILES = [
#     r"path/to/SOIA_13.txt",
#     r"path/to/SOIA_13_ENMAP_ovrlp.txt",
#     r"path/to/SOIA_13_PRISMA_savgol.txt",
#     r"path/to/SOIA_13_EMIT.txt",
# ]
# SPECTRUM_INPUT_ENTRIES = []
# Se True e INPUT_SPECTRUM_FILES è valorizzata,
# aggiunge in coda anche gli altri file trovati in ASD_TXT_DIR non già inclusi
APPEND_REMAINING_FROM_DIR = False

OUTPUT_DIR = f"{OUT_BASE}/{PANEL_NAME}"  # reimpostata per ogni pannello

WAVELENGTH_MIN = None
WAVELENGTH_MAX = None

OFFSET_STEP = 0.3          # distanza tra ASD ricampionato e pixel della stessa coppia
GROUP_OFFSET_STEP = 1.6    # distanza tra le coppie (PRISMA, EnMAP, EMIT)
SHOW_GROUP_LABELS = False   # True scrive il nome del blocco a sinistra del plot
X_UNITS = "um"  # "um" oppure "nm"

# ------------------------------------------------------------------
# MASK ASSORBIMENTI ATMOSFERICI
# ------------------------------------------------------------------
APPLY_ATM_ABSORPTION_MASK = True

ATM_MASK_RANGES_NM = [
    (1330.0, 1550.0),
    (1770.0, 2040.0),
]

# ------------------------------------------------------------------
# REGIONI INTERPRETATIVE SEMPLIFICATE
# ------------------------------------------------------------------
MACRO_REGIONS_NM = [
    ("VNIR",   350.0,  1350.0, "rgba(100, 149, 237, 0.05)"),
    ("ATM 1", 1350.0,  1550.0, "rgba(180, 180, 180, 0.16)"),
    ("SWIR 1",1550.0,  1880.0, "rgba(60, 179, 113, 0.05)"),
    ("ATM 2", 1880.0,  2040.0, "rgba(180, 180, 180, 0.16)"),
    ("SWIR 2",2040.0,  2450.0, "rgba(255, 165, 0, 0.05)"),
]

SIMPLE_DIAGNOSTIC_REGIONS_NM = [
    ("Water and OH", 1300.0, 1550.0, "rgba(30, 144, 255, 0.08)", "top left"),
    ("OH",           1720.0, 1860.0, "rgba(70, 130, 180, 0.08)", "top left"),
    ("Water",        1880.0, 2040.0, "rgba(30, 144, 255, 0.08)", "top left"),
    ("AlOH",         2160.0, 2220.0, "rgba(255, 0, 255, 0.08)",  "top right"),
    ("FeOH",         2230.0, 2296.0, "rgba(178, 34, 34, 0.08)",  "top right"),
    ("CO3",          2300.0, 2365.0, "rgba(255, 140, 0, 0.08)",  "top right"),
]

SIMPLE_GUIDE_LINES_NM = [
    1350.0, 1550.0, 1720.0, 1860.0, 1880.0, 2040.0, 2160.0, 2220.0, 2296.0, 2365.0
]

# ------------------------------------------------------------------
# VISIBILITÀ FASCE E LINEE  ← modifica qui
# ------------------------------------------------------------------
SHOW_MACRO_REGIONS            = False   # False: nasconde le fasce colorate di sfondo
SHOW_SIMPLE_DIAGNOSTIC_REGIONS = False  # False: nasconde le fasce diagnostiche (AlOH, CO3...)
SHOW_SIMPLE_GUIDE_LINES       = False   # linee verticali tratteggiate

# ------------------------------------------------------------------
# ASSI  ← modifica qui
# ------------------------------------------------------------------
# Mostra la riga continua dell'asse X (in basso) e dell'asse Y (a sinistra)
SHOW_AXIS_LINES  = True            # True = bordo continuo su X e Y
AXIS_LINE_COLOR  = "rgb(80,80,80)" # colore della riga degli assi
AXIS_LINE_WIDTH  = 1.5             # spessore in px
AXIS_TITLE_FONT_SIZE = 20    # dimensione label "Wavelength (µm)" e "Normalized reflectance..."
AXIS_TICK_FONT_SIZE  = 16    # dimensione numeri sui tick degli assi

# Usato solo se INPUT_SPECTRUM_FILES è vuota
PREFERRED_ORDER = [
    # "SOIA_5",
    # "SOIA_6",
]

FIG_WIDTH         = 1400
BASE_HEIGHT       = 220
HEIGHT_PER_SPECTRUM = 55
MIN_HEIGHT        = 950
MAX_HEIGHT        = 2600

SAVE_HTML          = True
HTML_FILENAME_FULL   = "Fig10_{panel}_FULL.html"     # {panel} -> AS_13, AS_2, AS_1_1, AS_1_2
HTML_FILENAME_MASKED = "Fig10_{panel}_MASKED.html"
XML_FILENAME         = "spectra_offset_overview_normalized.xml"

LINE_WIDTH = 1.7

# ------------------------------------------------------------------
# COLORI SPETTRI - MODIFICA FACILE
# ------------------------------------------------------------------
# Regola pratica:
# 1) Se vuoi controllare un singolo spettro, usa SPECTRUM_COLORS.
# 2) La chiave può essere:
#    - sample_id letto dentro il file, es. "SOIA_2"
#    - nome file senza estensione, es. "SOIA_2_PRISMA"
#    - nome file con estensione, es. "SOIA_2_PRISMA.txt"
# 3) Se più spettri hanno lo stesso sample_id, conviene usare il nome file.
SPECTRUM_COLORS = make_colors(*PANELS[0])

# Colori automatici per parole chiave nel nome file o nel sample_id.
# Viene usato solo se non trova un colore specifico in SPECTRUM_COLORS.
KEYWORD_COLORS = {
    "ENMAP": "#BE1075",
    "PRISMA": "#A16207",
    "EMIT": "#4B494D",
}

# Palette di fallback: usata quando lo spettro non è in SPECTRUM_COLORS
# e non contiene nessuna parola chiave in KEYWORD_COLORS.
COLORS = [
    "#6A0DAD", "#D97706", "#0EA5E9", "#16A34A", "#DC2626", "#7C3AED",
    "#EA580C", "#0891B2", "#65A30D", "#BE123C", "#4F46E5", "#A16207",
    "#0F766E", "#15803D", "#B91C1C", "#6D28D9"
]




def build_group_lookup(groups: List[dict]) -> Dict[str, Tuple[str, int, int]]:
    """
    Costruisce una mappa path -> metadati gruppo.
    Serve per sapere a quale blocco appartiene ogni file.
    """
    lookup: Dict[str, Tuple[str, int, int]] = {}

    for group_index, group in enumerate(groups):
        group_label = str(group.get("group_label", f"group_{group_index + 1}"))
        for within_index, file_str in enumerate(group.get("files", [])):
            # Chiave 1: path esatto come stringa
            lookup[str(file_str)] = (group_label, group_index, within_index)

            # Chiave 2: path normalizzato Path, utile su Windows
            try:
                lookup[str(Path(file_str))] = (group_label, group_index, within_index)
            except Exception:
                pass

            # Chiave 3: nome file semplice
            try:
                lookup[Path(file_str).name] = (group_label, group_index, within_index)
            except Exception:
                pass

    return lookup


GROUP_LOOKUP = build_group_lookup(SPECTRUM_GROUPS)
USE_GROUPED_INPUT_MODE = len(SPECTRUM_GROUPS) > 0


def get_group_metadata(file_path: Path, flat_index: int) -> Tuple[str, int, int]:
    """
    Recupera group_label, group_index e indice interno al gruppo.
    Se SPECTRUM_GROUPS è vuoto, mantiene una modalità piatta compatibile
    con il codice precedente.
    """
    if not USE_GROUPED_INPUT_MODE:
        return "", flat_index, 0

    keys = [
        str(file_path),
        str(Path(file_path)),
        file_path.name,
    ]

    for key in keys:
        if key in GROUP_LOOKUP:
            return GROUP_LOOKUP[key]

    # Fallback prudente: se non trova il path nella lookup,
    # lo mette in un blocco separato seguendo l'ordine di lettura.
    return f"ungrouped_{flat_index + 1}", flat_index, 0


def compute_vertical_offset(sp: "Spectrum", flat_index: int) -> float:
    """
    Offset finale del tracciato.
    In modalità a blocchi:
        offset = offset_del_blocco + offset_interno_al_blocco
    In modalità piatta:
        offset = indice * OFFSET_STEP, come nel codice originale.
    """
    if USE_GROUPED_INPUT_MODE:
        return (sp.group_index * GROUP_OFFSET_STEP) + (sp.within_group_index * OFFSET_STEP)
    return flat_index * OFFSET_STEP


@dataclass
class Spectrum:
    sample_id: str
    wavelength: np.ndarray
    reflectance: np.ndarray
    source_file: str
    group_label: str = ""
    group_index: int = 0
    within_group_index: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_separator(header_line: str) -> Optional[str]:
    if "\t" in header_line:
        return "\t"
    if ";" in header_line:
        return ";"
    if "," in header_line:
        return ","
    return None


def clean_reflectance(y: np.ndarray) -> np.ndarray:
    y = np.array(y, dtype=float)  # copia: evita errori "read-only" con pandas recenti
    y[~np.isfinite(y)] = np.nan
    y[y < -1e20] = np.nan
    return y


def normalize_0_1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(y, np.nan, dtype=float)
    ymin = np.nanmin(y[finite])
    ymax = np.nanmax(y[finite])
    if not np.isfinite(ymin) or not np.isfinite(ymax) or np.isclose(ymin, ymax):
        return np.full_like(y, np.nan, dtype=float)
    out = (y - ymin) / (ymax - ymin)
    out[~finite] = np.nan
    return out


def apply_wavelength_range(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    if WAVELENGTH_MIN is not None:
        mask &= x >= WAVELENGTH_MIN
    if WAVELENGTH_MAX is not None:
        mask &= x <= WAVELENGTH_MAX
    return x[mask], y[mask]


def apply_atm_absorption_mask(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    if not APPLY_ATM_ABSORPTION_MASK:
        return x, y
    for lo, hi in ATM_MASK_RANGES_NM:
        mask = (x >= lo) & (x <= hi)
        y[mask] = np.nan
    return x, y


def sort_unique_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    df = df.groupby("x", as_index=False)["y"].mean().sort_values("x")
    return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def order_sample_ids(sample_ids: List[str]) -> List[str]:
    preferred = [sid for sid in PREFERRED_ORDER if sid in sample_ids]
    remaining = sorted([sid for sid in sample_ids if sid not in preferred])
    return preferred + remaining


def convert_x_units(x_nm: np.ndarray) -> np.ndarray:
    return x_nm / 1000.0 if X_UNITS == "um" else x_nm


def convert_single_x(value_nm: float) -> float:
    return value_nm / 1000.0 if X_UNITS == "um" else value_nm


def get_x_label() -> str:
    return "Wavelength (µm)" if X_UNITS == "um" else "Wavelength (nm)"


def get_spectrum_color(sp: Spectrum, index: int) -> str:
    """
    Priorità colore:
    1. chiave esatta in SPECTRUM_COLORS (sample_id, file stem, file name)
    2. parola chiave in KEYWORD_COLORS
    3. palette automatica COLORS
    """
    file_path = Path(sp.source_file)
    file_stem = file_path.stem
    file_name = file_path.name

    # Prima controllo il nome file, poi il sample_id.
    # Questo è importante perché più sensori possono avere lo stesso sample_id
    # ma file diversi, ad esempio SOIA_13, SOIA_13_PRISMA_savgol, SOIA_13_EMIT.
    for key in [file_stem, file_name, sp.sample_id]:
        if key in SPECTRUM_COLORS:
            return SPECTRUM_COLORS[key]

    haystack = f"{sp.sample_id} {file_stem} {file_name}".lower()
    for keyword, color in KEYWORD_COLORS.items():
        if keyword.lower() in haystack:
            return color

    return COLORS[index % len(COLORS)]


def get_macro_regions_for_plot() -> List[Tuple[str, float, float, str]]:
    return [(name, convert_single_x(lo), convert_single_x(hi), color)
            for name, lo, hi, color in MACRO_REGIONS_NM]


def get_simple_diagnostic_regions_for_plot() -> List[Tuple[str, float, float, str, str]]:
    return [(name, convert_single_x(lo), convert_single_x(hi), color, pos)
            for name, lo, hi, color, pos in SIMPLE_DIAGNOSTIC_REGIONS_NM]


def get_simple_guide_lines_for_plot() -> List[float]:
    return [convert_single_x(x) for x in SIMPLE_GUIDE_LINES_NM]


def compute_fig_height(n_spectra: int) -> int:
    h = BASE_HEIGHT + n_spectra * HEIGHT_PER_SPECTRUM
    return int(max(MIN_HEIGHT, min(MAX_HEIGHT, h)))


def read_spectrum_txt(path: Path) -> Spectrum:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    sep = detect_separator(first_line)
    if sep is not None:
        df = pd.read_csv(path, sep=sep, engine="python")
    else:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    cols = {c.strip().lower(): c for c in df.columns}
    required = ["sample_id", "wavelength", "reflectance"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{path.name}: colonne mancanti {missing}. Colonne trovate: {list(df.columns)}")

    sample_col = cols["sample_id"]
    wl_col     = cols["wavelength"]
    refl_col   = cols["reflectance"]

    sample_ids = df[sample_col].dropna().astype(str).unique().tolist()
    sample_id  = sample_ids[0] if sample_ids else path.stem

    x = pd.to_numeric(df[wl_col],   errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[refl_col], errors="coerce").to_numpy(dtype=float)

    y = clean_reflectance(y)
    x, y = apply_wavelength_range(x, y)
    x, y = sort_unique_xy(x, y)

    if len(x) < 2:
        raise ValueError(f"{path.name}: spettro non valido o troppo corto dopo la pulizia.")

    return Spectrum(sample_id=str(sample_id), wavelength=x, reflectance=y, source_file=str(path))


def list_candidate_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory non trovata: {directory}")
    return sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in [".txt", ".asc", ".dat", ".csv"]
    ])


def read_spectra_from_dir(directory: Path) -> Dict[str, Spectrum]:
    txt_files = list_candidate_files(directory)
    if not txt_files:
        raise FileNotFoundError(f"Nessun file TXT/ASC/DAT/CSV trovato in: {directory}")

    spectra: Dict[str, Spectrum] = {}
    errors = []
    for p in txt_files:
        try:
            sp = read_spectrum_txt(p)
            spectra[sp.sample_id] = sp
        except Exception as e:
            errors.append(f"{p.name}: {e}")

    if not spectra:
        raise RuntimeError("Nessuno spettro valido caricato.\n" + "\n".join(errors))
    if errors:
        print("ATTENZIONE: alcuni file non sono stati letti:")
        for e in errors:
            print(" -", e)
    return spectra


def build_ordered_spectra_from_file_entries(
    input_entries: List[dict],
    fallback_dir: Optional[Path] = None,
    append_remaining_from_dir: bool = False
) -> Tuple[List[Spectrum], List[str]]:
    """
    Legge gli spettri usando metadati di gruppo già espliciti.

    Questa è la funzione corretta per la modalità a blocchi:
    non deve ricostruire il gruppo dal path, perché quello era il punto fragile.
    """
    spectra_ordered: List[Spectrum] = []
    errors: List[str] = []
    used_paths = set()

    print("\n--- DEBUG LETTURA FILE ESPLICITI A BLOCCHI ---")
    print(f"Numero input in SPECTRUM_INPUT_ENTRIES: {len(input_entries)}")

    for i, entry in enumerate(input_entries, start=1):
        file_str = entry["file"]
        p = Path(file_str)

        group_label = str(entry.get("group_label", ""))
        group_index = int(entry.get("group_index", i - 1))
        within_group_index = int(entry.get("within_group_index", 0))

        print(f"\n[{i}/{len(input_entries)}] TENTO: {p}")
        print(f"  gruppo atteso: {group_label} | group_index={group_index} | within_group_index={within_group_index}")

        used_paths.add(str(p.resolve()) if p.exists() else str(p))

        try:
            if not p.exists():
                raise FileNotFoundError(f"File non trovato: {p}")

            sp = read_spectrum_txt(p)

            # Metadati presi direttamente da SPECTRUM_GROUPS,
            # non ricostruiti dal path.
            sp.group_label = group_label
            sp.group_index = group_index
            sp.within_group_index = within_group_index

            print(f"  OK: {p.name}")
            print(f"      sample_id letto dal file: {sp.sample_id}")
            print(f"      group finale: {sp.group_label} | group_index={sp.group_index} | within_group_index={sp.within_group_index}")
            print(f"      offset finale: {compute_vertical_offset(sp, i - 1):.2f}")
            print(f"      n_points: {len(sp.wavelength)}")
            print(f"      wl_min: {float(np.nanmin(sp.wavelength))}")
            print(f"      wl_max: {float(np.nanmax(sp.wavelength))}")

            spectra_ordered.append(sp)

        except Exception as e:
            print(f"  ERRORE su {p.name}: {e}")
            errors.append(f"{p.name}: {e}")

    if append_remaining_from_dir and fallback_dir is not None:
        print("\n--- DEBUG APPEND DA DIRECTORY ---")
        try:
            dir_files = list_candidate_files(fallback_dir)
            print(f"File trovati in fallback_dir: {len(dir_files)}")
            for p in dir_files:
                key = str(p.resolve())
                if key in used_paths:
                    print(f"  SKIP già presente: {p.name}")
                    continue
                try:
                    sp = read_spectrum_txt(p)
                    sp.group_label = p.stem
                    sp.group_index = len(SPECTRUM_GROUPS) + len(spectra_ordered)
                    sp.within_group_index = 0
                    print(f"  APPESO: {p.name} -> {len(sp.wavelength)} punti")
                    spectra_ordered.append(sp)
                except Exception as e:
                    print(f"  ERRORE append {p.name}: {e}")
                    errors.append(f"{p.name}: {e}")
        except Exception as e:
            print(f"ERRORE lettura directory fallback {fallback_dir}: {e}")
            errors.append(f"Errore lettura directory fallback {fallback_dir}: {e}")

    if not spectra_ordered:
        raise RuntimeError("Nessuno spettro valido caricato dalla lista di file specificata.")

    return spectra_ordered, errors


def build_ordered_spectra_from_file_list(
    input_files: List[str],
    fallback_dir: Optional[Path] = None,
    append_remaining_from_dir: bool = False
) -> Tuple[List[Spectrum], List[str]]:
    """
    Modalità piatta di compatibilità con il vecchio codice.
    Usala solo se SPECTRUM_GROUPS è vuoto.
    """
    input_entries = [
        {
            "file": file_str,
            "group_label": "",
            "group_index": i,
            "within_group_index": 0,
        }
        for i, file_str in enumerate(input_files)
    ]
    return build_ordered_spectra_from_file_entries(
        input_entries=input_entries,
        fallback_dir=fallback_dir,
        append_remaining_from_dir=append_remaining_from_dir,
    )


def add_macro_regions(fig: go.Figure) -> None:
    for name, lo, hi, color in get_macro_regions_for_plot():
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=1.0, line_width=0)


def add_simple_diagnostic_regions(fig: go.Figure) -> None:
    for name, lo, hi, color, ann_pos in get_simple_diagnostic_regions_for_plot():
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=color,
            opacity=1.0,
            line_width=0,
            annotation_text=name,
            annotation_position=ann_pos,
            annotation=dict(font=dict(size=15, color="black")),
        )


def add_simple_guide_lines(fig: go.Figure) -> None:
    for x in get_simple_guide_lines_for_plot():
        fig.add_vline(
            x=x,
            line_width=0.8,
            line_dash="dot",
            line_color="rgba(80,80,80,0.45)",
        )


def build_offset_plot(spectra_ordered: List[Spectrum], use_atm_mask: bool) -> go.Figure:
    fig = go.Figure()
    fig_height = compute_fig_height(len(spectra_ordered))

    group_label_positions: Dict[str, float] = {}

    for i, sp in enumerate(spectra_ordered):
        color  = get_spectrum_color(sp, i)
        offset = compute_vertical_offset(sp, i)
        label  = DISPLAY_LABELS.get(Path(sp.source_file).stem, as_name(sp.sample_id))

        group_info = f" | group={sp.group_label}" if sp.group_label else ""
        print(f"  Trace {i+1}/{len(spectra_ordered)}: {sp.sample_id}{group_info} | offset={offset:.2f}")

        if sp.group_label and sp.group_label not in group_label_positions:
            # Posizione indicativa del blocco: base del blocco + metà ampiezza interna.
            # Con 4 spettri per blocco, il centro visivo è circa 1.5 * OFFSET_STEP sopra la base.
            group_label_positions[sp.group_label] = (sp.group_index * GROUP_OFFSET_STEP) + (1.5 * OFFSET_STEP)

        if use_atm_mask:
            x_used, y_used = apply_atm_absorption_mask(sp.wavelength, sp.reflectance)
        else:
            x_used, y_used = sp.wavelength.copy(), sp.reflectance.copy()

        y_norm = normalize_0_1(y_used)
        y_plot = y_norm + offset
        x_plot = convert_x_units(x_used)

        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            name=label,
            line=dict(color=color, width=LINE_WIDTH),
            hovertemplate=(
                f"{label}"
                f"<br>group={sp.group_label if sp.group_label else 'flat'}"
                "<br>wavelength=%{x:.4f}"
                "<br>normalized+offset=%{y:.6f}"
                f"<br>offset={offset:.2f}<extra></extra>"
            ),
        ))

        finite = np.isfinite(x_plot) & np.isfinite(y_plot)
        if np.any(finite):
            x_last = x_plot[finite][-1]
            y_last = y_plot[finite][-1]
            fig.add_annotation(
                x=x_last, y=y_last,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=10,
                font=dict(size=20, color=color),
            )

    if SHOW_GROUP_LABELS and USE_GROUPED_INPUT_MODE:
        for group_label, y_pos in group_label_positions.items():
            fig.add_annotation(
                xref="paper",
                x=-0.055,
                y=y_pos,
                text=f"<b>{group_label}</b>",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=22, color="black"),
            )

    if SHOW_MACRO_REGIONS:
        add_macro_regions(fig)

    if SHOW_SIMPLE_DIAGNOSTIC_REGIONS:
        add_simple_diagnostic_regions(fig)

    if SHOW_SIMPLE_GUIDE_LINES:
        add_simple_guide_lines(fig)

    title_suffix = "con mask atmosferica" if use_atm_mask else "spettro completo"

    fig.update_layout(
        title=f"Overview spettri con offset verticale (normalizzati 0–1) - {title_suffix}",
        width=FIG_WIDTH,
        height=fig_height,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            itemsizing="constant",
            font=dict(size=20),
        ),
        margin=dict(l=80, r=340, t=110, b=80),
    )

    # ── ASSI: griglia + riga continua di bordo ────────────────────────────────
    fig.update_xaxes(
        title_text=get_x_label(),
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=AXIS_TICK_FONT_SIZE),
        showgrid=True,
        showline=SHOW_AXIS_LINES,
        linecolor=AXIS_LINE_COLOR,
        linewidth=AXIS_LINE_WIDTH,
        mirror=False,          # riga solo sul lato inferiore
    )
    fig.update_yaxes(
        title_text="Normalized reflectance (0–1) + vertical offset",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=AXIS_TICK_FONT_SIZE),
        showgrid=True,
        showline=SHOW_AXIS_LINES,
        linecolor=AXIS_LINE_COLOR,
        linewidth=AXIS_LINE_WIDTH,
        mirror=False,          # riga solo sul lato sinistro
    )

    return fig


def save_html(fig: go.Figure, html_path: Path) -> None:
    if SAVE_HTML:
        print(f"  -> Scrittura file HTML: {html_path}")
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            config={"toImageButtonOptions": {
                "format": "png",
                "filename": f"Fig10_{PANEL_NAME}_{html_path.stem.split('_')[-1]}",
                "scale": 3,
            }},
        )
        print(f"  -> Scrittura completata: {html_path.name}")


def write_run_summary_xml(xml_path: Path, spectra_ordered: List[Spectrum], errors: List[str], input_mode: str) -> None:
    root = ET.Element("run_summary")
    ET.SubElement(root, "timestamp").text = datetime.now().isoformat()

    config = ET.SubElement(root, "config")
    ET.SubElement(config, "asd_txt_dir").text = ASD_TXT_DIR
    ET.SubElement(config, "output_dir").text = OUTPUT_DIR
    ET.SubElement(config, "input_mode").text = input_mode
    ET.SubElement(config, "append_remaining_from_dir").text = str(APPEND_REMAINING_FROM_DIR)
    ET.SubElement(config, "wavelength_min").text = str(WAVELENGTH_MIN)
    ET.SubElement(config, "wavelength_max").text = str(WAVELENGTH_MAX)
    ET.SubElement(config, "offset_step").text = str(OFFSET_STEP)
    ET.SubElement(config, "group_offset_step").text = str(GROUP_OFFSET_STEP)
    ET.SubElement(config, "use_grouped_input_mode").text = str(USE_GROUPED_INPUT_MODE)
    ET.SubElement(config, "show_group_labels").text = str(SHOW_GROUP_LABELS)
    ET.SubElement(config, "x_units").text = X_UNITS
    ET.SubElement(config, "normalized").text = "True"
    ET.SubElement(config, "apply_atm_absorption_mask").text = str(APPLY_ATM_ABSORPTION_MASK)

    atm_mask_node = ET.SubElement(config, "atm_mask_ranges_nm")
    for lo, hi in ATM_MASK_RANGES_NM:
        r = ET.SubElement(atm_mask_node, "range")
        ET.SubElement(r, "min_nm").text = str(lo)
        ET.SubElement(r, "max_nm").text = str(hi)

    ET.SubElement(config, "fig_width").text = str(FIG_WIDTH)
    ET.SubElement(config, "base_height").text = str(BASE_HEIGHT)
    ET.SubElement(config, "height_per_spectrum").text = str(HEIGHT_PER_SPECTRUM)
    ET.SubElement(config, "show_macro_regions").text = str(SHOW_MACRO_REGIONS)
    ET.SubElement(config, "show_simple_diagnostic_regions").text = str(SHOW_SIMPLE_DIAGNOSTIC_REGIONS)
    ET.SubElement(config, "show_simple_guide_lines").text = str(SHOW_SIMPLE_GUIDE_LINES)
    ET.SubElement(config, "show_axis_lines").text = str(SHOW_AXIS_LINES)
    ET.SubElement(config, "axis_line_color").text = AXIS_LINE_COLOR
    ET.SubElement(config, "axis_line_width").text = str(AXIS_LINE_WIDTH)

    spectrum_colors_node = ET.SubElement(config, "spectrum_colors")
    for key, color in SPECTRUM_COLORS.items():
        item = ET.SubElement(spectrum_colors_node, "color")
        ET.SubElement(item, "key").text = str(key)
        ET.SubElement(item, "value").text = str(color)

    keyword_colors_node = ET.SubElement(config, "keyword_colors")
    for key, color in KEYWORD_COLORS.items():
        item = ET.SubElement(keyword_colors_node, "color")
        ET.SubElement(item, "key").text = str(key)
        ET.SubElement(item, "value").text = str(color)

    file_list_node = ET.SubElement(config, "input_spectrum_files")
    for fp in INPUT_SPECTRUM_FILES:
        ET.SubElement(file_list_node, "file").text = fp

    groups_node = ET.SubElement(config, "spectrum_groups")
    for gi, group in enumerate(SPECTRUM_GROUPS):
        g = ET.SubElement(groups_node, "group")
        ET.SubElement(g, "group_index").text = str(gi)
        ET.SubElement(g, "group_label").text = str(group.get("group_label", f"group_{gi + 1}"))
        files_node = ET.SubElement(g, "files")
        for fp in group.get("files", []):
            ET.SubElement(files_node, "file").text = fp

    outputs_node = ET.SubElement(config, "html_outputs")
    ET.SubElement(outputs_node, "full").text = HTML_FILENAME_FULL
    ET.SubElement(outputs_node, "masked").text = HTML_FILENAME_MASKED

    inputs = ET.SubElement(root, "inputs")
    ET.SubElement(inputs, "spectra_count").text = str(len(spectra_ordered))

    node = ET.SubElement(inputs, "spectra")
    for i, sp in enumerate(spectra_ordered):
        s = ET.SubElement(node, "spectrum")
        ET.SubElement(s, "order_index").text = str(i + 1)
        ET.SubElement(s, "sample_id").text = sp.sample_id
        ET.SubElement(s, "group_label").text = sp.group_label
        ET.SubElement(s, "group_index").text = str(sp.group_index)
        ET.SubElement(s, "within_group_index").text = str(sp.within_group_index)
        ET.SubElement(s, "source_file").text = sp.source_file
        ET.SubElement(s, "n_points").text = str(len(sp.wavelength))
        ET.SubElement(s, "wl_min_nm").text = str(float(np.nanmin(sp.wavelength)))
        ET.SubElement(s, "wl_max_nm").text = str(float(np.nanmax(sp.wavelength)))
        ET.SubElement(s, "offset").text = str(float(compute_vertical_offset(sp, i)))

    macro_node = ET.SubElement(root, "macro_regions_nm")
    for name, lo, hi, _ in MACRO_REGIONS_NM:
        r = ET.SubElement(macro_node, "region")
        ET.SubElement(r, "name").text = name
        ET.SubElement(r, "min_nm").text = str(lo)
        ET.SubElement(r, "max_nm").text = str(hi)

    diag_node = ET.SubElement(root, "simple_diagnostic_regions_nm")
    for name, lo, hi, _, pos in SIMPLE_DIAGNOSTIC_REGIONS_NM:
        r = ET.SubElement(diag_node, "region")
        ET.SubElement(r, "name").text = name
        ET.SubElement(r, "min_nm").text = str(lo)
        ET.SubElement(r, "max_nm").text = str(hi)
        ET.SubElement(r, "annotation_position").text = pos

    line_node = ET.SubElement(root, "simple_guide_lines_nm")
    for x in SIMPLE_GUIDE_LINES_NM:
        r = ET.SubElement(line_node, "line")
        ET.SubElement(r, "x_nm").text = str(x)

    err_node = ET.SubElement(root, "errors")
    for e in errors:
        ET.SubElement(err_node, "error").text = e

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    errors: List[str] = []

    if INPUT_SPECTRUM_FILES:
        if SPECTRUM_INPUT_ENTRIES:
            print("Lettura spettri da blocchi espliciti...")
            spectra_ordered, read_errors = build_ordered_spectra_from_file_entries(
                input_entries=SPECTRUM_INPUT_ENTRIES,
                fallback_dir=Path(ASD_TXT_DIR),
                append_remaining_from_dir=APPEND_REMAINING_FROM_DIR,
            )
            input_mode = "explicit_grouped_blocks"
        else:
            print("Lettura spettri da lista file esplicita piatta...")
            spectra_ordered, read_errors = build_ordered_spectra_from_file_list(
                input_files=INPUT_SPECTRUM_FILES,
                fallback_dir=Path(ASD_TXT_DIR),
                append_remaining_from_dir=APPEND_REMAINING_FROM_DIR,
            )
            input_mode = "explicit_file_list"

        errors.extend(read_errors)
    else:
        print("Lettura spettri da directory...")
        spectra = read_spectra_from_dir(Path(ASD_TXT_DIR))
        ordered_ids = order_sample_ids(list(spectra.keys()))
        spectra_ordered = [spectra[sid] for sid in ordered_ids]
        input_mode = "directory_scan"
        print(f"\nSpettri caricati: {len(spectra_ordered)}")

    if errors:
        print("\n--- ERRORI/WARNING DI LETTURA ---")
        for e in errors:
            print(" -", e)
    else:
        print("\nNessun errore di lettura file.")

    print("\n--- FASE PLOT FULL ---")
    print("Costruisco fig_full...")
    fig_full = build_offset_plot(spectra_ordered, use_atm_mask=False)
    print("Salvo HTML full...")
    save_html(fig_full, out_dir / HTML_FILENAME_FULL.format(panel=PANEL_NAME))
    print("HTML full salvato.")

    print("\n--- FASE PLOT MASKED ---")
    print("Costruisco fig_masked...")
    fig_masked = build_offset_plot(spectra_ordered, use_atm_mask=True)
    print("Salvo HTML masked...")
    save_html(fig_masked, out_dir / HTML_FILENAME_MASKED.format(panel=PANEL_NAME))
    print("HTML masked salvato.")

    print("\n--- FASE XML ---")
    write_run_summary_xml(out_dir / XML_FILENAME, spectra_ordered, errors, input_mode)
    print("XML scritto.")

    with open(out_dir / "run_info.txt", "w", encoding="utf-8") as f:
        f.write("7_plot_asd_samples_offset_html.py\n")
        f.write("Versione: HTML con spettri in offset, ordine controllabile da lista file esplicita\n")
        f.write("Output doppi: spettro completo + spettro con mask atmosferica\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"ASD_TXT_DIR: {ASD_TXT_DIR}\n")
        f.write(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
        f.write(f"INPUT_MODE: {input_mode}\n")
        f.write(f"APPEND_REMAINING_FROM_DIR: {APPEND_REMAINING_FROM_DIR}\n")
        f.write(f"Spettri caricati: {len(spectra_ordered)}\n")
        f.write(f"OFFSET_STEP: {OFFSET_STEP}\n")
        f.write(f"GROUP_OFFSET_STEP: {GROUP_OFFSET_STEP}\n")
        f.write(f"USE_GROUPED_INPUT_MODE: {USE_GROUPED_INPUT_MODE}\n")
        f.write(f"SHOW_GROUP_LABELS: {SHOW_GROUP_LABELS}\n")
        f.write(f"X_UNITS: {X_UNITS}\n")
        f.write(f"APPLY_ATM_ABSORPTION_MASK: {APPLY_ATM_ABSORPTION_MASK}\n")
        f.write(f"ATM_MASK_RANGES_NM: {ATM_MASK_RANGES_NM}\n")
        f.write(f"FIG_WIDTH: {FIG_WIDTH}\n")
        f.write(f"BASE_HEIGHT: {BASE_HEIGHT}\n")
        f.write(f"HEIGHT_PER_SPECTRUM: {HEIGHT_PER_SPECTRUM}\n")
        f.write(f"SHOW_MACRO_REGIONS: {SHOW_MACRO_REGIONS}\n")
        f.write(f"SHOW_SIMPLE_DIAGNOSTIC_REGIONS: {SHOW_SIMPLE_DIAGNOSTIC_REGIONS}\n")
        f.write(f"SHOW_SIMPLE_GUIDE_LINES: {SHOW_SIMPLE_GUIDE_LINES}\n")
        f.write(f"SHOW_AXIS_LINES: {SHOW_AXIS_LINES}\n")
        f.write(f"AXIS_LINE_COLOR: {AXIS_LINE_COLOR}\n")
        f.write(f"AXIS_LINE_WIDTH: {AXIS_LINE_WIDTH}\n")
        f.write(f"SPECTRUM_COLORS: {SPECTRUM_COLORS}\n")
        f.write(f"KEYWORD_COLORS: {KEYWORD_COLORS}\n")
        f.write(f"HTML_FILENAME_FULL: {HTML_FILENAME_FULL}\n")
        f.write(f"HTML_FILENAME_MASKED: {HTML_FILENAME_MASKED}\n")
        f.write("\nINPUT_SPECTRUM_FILES:\n")
        if INPUT_SPECTRUM_FILES:
            for p in INPUT_SPECTRUM_FILES:
                f.write(f"- {p}\n")
        else:
            f.write("- (vuota: uso ASD_TXT_DIR)\n")
        if errors:
            f.write("\nERRORS/WARNINGS:\n")
            for e in errors:
                f.write(f"- {e}\n")

    print("\nCompletato.")
    print(f"Output FULL:   {out_dir / HTML_FILENAME_FULL.format(panel=PANEL_NAME)}")
    print(f"Output MASKED: {out_dir / HTML_FILENAME_MASKED.format(panel=PANEL_NAME)}")


def check_panel_files(groups: List[dict]) -> List[str]:
    missing = []
    for g in groups:
        for f in g["files"]:
            if not Path(f).exists():
                missing.append(f)
    return missing


def main_all() -> None:
    global SPECTRUM_GROUPS, SPECTRUM_INPUT_ENTRIES, INPUT_SPECTRUM_FILES
    global GROUP_LOOKUP, USE_GROUPED_INPUT_MODE, OUTPUT_DIR, SPECTRUM_COLORS
    global DISPLAY_LABELS, PANEL_NAME

    for S, PX in PANELS:
        print("\n" + "=" * 70)
        print(f"PANNELLO: {as_name(S)}  (file ASD {S}  |  pixel {PX})")
        print("=" * 70)

        SPECTRUM_GROUPS = make_groups(S, PX)
        missing = check_panel_files(SPECTRUM_GROUPS)
        if missing:
            print("FILE MANCANTI, pannello saltato:")
            for m in missing:
                print("  -", m)
            print("Controlla R e P in cima allo script, oppure cerca i file con:")
            print('  find "path/to/MASPAG_2025" -name "' + PX + '_*.txt"')
            continue

        SPECTRUM_INPUT_ENTRIES = build_spectrum_input_entries(SPECTRUM_GROUPS)
        INPUT_SPECTRUM_FILES = [e["file"] for e in SPECTRUM_INPUT_ENTRIES]
        GROUP_LOOKUP = build_group_lookup(SPECTRUM_GROUPS)
        USE_GROUPED_INPUT_MODE = True
        SPECTRUM_COLORS = make_colors(S, PX)
        DISPLAY_LABELS = make_labels(S, PX)
        PANEL_NAME = as_name(S).replace(" ", "_")
        OUTPUT_DIR = f"{OUT_BASE}/{PANEL_NAME}"

        main()


if __name__ == "__main__":
    main_all()