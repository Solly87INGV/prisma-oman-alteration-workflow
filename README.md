# prisma-oman-alteration-workflow

Python scripts used for PRISMA-based pre-survey spectral screening, generation of the Spectral
Alteration Priority Mask (SAPM), and post-survey comparison of laboratory (ASD) and satellite
(PRISMA, EMIT, EnMAP) spectra in the Wadi al-Maʿawil sector of the Semail Ophiolite
(western Hajar, northern Oman), MASPAG 2025 field campaign.

- Associated manuscript: [to be added after acceptance]
- Data (samples, pXRF, ASD spectra, pixel spectra, SAPM): https://doi.org/10.5281/zenodo.24495820

The SAPM is an operational prioritization layer for field targeting and should not be
interpreted as a mineral map.

## Repository structure

| Folder | Script | Purpose |
|---|---|---|
| `code/1_SAPM/` | `hyperspectral_SAPM_analysis.py` | Spectral Angle Mapper on the PRISMA L2D cube with the reference library in `spectral_library/`; produces the SAPM (target mask), confidence (1 − angle/90°) and best-match mineral map (Figures 3 and 5). The script also writes additional layers (alteration mask, prospectivity, certainty map) that are not used in the paper. |
| | `unify_sapm.py` | Merges the SAPM masks shown in the figures into the single binary GeoTIFF (`SAPM.tif`, 0/255) published on Zenodo. |
| `code/2_pixel_extraction/` | `1_extract_prisma_spectra.py` | Extracts the PRISMA spectrum of the pixel containing each sample location. |
| | `2_PRISMA_Savitzky-Golay.py` | Savitzky–Golay smoothing (window 7, polynomial order 2) of the PRISMA pixel spectra used in Figures 10 and 11 (`*_PRISMA_savgol.txt`). |
| | `estractEMIT_RFL.py` | Extracts EMIT L2A reflectance spectra (netCDF) at the sample locations. |
| | `1_1_extract_EnMAP_spectra.py`, `1_2_EnMAP_daCSV_a_txt.py`, `1_3_Clean_Spec_EnMAP.py` | Extract EnMAP L2A pixel spectra, convert them to one text file per sample, and mask the VNIR–SWIR overlap bands (`*_ENMAP_ovrlp.txt`). |
| `code/3_ASD_resampling/` | `resample_ASD_to_sensors.py` | Resamples the 1 nm ASD laboratory spectra to the PRISMA, EMIT and EnMAP band centres by linear interpolation (no spectral response functions). |
| `code/4_figures/` | `7_plot_asd_samples_offset_COLOR.py` | Laboratory and reference spectra, normalized with vertical offset (Figures 7, 8 and 9). |
| | `7_plot_fig10_multisensor.py` | Resampled ASD spectra vs. PRISMA, EnMAP and EMIT pixel spectra (Figures 10 and 11). |
| `spectral_library/` | — | USGS Spectral Library Version 7 reference spectra used for the SAM screening (Appendix A of the paper). |

## Input data

- PRISMA L2D scene `PRS_L2D_STD_20220609065219_20220609065223_0001` (9 June 2022; cube
  `VNIR_SWIR_latlon_219`), available from the Italian Space Agency (ASI).
- EMIT L2A granule `EMIT_L2A_RFL_001_20260216T085347_2604706_019` (16 February 2026), NASA LP DAAC.
- EnMAP L2A scene `ENMAP01-____L2A-DT0000128885_20250503T072700Z_003_V010506_20260422T074655Z`
  (3 May 2025), DLR.
- Sample locations, ASD spectra and pixel spectra: Zenodo record above.

Satellite scenes are not redistributed.

## Usage

```
pip install -r requirements.txt
```

GDAL Python bindings (`osgeo`) are needed only by `unify_sapm.py` (see `requirements.txt`).
Each script has a configuration block at the top: replace the `path/to/...` placeholders with
local paths and run it with `python3 <script>.py`. The scripts were written for this study and
may need minor adaptation before reuse.

SAPM parameters used in the paper (`hyperspectral_SAPM_analysis.py`): `SUBSET_MASK_MODE = "all"`,
`SUBSET_PERCENTILE = 0.10`, `MIN_CONF_SUBSET = 0.70`.

## Sample naming

Files and scripts use the field codes: `SOIA_n` corresponds to `AS n` (alteration samples) and
`SOR_n` to `HR n` (reference samples) in the paper.

## Authors

Marco Solinas (Istituto Nazionale di Geofisica e Vulcanologia, INGV) and
Ana Sofia Meneses Pineda (Sapienza University of Rome).

## License

MIT License (see `LICENSE`).
