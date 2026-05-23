prisma-oman-alteration-workflow

PRISMA-based workflow for pre-survey spectral screening, SAPM generation, and post-survey sample-oriented analysis in the Oman ophiolite.

Overview

This repository contains custom Python scripts and supporting spectral libraries used in a PRISMA VNIR-SWIR workflow developed for alteration-related surface pattern analysis in the Wadi al-Maʿawil sector, northern Oman.

The workflow was conceived as a reproducible two-stage approach:

Pre-survey screening
generation of quicklook spectral products;
selective SAM-based screening;
production of the Spectral Alteration Priority Mask (SAPM), used as an operational targeting layer during field survey.
Post-survey analysis
organization and comparison of selected sample-related spectral information;
support for laboratory ASD FieldSpec interpretation;
comparison between field/laboratory evidence and satellite-derived spectral responses.

The workflow is intended as a screening and interpretative support approach, not as a deterministic mineral identification system.

Repository structure
code/

Python scripts used in the workflow.

1_quicklook_index/

Generation of quicklook spectral products such as indices and band-depth style outputs from the PRISMA clean cube.

2_SAM/
selective Spectral Angle Mapper (SAM) screening;
generation of the Spectral Alteration Priority Mask (SAPM) and associated diagnostic outputs.
3_postsurvey/

Scripts and utilities used for post-survey spectral organization, sample-oriented analysis, and comparison workflows.

4_roi_comparison/

Raster calculations and ROI-based statistics used to compare selected areas and support interpretation of SAPM-positive versus background sectors.

metadata/USGS_spectral_library/

Supporting spectral libraries and metadata used in the workflow.

These folders contain selected USGS-derived reference spectra and supporting files organized according to their interpretative role in the screening and post-survey comparison steps.

Main workflow logic

The workflow follows these general steps:

Start from a georeferenced and spectrally cleaned PRISMA L2D VNIR-SWIR cube;
Generate quicklook products for preliminary scene inspection;
Apply selective SAM screening to derive the Spectral Alteration Priority Mask (SAPM);
Use the SAPM to support field targeting and sample collection;
Compare selected field samples with laboratory and reference spectral information;
Support interpretation through qualitative comparison with ASD FieldSpec laboratory spectra and other complementary datasets.
Notes
The scripts were developed for the specific study presented in the associated manuscript and may require path adaptation and minor adjustments before reuse.
The repository is organized to keep the workflow lightweight, transparent, and reproducible rather than fully automated as a standalone package.
Outputs should be interpreted conservatively, especially in arid ophiolitic settings affected by sub-pixel mixing, structural complexity, supergene overprint, and spectrally similar alteration assemblages.
The SAPM is an operational prioritization layer and should not be interpreted as a direct mineral map.
Author

Marco Solinas

Citation

If you use this repository, please cite the related manuscript once available.