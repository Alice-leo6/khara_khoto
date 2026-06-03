# Climate Data Analysis Scripts
This repository contains several Python scripts used for climate data processing and circulation-pattern analysis, including model evaluation, spatial interpolation, and SLP EOF decomposition.
## File description
### 1. `Taylor Diagram.py`
This script is used to evaluate the performance of climate models against reference or reanalysis data using a Taylor diagram.
Main functions include:
- calculating spatial correlation coefficients;
- calculating normalized standard deviation;
- calculating centered root-mean-square difference;
- comparing multiple climate models with reference data;
- visualizing model performance in a Taylor diagram.
This script is mainly used for selecting better-performing climate models before multi-model ensemble or multi-model median analysis.
---
### 2. `nao_index_from_slp_eof.py`
This script is used to extract the North Atlantic Oscillation (NAO) index from winter sea level pressure (SLP) anomaly fields using empirical orthogonal function (EOF) analysis.
Main functions include:
- reading winter SLP data for the North Atlantic sector (30°–90°N, 90°W–30°E);
- calculating SLP anomalies relative to the seasonal climatology;
- applying area-weighted EOF decomposition over the domain;
- extracting EOF1 as the dominant NAO spatial mode;
- using the corresponding PC1 time series as the NAO index;
- visualizing NAO spatial patterns and temporal variability.
This script is suitable for analyzing large-scale winter atmospheric circulation variability over the North Atlantic–Eurasian region.
---
### 3. `bilinear_interpolation.py`
This script is used to interpolate gridded climate data onto a target spatial resolution using bilinear interpolation.
Main functions include:
- reading gridded climate model or reanalysis data;
- defining a target latitude–longitude grid;
- applying bilinear interpolation;
- exporting interpolated data for further analysis.
This script is mainly used to place climate model outputs with different spatial resolutions onto a common grid before inter-model comparison or ensemble analysis.
---
## Author
Xiaoyi Cui
