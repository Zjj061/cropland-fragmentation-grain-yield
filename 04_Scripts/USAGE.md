# Step-by-Step Usage Guidelines
This document provides a complete, reproducible workflow for the cropland fragmentation and grain yield coupling analysis, following the research framework presented in the study.

---

## 📌 Overview of the Workflow
The analysis is structured into five main stages:
1.  Cropland data acquisition and preprocessing
2.  Determination of optimal landscape scale (granularity & amplitude)
3.  Calculation of cropland fragmentation index
4.  Grain yield data acquisition and spatialization
5.  Coupling relationship analysis between fragmentation and grain yield

---

## 1. Environment and Software Requirements
All steps in this study require the following software and packages:
- ArcGIS Pro 3.1 (for resampling, spatial analysis, and grid processing)
- Fragstats 4.2 (for landscape metric calculation)
- Python 3.7+ with ArcPy & NumPy (for normalization, entropy weight, and composite index)
- SPSS (for correlation analysis)
- GS+ 9.0 (for semivariogram analysis)

---

## 2. Stage (a) Cropland Data Acquisition & Preprocessing
### Input data:
- China Land Cover Dataset (CLCD) national land cover maps (2010, 2014, 2018, 2022)

### Steps:
1.  **Study area extraction**
    - Use the Anhui Province boundary mask to clip the national CLCD data.
    - Output: Raw CLCD data for Anhui Province (stored in `01_Raw_Data/`).
2.  **Cropland pixel extraction**
    - Reclassify the clipped data to retain only cropland pixels; set non-cropland pixels to NoData.
    - Resample to the optimal granularity (determined in Stage b).
    - Output: Processed cropland rasters (stored in `02_Processed_Data/`).

---

## 3. Stage (b) Optimal Landscape Scale Determination
### 3.1 Optimal Granularity
1.  **Multi-granularity resampling**
    - Resample the cropland raster at 30 m to 360 m granularities in ArcGIS Pro.
2.  **Landscape index calculation**
    - Run Fragstats 4.2 on each resampled raster to generate granularity response curves for key metrics.
3.  **Scale domain evaluation**
    - Use the area information loss evaluation model to identify the optimal granularity.
    - Output: 180 m optimal granularity.

### 3.2 Optimal Amplitude
1.  **Multi-window landscape index**
    - At 180 m granularity, calculate landscape metrics using 360 m to 2160 m moving windows in Fragstats.
2.  **Random point sampling**
    - Extract landscape index values at 1200 random points for each window size.
3.  **Semivariogram analysis**
    - Use GS+ 9.0 to fit semivariograms and calculate the block-to-base ratio `Co/(C+Co)`.
    - Output: 1080 m optimal amplitude.

---

## 4. Stage (c) Cropland Landscape Fragmentation Evaluation
### 4.1 Correlation screening
1.  Run Pearson correlation analysis in SPSS on the nine landscape metrics.
2.  Retain only indicators with strong explanatory power and low redundancy.

### 4.2 Entropy weight & composite index calculation
1.  **Normalization**
    - Run the Python script in `04_Scripts/cropland_fragmentation_index.py`.
    - Normalize metrics (positive/negative direction).
2.  **Entropy weight calculation**
    - Compute weights for each indicator using the entropy weight method.
3.  **Composite index generation**
    - Integrate weighted metrics into a single cropland fragmentation index raster.
    - Output: `composite_index.tif` (final fragmentation index).

---

## 5. Stage (d) Grain Yield Data Acquisition & Spatialization
1.  **Statistical data collection**
    - Obtain grain yield data from Anhui Provincial Statistical Yearbooks (2011, 2015, 2019, 2023).
2.  **Spatial interpolation**
    - Use ArcGIS Pro with the rotating grid method to convert statistical yield data into continuous spatial rasters.
    - Output: Spatialized grain yield data for each study year.

---

## 6. Stage (e) Coupling Relationship Analysis
1.  **Coupling degree model**
    - Calculate the coupling degree between fragmentation index and grain yield raster layers.
    - Classify into high, moderate, weak, and low coupling levels.
2.  **Coupling coordination degree model**
    - Further compute the coupling coordination degree.
    - Classify into high, good, moderate, poor, and low coordination levels.

---

## 7. Output Files Summary
| Folder/File | Description |
|---|---|
| `01_Raw_Data/` | Clipped CLCD land cover data for Anhui Province |
| `02_Processed_Data/` | 180 m cropland rasters with NoData for non-cropland |
| `03_Landscape_Metrics/` | Fragstats-derived landscape index rasters |
| `04_Scripts/` | Python script for normalization, entropy weight, and composite index |
| `weights.csv` | Indicator weights from entropy method |
| `composite_index.tif` | Final cropland fragmentation index raster |

---

## 8. Reproduction Notes
- All raster processing must be performed using the same spatial reference and cell size.
- For the Python script, update the `arcpy.env.workspace` path to your local directory before running.
- The workflow is modular; each stage can be reproduced independently with the corresponding input data.

---

If you have questions about any step, please refer to the paper methodology section or contact the corresponding author.