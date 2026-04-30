# Step-by-Step Usage Guidelines
This document provides a complete, reproducible workflow for the cropland fragmentation and grain yield coupling analysis, following the research framework presented in the study.

---

## 📌 Overview of the Workflow
(Fig. 2 Schematic representation of the comprehensive multiscale evaluation framework.png)

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
    - Classify into high, strong, moderate, weak, and low coupling levels.
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

## 9. Stage (f) Spatial Visualization & Coupling Coordination Mapping
### 9.1 Script: CLF_Coupling_Coordination_Spatial_Analysis.py
This script performs spatiotemporal visualization and statistical analysis for cropland fragmentation (CLF), coupling degree (C), and coordination degree (D).

**Input**:
- Multi-temporal CLF, C, D raster data (2010, 2014, 2018, 2022) in `05_Coupling_Coordination_Data/`

**Steps**:
1. Place `CLF_YYYY.tif`, `C_YYYY.tif`, `D_YYYY.tif` in the root directory or specify the path in the script.
2. Run the script directly using Python 3.7+.
3. The script automatically loads, classifies, maps, and generates publication-quality figures.

**Output**:
- 4-year spatiotemporal distribution maps (3×4 panel)
- Temporal trend charts of key indicators
- Bivariate CLF–coordination spatial maps
- Critical region identification
- Statistical comparison table (PNG)
- All figures saved to `visual_computing_output/`

**Dependencies**:
- numpy, matplotlib, rasterio, scipy, pandas

---

## 10. Stage (g) Multi‑Scenario Generative Simulation & Future Prediction
### 10.1 Script: CLF_Coupling_Coordination_Generative_Simulation.py
This script implements trend extrapolation, generative modeling, and multi‑scenario prediction for 2030 under four development pathways.
**Input**:
- Multi-temporal CLF, C, D raster data (2010, 2014, 2018, 2022) in `05_Coupling_Coordination_Data/`
**Steps**:
1. Ensure input rasters are placed correctly.
2. Run the script to perform:
   - Temporal trend calculation
   - Weighted multi-year data fusion
   - Four-scenario generative simulation (Restoration / Baseline / Trend / Extreme)
   - Coupling-coordination feedback mechanism simulation
   - Critical region detection and zoomed hotspot analysis
3. Results are automatically exported as high-resolution figures and statistics.
**Output**:
- 4×4 multi-scenario spatial comparison maps
- Zoomed critical region detail maps
- Classification proportion statistics (`class_proportions.csv`)
- All outputs saved to `output_top/`
**Scenarios**:
- Restoration (λ = -0.6): Ecological restoration and defragmentation
- Baseline (λ = 0): Current trend continuation
- Trend (λ = 1.0): Historical trend extrapolation
- Extreme (λ = 1.6): High fragmentation pressure
**Dependencies**:
- numpy, matplotlib, rasterio, scipy, pandas

---

## 11. Full Output File Structure Summary
| Folder/File | Description |
|---|---|
| `05_Coupling_Coordination_Data/` | CLF, coupling degree (C), coordination degree (D) for four years |
| `output_top/` | Multi-scenario simulation figures and statistical tables |
| `visual_computing_output/` | Spatiotemporal visualization and trend analysis figures |
| `class_proportions.csv` | Classification area proportions under different scenarios |

---

## 12. Full Workflow Summary
The complete reproducible pipeline:
1. Data preprocessing → 2. Optimal scale determination → 3. CLF index calculation
4. Grain yield spatialization → 5. Coupling-coordination modeling
6. **Spatial visualization & mapping** → 7. **Multi-scenario generative prediction**

---

If you have questions about any step, please refer to the paper methodology section or contact the corresponding author.
