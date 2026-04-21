# -*- coding: utf-8 -*-
"""
Cropland Fragmentation Comprehensive Index Calculation
Using Entropy Weight Method (Entropy Weight)
Based on 9 Landscape Metrics

Processing Flow:
1. Raster normalization (positive/negative)
2. Entropy weight calculation
3. Comprehensive fragmentation index integration

Author: Your Name
Date: 2025
Study Scale: 180m granularity, 1080m amplitude
"""

import arcpy
import numpy as np
import os

# ---------------------- ENVIRONMENT SETTINGS ----------------------
# Set workspace (MODIFY THIS PATH BEFORE RUNNING)
arcpy.env.workspace = r"D:\ZJJ\GD_CLF\2010"
arcpy.env.overwriteOutput = True

# ---------------------- INPUT RASTER LIST ----------------------
# 9 landscape metrics input files
raster_list = {
    "NP": "NP_2010_180_1080.tif",
    "PD": "PD_2010_180_1080.tif",
    "LSI": "LSI_2010_180_1080.tif",
    "DIVISION": "DIVISION_2010_180_1080.tif",
    "AREA_MN": "AREA_MN_2010_180_1080.tif",
    "PARA_MN": "PARA_MN_2010_180_1080.tif",
    "CLUMPY": "CLUMPY_2010_180_1080.tif",
    "COHESION": "COHESION_2010_180_1080.tif",
    "AI": "AI_2010_180_1080.tif"
}

# ---------------------- MAIN PROCESSING ----------------------
if __name__ == "__main__":
    normalized_rasters = {}

    # Step 1: Normalize each landscape metric
    for indicator, raster in raster_list.items():
        print(f"Processing: {indicator}")

        # Read raster and NoData
        raster_obj = arcpy.Raster(raster)
        no_data = raster_obj.noDataValue
        arr = arcpy.RasterToNumPyArray(raster_obj, nodata_to_value=no_data)
        masked_arr = np.ma.masked_equal(arr, no_data)

        min_v = masked_arr.min()
        max_v = masked_arr.max()
        print(f"{indicator} | Min: {min_v}, Max: {max_v}")

        # Normalization
        if indicator in ["NP", "PD", "LSI", "DIVISION"]:
            norm_arr = (masked_arr - min_v) / (max_v - min_v)
        else:
            norm_arr = (max_v - masked_arr) / (max_v - min_v)

        # Save normalized raster
        out_path = os.path.join(arcpy.env.workspace, f"normalized_{indicator}.tif")
        ll = arcpy.Describe(raster).extent.lowerLeft
        cell_size = arcpy.Describe(raster).meanCellWidth

        norm_raster = arcpy.NumPyArrayToRaster(
            norm_arr.filled(no_data), ll, cell_size, cell_size, no_data
        )
        norm_raster.save(out_path)
        normalized_rasters[indicator] = out_path
        print(f"{indicator} normalization completed.\n")

    # Step 2: Calculate entropy weight
    print("Calculating entropy weights...")
    entropy_list = []
    for path in normalized_rasters.values():
        arr = arcpy.RasterToNumPyArray(path)
        total = np.sum(arr)
        prop = arr / total
        prop = np.where(prop > 0, prop, 1)
        entropy = -np.sum(prop * np.log(prop))
        entropy_list.append(entropy)

    # Step 3: Calculate redundancy and weights
    n = len(raster_list)
    redundancy = [1 - (e / np.log(n)) for e in entropy_list]
    total_redun = sum(redundancy)
    weights = [r / total_redun for r in redundancy]

    # Save weights to CSV
    csv_path = os.path.join(arcpy.env.workspace, "weights.csv")
    with open(csv_path, "w") as f:
        f.write("Indicator,Weight\n")
        for ind, w in zip(raster_list.keys(), weights):
            f.write(f"{ind},{w:.4f}\n")
    print(f"Weights saved to: {csv_path}")

    # Step 4: Compute comprehensive fragmentation index
    print("Generating cropland fragmentation composite index...")
    composite = None

    for i, path in enumerate(normalized_rasters.values()):
        arr = arcpy.RasterToNumPyArray(path)
        weighted = weights[i] * arr
        composite = weighted if composite is None else composite + weighted

    # Get spatial reference
    first_raster = list(raster_list.values())[0]
    desc = arcpy.Describe(first_raster)
    ll = desc.extent.lowerLeft
    cell_x = desc.meanCellWidth
    cell_y = desc.meanCellHeight
    sr = desc.spatialReference

    # Save final index
    out_composite = os.path.join(arcpy.env.workspace, "composite_index.tif")
    composite_raster = arcpy.NumPyArrayToRaster(composite, ll, cell_x, cell_y)
    composite_raster.save(out_composite)
    arcpy.DefineProjection_management(out_composite, sr)

    # Print min/max of composite index
    comp_arr = arcpy.RasterToNumPyArray(out_composite)
    masked_comp = np.ma.masked_invalid(comp_arr)
    print(f"\nComposite Index | Min: {masked_comp.min()}, Max: {masked_comp.max()}")
    print(f"\n✅ All tasks completed! Output: {out_composite}")
