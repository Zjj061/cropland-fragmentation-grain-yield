"""
Visual Computing-Based Spatial Visualization and Analysis
Processing cropland fragmentation (CLF), coupling degree (C), and coordination degree (D)
for years 2010, 2014, 2018, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
from scipy.ndimage import gaussian_filter, label
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

CLF_CLASSES = {
    'Slight': [0.00, 0.04],
    'Low': [0.04, 0.14],
    'Moderate': [0.14, 0.26],
    'High': [0.26, 0.42],
    'Extreme': [0.42, 1.00]
}

COUPLING_CLASSES = {
    'Low': [0.00, 0.20],
    'Weak': [0.20, 0.40],
    'Moderate': [0.40, 0.60],
    'Strong': [0.60, 0.80],
    'High': [0.80, 1.00]
}

COORDINATION_CLASSES = {
    'Poor': [0.00, 0.20],
    'Weak': [0.20, 0.40],
    'Moderate': [0.40, 0.60],
    'Good': [0.60, 0.80],
    'High': [0.80, 1.00]
}

CLF_CMAP = ListedColormap(['#1a9850', '#66bd63', '#fee08b', '#f46d43', '#d73027'])
COUPLING_CMAP = ListedColormap(['#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])
COORDINATION_CMAP = ListedColormap(['#fff5eb', '#fdd0a2', '#fdae6b', '#e6550d', '#a63603'])

CLF_CMAP.set_bad(color='white')
COUPLING_CMAP.set_bad(color='white')
COORDINATION_CMAP.set_bad(color='white')

def load_raster_data(year, base_path='.'):
    clf_path = os.path.join(base_path, f"CLF_{year}.tif")
    d_path = os.path.join(base_path, f"D_{year}.tif")
    c_path = os.path.join(base_path, f"C_{year}.tif")
    
    with rasterio.open(clf_path) as src_clf, \
         rasterio.open(d_path) as src_d, \
         rasterio.open(c_path) as src_c:
        
        clf_data = src_clf.read(1).astype(np.float32)
        d_data = src_d.read(1).astype(np.float32)
        c_data = src_c.read(1).astype(np.float32)
        
        metadata = src_clf.meta
        transform = src_clf.transform
        bounds = src_clf.bounds
        crs = src_clf.crs
        
        nodata = src_clf.nodata if src_clf.nodata is not None else -9999
        valid_mask = (clf_data != nodata) & (d_data != nodata) & (c_data != nodata)
        
        clf_data = np.where(valid_mask, clf_data, np.nan)
        d_data = np.where(valid_mask, d_data, np.nan)
        c_data = np.where(valid_mask, c_data, np.nan)
        
        return {
            'clf': clf_data,
            'd': d_data,
            'c': c_data,
            'metadata': metadata,
            'transform': transform,
            'bounds': bounds,
            'crs': crs,
            'valid_mask': valid_mask,
            'year': year
        }

def classify_clf(clf_data, valid_mask):
    classified = np.full_like(clf_data, np.nan, dtype=float)
    class_labels = ['Slight', 'Low', 'Moderate', 'High', 'Extreme']
    thresholds = [0.00, 0.04, 0.14, 0.26, 0.42, 1.00]
    proportions = {}
    for i in range(5):
        mask = (clf_data >= thresholds[i]) & (clf_data < thresholds[i+1]) & valid_mask
        classified[mask] = i
        proportions[class_labels[i]] = np.sum(mask) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0
    return classified, class_labels, proportions

def classify_coupling(c_data, valid_mask):
    classified = np.full_like(c_data, np.nan, dtype=float)
    class_labels = ['Low', 'Weak', 'Moderate', 'Strong', 'High']
    thresholds = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    proportions = {}
    for i in range(5):
        mask = (c_data >= thresholds[i]) & (c_data < thresholds[i+1]) & ~np.isnan(c_data)
        if i == 4:
            mask = (c_data >= thresholds[i]) & (c_data <= thresholds[i+1]) & ~np.isnan(c_data)
        classified[mask] = i
        proportions[class_labels[i]] = np.sum(mask) / np.sum(~np.isnan(c_data)) if np.sum(~np.isnan(c_data)) > 0 else 0
    return classified, class_labels, proportions

def classify_coordination(d_data, valid_mask):
    classified = np.full_like(d_data, np.nan, dtype=float)
    class_labels = ['Poor', 'Weak', 'Moderate', 'Good', 'High']
    thresholds = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    proportions = {}
    for i in range(5):
        mask = (d_data >= thresholds[i]) & (d_data < thresholds[i+1]) & ~np.isnan(d_data)
        if i == 4:
            mask = (d_data >= thresholds[i]) & (d_data <= thresholds[i+1]) & ~np.isnan(d_data)
        classified[mask] = i
        proportions[class_labels[i]] = np.sum(mask) / np.sum(~np.isnan(d_data)) if np.sum(~np.isnan(d_data)) > 0 else 0
    return classified, class_labels, proportions

def bivariate_color_mapping(fragmentation, coordination):
    f_norm = np.clip(fragmentation, 0, 1)
    d_norm = np.clip(coordination, 0, 1)
    hue = 0.33 * (1 - f_norm) + 0.02 * f_norm
    saturation = np.ones_like(f_norm) * 0.9
    brightness = 0.3 + 0.5 * d_norm
    h = hue * 6.0
    c = saturation * brightness
    x = c * (1 - np.abs((h % 2) - 1))
    m = brightness - c
    h_int = np.floor(h).astype(int) % 6
    rgb = np.zeros((h.shape[0], h.shape[1], 3))
    
    rgb[h_int == 0] = np.stack([c[h_int == 0], x[h_int == 0], np.zeros_like(c[h_int == 0])], axis=-1)
    rgb[h_int == 1] = np.stack([x[h_int == 1], c[h_int == 1], np.zeros_like(c[h_int == 1])], axis=-1)
    rgb[h_int == 2] = np.stack([np.zeros_like(c[h_int == 2]), c[h_int == 2], x[h_int == 2]], axis=-1)
    rgb[h_int == 3] = np.stack([np.zeros_like(c[h_int == 3]), x[h_int == 3], c[h_int == 3]], axis=-1)
    rgb[h_int == 4] = np.stack([x[h_int == 4], np.zeros_like(c[h_int == 4]), c[h_int == 4]], axis=-1)
    rgb[h_int == 5] = np.stack([c[h_int == 5], np.zeros_like(c[h_int == 5]), x[h_int == 5]], axis=-1)
    
    rgb = rgb + m[..., np.newaxis]
    rgb = np.clip(rgb, 0, 1)
    nan_mask = np.isnan(fragmentation) | np.isnan(coordination)
    rgb[nan_mask] = [1.0, 1.0, 1.0]
    return rgb

def identify_critical_regions(coupling_map, coordination_map, thresholds=None):
    if thresholds is None:
        thresholds = {'high_coupling': 0.6, 'low_coordination': 0.4}
    valid = ~(np.isnan(coupling_map) | np.isnan(coordination_map))
    coupling_clean = np.where(valid, coupling_map, 0)
    coordination_clean = np.where(valid, coordination_map, np.nan)
    
    high_coupling = coupling_clean >= thresholds['high_coupling']
    low_coordination = coordination_clean <= thresholds['low_coordination']
    critical_mask = high_coupling & low_coordination & valid
    
    labeled, num_features = label(critical_mask)
    sizes = label(critical_mask, labeled, range(1, num_features+1))
    for i, size in enumerate(sizes, 1):
        if size < 9:
            critical_mask[labeled == i] = False
    
    total_valid = np.sum(valid)
    critical_count = np.sum(critical_mask)
    return critical_mask, critical_count / total_valid if total_valid > 0 else 0

def calculate_restrictive_coupling_ratio(coupling_map, coordination_map, critical_mask):
    valid = ~(np.isnan(coupling_map) | np.isnan(coordination_map))
    total_valid = np.sum(valid)
    if total_valid == 0:
        return 0.0, 0.0, 0.0
    high_coupling_ratio = np.sum((coupling_map >= 0.6) & valid) / total_valid
    poor_coordination_ratio = np.sum((coordination_map <= 0.4) & valid) / total_valid
    restrictive_ratio = np.sum(critical_mask & valid) / total_valid
    return high_coupling_ratio, poor_coordination_ratio, restrictive_ratio

def create_4year_figure(data_list, output_dir='./output'):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, wspace=0.02, hspace=0.02)
    years = [2010,2014,2018,2022]
    
    for col_idx, year in enumerate(years):
        data = data_list[col_idx]
        clf_data = data['clf']
        d_data = data['d']
        c_data = data['c']
        valid_mask = data['valid_mask']
        
        clf_classified, clf_labels, clf_props = classify_clf(clf_data, valid_mask)
        coupling_classified, coupling_labels, coupling_props = classify_coupling(c_data, valid_mask)
        coord_classified, coord_labels, coord_props = classify_coordination(d_data, valid_mask)

        ax1 = fig.add_subplot(gs[0, col_idx])
        ax1.imshow(clf_classified, cmap=CLF_CMAP, vmin=0, vmax=4)
        ax1.set_title(f'Cropland Fragmentation ({year})', fontsize=10, fontweight='bold', y=0.98)
        ax1.axis('off')
        leg1 = [Patch(facecolor=CLF_CMAP(i), label=f'{clf_labels[i]}: {clf_props[clf_labels[i]]:.2%}') for i in range(5)]
        ax1.legend(handles=leg1, loc='upper right', fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.05)

        ax2 = fig.add_subplot(gs[1, col_idx])
        ax2.imshow(coupling_classified, cmap=COUPLING_CMAP, vmin=0, vmax=4)
        ax2.set_title(f'Coupling Degree ({year})', fontsize=10, fontweight='bold', y=0.98)
        ax2.axis('off')
        leg2 = [Patch(facecolor=COUPLING_CMAP(i), label=f'{coupling_labels[i]}: {coupling_props[coupling_labels[i]]:.2%}') for i in range(5)]
        ax2.legend(handles=leg2, loc='upper right', fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.05)

        ax3 = fig.add_subplot(gs[2, col_idx])
        ax3.imshow(coord_classified, cmap=COORDINATION_CMAP, vmin=0, vmax=4)
        ax3.set_title(f'Coordination Degree ({year})', fontsize=10, fontweight='bold', y=0.98)
        ax3.axis('off')
        leg3 = [Patch(facecolor=COORDINATION_CMAP(i), label=f'{coord_labels[i]}: {coord_props[coord_labels[i]]:.2%}') for i in range(5)]
        ax3.legend(handles=leg3, loc='upper right', fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.05)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Fig_4Years_CLF_C_D.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close(fig)
    return output_path

def create_comparison_table(results, output_dir='./output'):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Year'] + list(results[0]['clf_proportions'].keys()) + ['High\nCoupling', 'Poor\nCoord', 'Critical\nRegion']
    table_data = [headers]
    
    for r in results:
        row = [str(r['year'])]
        row.extend([f"{r['clf_proportions'][k]:.2%}" for k in r['clf_proportions'].keys()])
        row.extend([f"{r['high_coupling_ratio']:.2%}", f"{r['poor_coordination_ratio']:.2%}", f"{r['restrictive_ratio']:.2%}"])
        table_data.append(row)
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Table: Spatiotemporal Evolution of Cropland Fragmentation and Coupling Coordination\n(2010-2022)', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'Table_Comparison_2010_2022.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path

def create_temporal_trend_plot(results, output_dir='./output'):
    years = [r['year'] for r in results]
    high_coupling = [r['high_coupling_ratio'] * 100 for r in results]
    poor_coordination = [r['poor_coordination_ratio'] * 100 for r in results]
    critical_regions = [r['restrictive_ratio'] * 100 for r in results]
    
    clf_classes = list(results[0]['clf_proportions'].keys())
    clf_trends = {cls: [r['clf_proportions'][cls] * 100 for r in results] for cls in clf_classes}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.plot(years, high_coupling, 'o-', linewidth=2, markersize=8, label='High Coupling (C≥0.6)', color='#d73027')
    ax1.plot(years, poor_coordination, 's-', linewidth=2, markersize=8, label='Poor Coordination (D≤0.4)', color='#4575b4')
    ax1.plot(years, critical_regions, '^-', linewidth=2, markersize=8, label='Critical Regions', color='#e0a01e')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Proportion (%)', fontsize=11)
    ax1.set_title('(a) Temporal Evolution of Key Indicators', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(years)
    
    ax2 = axes[1]
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    ax2.stackplot(years, clf_trends['Slight'], clf_trends['Low'], clf_trends['Moderate'], clf_trends['High'], clf_trends['Extreme'],
                  labels=['Slight', 'Low', 'Moderate', 'High', 'Extreme'], colors=colors, alpha=0.8)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Proportion (%)', fontsize=11)
    ax2.set_title('(b) CLF Fragmentation Class Composition', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(years)
    
    plt.suptitle('Temporal Trends of Cropland Fragmentation and Coupling Coordination (2010-2022)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'Fig_Temporal_Trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path

def create_bivariate_2x3_figure(data_list, output_dir='./output'):
    import matplotlib.patches as patches
    
    fig = plt.figure(figsize=(10, 10), dpi=300)
    gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1,1,0.6], height_ratios=[1,1], hspace=0.02, wspace=0.02)
    years = [2010, 2014, 2018, 2022]
    positions = [(0,0), (0,1), (1,0), (1,1)]
    year_data = {d['year']: d for d in data_list}
    
    for idx, (year, (row, col)) in enumerate(zip(years, positions)):
        data = year_data[year]
        clf_data = data['clf']
        d_data = data['d']
        c_data = data['c']
        valid_mask = data['valid_mask']
        
        critical_mask, _ = identify_critical_regions(c_data, d_data)
        
        clf_min, clf_max = np.nanmin(clf_data[valid_mask]), np.nanmax(clf_data[valid_mask])
        clf_norm = np.clip((clf_data - clf_min)/(clf_max - clf_min), 0, 1) if (clf_max - clf_min) > 1e-8 else np.zeros_like(clf_data)
        
        d_min, d_max = np.nanmin(d_data[valid_mask]), np.nanmax(d_data[valid_mask])
        d_norm = np.clip((d_data - d_min)/(d_max - d_min), 0, 1) if (d_max - d_min) > 1e-8 else np.zeros_like(d_data)
        
        rgb_map = bivariate_color_mapping(clf_norm, d_norm)
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(rgb_map, aspect='auto')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        
        if np.any(critical_mask):
            labeled, num_features = label(critical_mask)
            for i in range(1, num_features+1):
                region_mask = labeled == i
                if np.sum(region_mask) > 600:
                    y, x = np.where(region_mask)
                    rect = patches.Rectangle((x.min(), y.min()), x.max()-x.min(), y.max()-y.min(),
                                           fill=False, edgecolor='red', linewidth=1.0, linestyle='--', alpha=0.8)
                    ax.add_patch(rect)
        
        ax.text(0.5, 0.96, f'Bivariate Map ({year})', ha='center', fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    ax_stats = fig.add_subplot(gs[:, 2])
    ax_stats.axis('off')
    ax_stats.text(0.55, 0.9, "Spatial Pattern Statistics (2010-2022)", ha='center', fontsize=9, fontweight='bold')
    
    table_data = []
    for year in years:
        d = year_data[year]
        cm, _ = identify_critical_regions(d['c'], d['d'])
        hc, pc, rr = calculate_restrictive_coupling_ratio(d['c'], d['d'], cm)
        table_data.append([f"{year}", f"{hc:.2%}", f"{pc:.2%}", f"{rr:.2%}"])
    
    stats_table = ax_stats.table(cellText=table_data, colLabels=["Year", "High Coupling\n(C≥0.6)", "Poor Coord\n(D≤0.4)", "Critical\nRegion"],
                               loc='upper center', bbox=[0.05,0.72,1.4,0.15], cellLoc='center', colWidths=[0.25,0.38,0.38,0.38])
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(7)
    stats_table.scale(1,1.5)
    
    for (row, col), cell in stats_table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
    
    y_pos, line_height = 0.70, 0.022
    text_lines = [("Dashed red boxes:", True), ("Critical regions", False), ("", False),
                  ("Critical regions:", True), ("High coupling (C ≥ 0.6) and", False), ("Low coordination (D ≤ 0.4)", False)]
    for txt, bold in text_lines:
        ax_stats.text(0.05, y_pos, txt, fontsize=8, weight="bold" if bold else "normal")
        y_pos -= line_height
    
    n = 80
    f_grid, d_grid = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    hue = 0.33*(1-f_grid)+0.02*f_grid
    brightness = 0.3+0.5*d_grid
    h = hue*6
    c_vals = 0.9*brightness
    x = c_vals*(1-abs((h%2)-1))
    m = brightness - c_vals
    h_int = np.floor(h).astype(int)%6
    legend_colors = np.zeros((n,n,3))
    
    for i in range(6):
        mask = h_int==i
        if i==0: legend_colors[mask] = np.stack([c_vals[mask],x[mask],np.zeros_like(c_vals[mask])],axis=-1)
        elif i==1: legend_colors[mask] = np.stack([x[mask],c_vals[mask],np.zeros_like(c_vals[mask])],axis=-1)
        elif i==2: legend_colors[mask] = np.stack([np.zeros_like(c_vals[mask]),c_vals[mask],x[mask]],axis=-1)
        elif i==3: legend_colors[mask] = np.stack([np.zeros_like(c_vals[mask]),x[mask],c_vals[mask]],axis=-1)
        elif i==4: legend_colors[mask] = np.stack([x[mask],np.zeros_like(c_vals[mask]),c_vals[mask]],axis=-1)
        elif i==5: legend_colors[mask] = np.stack([c_vals[mask],np.zeros_like(c_vals[mask]),x[mask]],axis=-1)
    
    legend_colors = np.clip(legend_colors + m[...,np.newaxis], 0,1)
    legend_ax = fig.add_axes([0.74,0.20,0.22,0.22])
    legend_ax.imshow(legend_colors[::-1], aspect='auto')
    legend_ax.set_aspect('equal', adjustable='box')
    legend_ax.set_title('Bivariate Mapping', fontsize=9, fontweight='bold', pad=4)
    legend_ax.set_xticks([0,n-1])
    legend_ax.set_xticklabels(['Low','High'], fontsize=7)
    legend_ax.set_yticks([0,n-1])
    legend_ax.set_yticklabels(['High','Low'], fontsize=7)
    legend_ax.text(n/2, n+1.5, 'Fragmentation →', ha='center', fontsize=8)
    legend_ax.text(-3, n/2, 'Coordination →', ha='right', rotation=90, fontsize=8)
    
    legend_ax.axhline(n/2.5, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    legend_ax.axvline(n*0.6, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    legend_ax.text(n*0.78, n*0.22, 'Critical\nRegion', ha='center', fontsize=6, color='white', weight='bold', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Fig_Bivariate_2x3_Critical_Regions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return save_path

def main():
    print("=" * 70)
    print("Visual Computing-Based Spatial Visualization and Analysis")
    print("Processing years: 2010, 2014, 2018, 2022")
    print("=" * 70)
    
    years = [2010, 2014, 2018, 2022]
    base_path = "."
    output_dir = "./visual_computing_output"
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    all_results = []
    
    try:
        for year in years:
            print(f"\nProcessing year: {year}")
            data = load_raster_data(year, base_path)
            all_data.append(data)
            
            critical_mask, critical_prop = identify_critical_regions(data['c'], data['d'])
            hc, pc, rr = calculate_restrictive_coupling_ratio(data['c'], data['d'], critical_mask)
            
            clf_classified, clf_labels, clf_props = classify_clf(data['clf'], data['valid_mask'])
            coupling_classified, coupling_labels, coupling_props = classify_coupling(data['c'], data['valid_mask'])
            coord_classified, coord_labels, coord_props = classify_coordination(data['d'], data['valid_mask'])
            
            res = {
                'year': year,
                'clf_proportions': clf_props,
                'coupling_proportions': coupling_props,
                'coordination_proportions': coord_props,
                'high_coupling_ratio': hc,
                'poor_coordination_ratio': pc,
                'restrictive_ratio': rr
            }
            all_results.append(res)
        
        create_4year_figure(all_data, output_dir)
        create_comparison_table(all_results, output_dir)
        create_temporal_trend_plot(all_results, output_dir)
        create_bivariate_2x3_figure(all_data, output_dir)
        
        print("\n✅ All figures generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
