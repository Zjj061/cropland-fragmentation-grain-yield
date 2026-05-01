"""
Generative Simulation and Spatial Analysis of Cropland Fragmentation 
and Coupling Coordination (2010-2022)
TOP Journal Visualization & 2030 Multi-Scenario Prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CLF_CLASSES = ['Slight', 'Low', 'Moderate', 'High', 'Extreme']
CLF_THRESH = [0.00, 0.04, 0.14, 0.26, 0.42, 1.00]
CLF_CMAP = ListedColormap(['#1a9850', '#66bd63', '#fee08b', '#f46d43', '#d73027'])

C_CLASSES = ['Low', 'Weak', 'Moderate', 'Strong', 'High']
C_THRESH = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
C_CMAP = ListedColormap(['#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])

D_CLASSES = ['Low', 'Poor', 'Moderate', 'Good', 'High']
D_THRESH = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
D_CMAP = ListedColormap(['#fff5eb', '#fdd0a2', '#fdae6b', '#e6550d', '#a63603'])

CLF_CMAP.set_bad('white')
C_CMAP.set_bad('white')
D_CMAP.set_bad('white')

def load_tif(path, verbose=True):
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else -9999
        data[data == nodata] = np.nan
        if verbose:
            print(f"  ✓ Loaded: {os.path.basename(path)} | shape={data.shape} | range=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
        return data
    except Exception as e:
        print(f"  ✗ Failed: {path} | {str(e)}")
        return None

def load_all(data_dir="./"):
    years = [2010, 2014, 2018, 2022]
    clf, c, d = {}, {}, {}
    for y in years:
        clf[y] = load_tif(os.path.join(data_dir, f"CLF_{y}.tif"))
        c[y] = load_tif(os.path.join(data_dir, f"C_{y}.tif"))
        d[y] = load_tif(os.path.join(data_dir, f"D_{y}.tif"))
    return clf, c, d

def fuse(data):
    years = [2010,2014,2018,2022]
    stack = np.stack([data[y] for y in years])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    weights = weights / weights.sum()
    out = np.nansum(stack * weights[:,None,None], axis=0)
    out[np.all(np.isnan(stack), axis=0)] = np.nan
    return out

def trend(clf):
    t = np.array([0, 1, 2, 3])
    stack = np.stack([clf[y] for y in [2010, 2014, 2018, 2022]])
    slope = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    
    for i in range(slope.shape[0]):
        for j in range(slope.shape[1]):
            yv = stack[:, i, j]
            if not np.any(np.isnan(yv)):
                slope[i, j] = np.polyfit(t, yv, 1)[0]
    return slope

def classify_5class(data, thresholds):
    classified = np.full_like(data, np.nan, dtype=np.float32)
    valid = ~np.isnan(data)
    total = np.sum(valid)
    props = []
    for i in range(5):
        mask = (data >= thresholds[i]) & (data < thresholds[i+1]) & valid
        classified[mask] = i
        props.append(np.sum(mask)/total if total>0 else 0)
    return classified, props

def generate(F, tr, lam, noise_scale=0.08, seed=42):
    if F is None or tr is None:
        return None
    np.random.seed(seed)
    noise = gaussian_filter(np.random.randn(*F.shape), sigma=3) * noise_scale
    res = np.clip(F + lam * tr + noise, 0, 1)
    res[np.isnan(F)] = np.nan
    return res

def simulate_D(F_orig, F_sim, D_orig, response_factor=1.3):
    valid = ~(np.isnan(F_orig) | np.isnan(F_sim))
    delta_F = np.zeros_like(F_orig)
    delta_F[valid] = F_sim[valid] - F_orig[valid]
    
    D_sim = D_orig.copy()
    D_sim[valid] = D_orig[valid] * (1 - response_factor * delta_F[valid])
    D_sim = np.clip(D_sim, 0, 1)
    D_sim[~valid] = np.nan
    return D_sim

def simulate_C(C_orig, D_sim):
    valid = (~np.isnan(C_orig)) & (~np.isnan(D_sim))
    C_from_D = 1 - D_sim
    C_sim = np.full_like(C_orig, np.nan)
    C_sim[valid] = np.clip(0.7 * C_from_D[valid] + 0.3 * C_orig[valid], 0, 1)
    return C_sim

def detect_critical(C, D, thresh_c=0.6, thresh_d=0.4):
    valid = (~np.isnan(C)) & (~np.isnan(D))
    critical = np.zeros_like(C, dtype=bool)
    critical[valid] = (C[valid] >= thresh_c) & (D[valid] <= thresh_d)
    return critical

def bivariate_map(F, D):
    F = np.clip(F, 0, 1)
    D = np.clip(D, 0, 1)
    rgb = np.zeros((*F.shape, 3), dtype=np.float32)
    valid = ~(np.isnan(F) | np.isnan(D))

    r = 0.2 + 0.7 * (1 - F) + 0.3 * D
    g = 0.2 + 0.5 * D
    b = 0.2 + 0.7 * F

    rgb[valid, 0] = r[valid]
    rgb[valid, 1] = g[valid]
    rgb[valid, 2] = b[valid]
    rgb[~valid] = 0.9

    return np.clip(rgb, 0, 1)

def main():
    print("\n" + "="*70)
    print("TOP-JOURNAL GENERATIVE SIMULATION (2010-2022)")
    print("="*70)

    clf, c, d = load_all()
    if clf[2022] is None:
        print("\n❌ ERROR: No data!")
        return

    F = fuse(clf)
    D = fuse(d)
    C = fuse(c)
    tr = trend(clf)

    scenarios = {
        "Restoration (λ=-0.6)": generate(F, tr, -0.6, seed=10),
        "Baseline (λ=0)":        generate(F, tr, 0.0, seed=20),
        "Trend (λ=1.0)":         generate(F, tr, 1.0, seed=30),
        "Extreme (λ=1.6)":       generate(F, tr, 1.6, seed=40)
    }

    D_scn, C_scn, Crit_scn = {}, {}, {}
    prop_records = []

    for name, fsim in scenarios.items():
        D_sim = simulate_D(F, fsim, D)
        C_sim = simulate_C(C, D_sim)
        Crit = detect_critical(C_sim, D_sim)
        
        D_scn[name] = D_sim
        C_scn[name] = C_sim
        Crit_scn[name] = Crit

        f_cls, f_p = classify_5class(fsim, CLF_THRESH)
        c_cls, c_p = classify_5class(C_sim, C_THRESH)
        d_cls, d_p = classify_5class(D_sim, D_THRESH)
        crit_p = np.sum(Crit)/np.sum(~np.isnan(fsim)) if np.sum(~np.isnan(fsim))>0 else 0

        prop_records.append({
            "Scenario": name,
            "CLF_Slight":f_p[0],"CLF_Low":f_p[1],"CLF_Moderate":f_p[2],"CLF_High":f_p[3],"CLF_Extreme":f_p[4],
            "C_Low":c_p[0],"C_Weak":c_p[1],"C_Moderate":c_p[2],"C_Strong":c_p[3],"C_High":c_p[4],
            "D_Low":d_p[0],"D_Poor":d_p[1],"D_Moderate":d_p[2],"D_Good":d_p[3],"D_High":d_p[4],
            "Critical_Region": crit_p
        })

    os.makedirs("output_top", exist_ok=True)
    df = pd.DataFrame(prop_records)
    df.to_csv("output_top/class_proportions.csv", index=False, encoding='utf-8-sig')

    names = list(scenarios.keys())

    fig = plt.figure(figsize=(17, 16))
    gs = fig.add_gridspec(4, 4, wspace=0.02, hspace=0.02)

    for col_idx, name in enumerate(names):
        f_sim = scenarios[name]
        c_sim = C_scn[name]
        d_sim = D_scn[name]
        crit = Crit_scn[name]

        f_cls, f_p = classify_5class(f_sim, CLF_THRESH)
        c_cls, c_p = classify_5class(c_sim, C_THRESH)
        d_cls, d_p = classify_5class(d_sim, D_THRESH)

        ax1 = fig.add_subplot(gs[0, col_idx])
        ax1.imshow(f_cls, cmap=CLF_CMAP, vmin=0, vmax=4)
        ax1.set_title(f'{name}\nCropland Fragmentation', fontsize=10, fontweight='bold', y=0.98)
        ax1.axis('off')
        leg1 = [Patch(facecolor=CLF_CMAP(i), label=f'{CLF_CLASSES[i]}: {f_p[i]:.2%}') for i in range(5)]
        ax1.legend(handles=leg1, loc='upper center', bbox_to_anchor=(1, 1), 
                   fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.2)

        ax2 = fig.add_subplot(gs[1, col_idx])
        ax2.imshow(c_cls, cmap=C_CMAP, vmin=0, vmax=4)
        ax2.set_title(f'Coupling Degree', fontsize=10, fontweight='bold', y=0.98)
        ax2.axis('off')
        leg2 = [Patch(facecolor=C_CMAP(i), label=f'{C_CLASSES[i]}: {c_p[i]:.2%}') for i in range(5)]
        ax2.legend(handles=leg2, loc='upper center', bbox_to_anchor=(1, 1), 
                   fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.3)

        ax3 = fig.add_subplot(gs[2, col_idx])
        ax3.imshow(d_cls, cmap=D_CMAP, vmin=0, vmax=4)
        ax3.set_title(f'Coupling Coordination Degree', fontsize=10, fontweight='bold', y=0.98)
        ax3.axis('off')
        leg3 = [Patch(facecolor=D_CMAP(i), label=f'{D_CLASSES[i]}: {d_p[i]:.2%}') for i in range(5)]
        ax3.legend(handles=leg3, loc='upper center', bbox_to_anchor=(1, 1), 
                   fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.2)

        ax4 = fig.add_subplot(gs[3, col_idx])
        ax4.imshow(crit, cmap=ListedColormap(['white','red']), alpha=0.8)
        ax4.set_title(f'Critical Regions', fontsize=10, fontweight='bold', y=0.98)
        ax4.axis('off')
        leg4 = [Patch(facecolor='red', label='Critical Region')]
        ax4.legend(handles=leg4, loc='upper center', bbox_to_anchor=(1, 1), 
                   fontsize=8, framealpha=1, facecolor='white', borderaxespad=0.2)
        
    output_path = "output_top/Fig_FINAL_4x4.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close(fig)

    all_critical = np.zeros_like(F, dtype=bool)
    for crit in Crit_scn.values():
        all_critical |= crit

    if np.any(all_critical):
        rows, cols = np.where(all_critical)
        center_row = int(np.mean(rows))
        center_col = int(np.mean(cols))
    else:
        center_row, center_col = F.shape[0] // 2, F.shape[1] // 2

    window_size = max(40, min(F.shape[0] // 8, F.shape[1] // 8))
    row_start = max(0, center_row - window_size // 2)
    row_end = min(F.shape[0], center_row + window_size // 2)
    col_start = max(0, center_col - window_size // 2)
    col_end = min(F.shape[1], center_col + window_size // 2)

    y1, y2, x1, x2 = row_start, row_end, col_start, col_end

    fig2, axes2 = plt.subplots(4, 4, figsize=(16,16), dpi=300)
    fig2.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    
    for col in range(4):
        f_sim = scenarios[names[col]]
        d_sim = D_scn[names[col]]
        c_sim = C_scn[names[col]]
        crit = Crit_scn[names[col]]

        f_zoom = f_sim[y1:y2, x1:x2]
        d_zoom = d_sim[y1:y2, x1:x2]
        c_zoom = c_sim[y1:y2, x1:x2]
        cr_zoom = crit[y1:y2, x1:x2]

        f_cls, _ = classify_5class(f_zoom, CLF_THRESH)
        d_cls, _ = classify_5class(d_zoom, D_THRESH)
        c_cls, _ = classify_5class(c_zoom, C_THRESH)

        axes2[0,col].imshow(f_cls, cmap=CLF_CMAP, vmin=0, vmax=4)
        axes2[0,col].set_title(f"{names[col]}", fontweight='bold', fontsize=13)
        axes2[0,col].axis('off')

        axes2[1,col].imshow(c_cls, cmap=C_CMAP, vmin=0, vmax=4)
        axes2[2,col].imshow(d_cls, cmap=D_CMAP, vmin=0, vmax=4)
        axes2[1,col].axis('off')
        axes2[2,col].axis('off')

        axes2[3,col].imshow(cr_zoom, cmap=ListedColormap(['white','red']), alpha=0.8)
        axes2[3,col].axis('off')

    axes2[0,0].set_ylabel('Cropland Fragmentation', fontsize=14, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    axes2[1,0].set_ylabel('Coupling Degree', fontsize=14, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    axes2[2,0].set_ylabel('Coupling Coordination Degree', fontsize=14, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    axes2[3,0].set_ylabel('Critical Regions', fontsize=14, fontweight='bold', fontfamily='Times New Roman', labelpad=15)

    plt.savefig("output_top/Fig_Zoomed_Critical_Region.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✅ All figures generated successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
