import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import string
import matplotlib.ticker as ticker

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def calculate_strict_metrics(ref_raw, mod_raw, ref_nodata, mod_nodata):
    r = np.copy(ref_raw).astype(float)
    f = np.copy(mod_raw).astype(float)
    if ref_nodata is not None: r[r == ref_nodata] = np.nan
    if mod_nodata is not None: f[f == mod_nodata] = np.nan
    common_mask = ~np.isnan(r) & ~np.isnan(f)
    r_v = r[common_mask]
    f_v = f[common_mask]
    n_points = r_v.size
    std_r = np.std(r_v)
    std_f = np.std(f_v)
    if std_f == 0 or std_r == 0:
        cc = np.nan
    else:
        cc = np.corrcoef(r_v, f_v)[0, 1]
    e_prime = np.sqrt(np.mean(((f_v - np.mean(f_v)) - (r_v - np.mean(r_v))) ** 2))
    return {
        'std_norm': std_f / std_r,
        'corr': cc,
        'rmse_norm': e_prime / std_r,
        'n': n_points
    }

merra_path = r"D:\CMIP6\MERRA-2\MERRA2_SLP_Mean_1980_2000.tif"
cmip6_dir = r"D:\CMIP6\PMIP\Picontrol\psl\psl_interpolated"
with rasterio.open(merra_path) as src:
    ref_vals = src.read(1)
    # MERRA-2 是 Top-down (90 -> 19)，我们在读取后立即翻转它
    ref_vals = np.flipud(ref_vals)
    ref_nd = src.nodata
    ref_shape = src.shape
results = []
tif_files = [f for f in os.listdir(cmip6_dir) if f.endswith('.tif')]
print(f"{'Model':<15} | {'Corr(R)':<8} | {'SD_norm':<8} | {'RMSE_norm':<10} | {'N_points':<8}")
print("-" * 65)

for f_name in tif_files:
    path = os.path.join(cmip6_dir, f_name)
    with rasterio.open(path) as src:
        if src.shape != ref_shape:
            print(f"Skipping {f_name}: Shape mismatch.")
            continue
        mod_vals = src.read(1).astype(float)
        mod_nd = src.nodata
    m = calculate_strict_metrics(ref_vals, mod_vals, ref_nd, mod_nd)
    if m:
        short_name = f_name.split('_')[3].upper()
        results.append({
            'label': short_name,
            'std_n': m['std_norm'],
            'corr': m['corr'],
            'rmse_n': m['rmse_norm']
        })
        print(f"{short_name:<15} | {m['corr']:<8.4f} | {m['std_norm']:<8.4f} | {m['rmse_norm']:<10.4f} | {m['n']:<8}")

if not results:
    print("no result, please check")
else:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    max_std = max([r['std_n'] for r in results] + [1.2])
    ax.set_ylim(0, np.ceil(max_std * 10) / 10)
    letters = string.ascii_uppercase
    mapping_text = []
    for i, res in enumerate(results):
        char = letters[i % 26]
        theta = np.arccos(np.clip(res['corr'], -1, 1))  # 限制范围防止报错
        rho = res['std_n']
        ax.text(theta, rho, f' {char}', fontsize=18, fontweight='bold', va='center', ha='center')
        mapping_text.append(f"{char}: {res['label']}")
    rs_grid = np.linspace(0, ax.get_ylim()[1], 100)
    ts_grid = np.linspace(0, np.pi / 2, 100)
    R_mesh, T_mesh = np.meshgrid(rs_grid, ts_grid)
    dist = np.sqrt(R_mesh ** 2 + 1 ** 2 - 2 * R_mesh * np.cos(T_mesh))
    contours = ax.contour(T_mesh, R_mesh, dist, levels=[0.2, 0.4, 0.6, 0.8, 1.0],
                          colors='green', linestyles='--', linewidths=0.7, alpha=0.8)

    r_ticks = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    r_ticks_minor = np.concatenate([
        np.arange(0.05, 0.9, 0.1),
        np.array([0.925, 0.97, 0.98])
    ])
    tick_angles = np.arccos(r_ticks)
    minor_tick_angles = np.arccos(r_ticks_minor)
    ax.set_xticks(tick_angles)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.set_xticklabels(r_ticks, family='Arial', fontsize=12)
    ax.tick_params(axis='x', which='major', pad=15)
    limit_r = ax.get_ylim()[1]
    length_major = limit_r * 0.02
    for ang in tick_angles:
        ax.plot([ang, ang], [limit_r, limit_r - length_major],
                color='black', linewidth=1, zorder=150)
    length_minor = limit_r * 0.01
    for ang in minor_tick_angles:
        ax.plot([ang, ang], [limit_r, limit_r - length_minor],
                color='black', linewidth=1, zorder=150)

    ax.set_xlabel("Correlation Coefficient (R)", fontweight='bold', labelpad=20)
    ax.set_ylabel("Standardized Deviation (Normalized)", fontweight='bold', labelpad=45)

    plt.figtext(0.5, 0.05, " | ".join(mapping_text), wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig("Taylor_Diagram_SLP.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()