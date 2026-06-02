import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from eofs.xarray import Eof
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cmaps
from scipy.interpolate import make_interp_spline

def make_boundary_path(lon, lat):
    lons, lats = np.meshgrid(lon, lat)
    coords = np.concatenate([
        np.stack([lons[-1, :], lats[-1, :]], axis=1),
        np.stack([lons[::-1, -1], lats[::-1, -1]], axis=1),
        np.stack([lons[0, ::-1], lats[0, ::-1]], axis=1),
        np.stack([lons[:, 0], lats[:, 0]], axis=1)
    ])
    return mpath.Path(coords)
slp_ds = xr.open_dataset(r'D:\CMIP6\PMIP\past1000\psl\psl_Amon_MRI-ESM2-0_past1000_r1i1p1f1_gn_085001-184912.nc')
slp = slp_ds['psl']  # Pa -> hPa
slp_djf = slp.resample(time='QS-DEC').mean().dropna('time')
slp_djf = slp_djf.sel(time=slp_djf['time.month'] == 12)
slp_djf = slp_djf.assign_coords(lon=((slp_djf.lon + 180) % 360) - 180).sortby(['lat', 'lon'])
slp_ao_base = slp_djf.sel(lat=slice(20, 90))
anom = slp_ao_base - slp_ao_base.mean('time')
weights = np.sqrt(np.cos(np.deg2rad(anom.lat)))
anom_w = anom * weights
solver = Eof(anom_w)
eof1 = solver.eofs(neofs=1)[0]
pc1 = solver.pcs(npcs=1, pcscaling=1).values.flatten()

if eof1.sel(lat=slice(80, 90)).mean() > 0:
    pc1 = -pc1
    eof1 = -eof1
ao_idx = xr.DataArray(pc1, coords={'time': anom.time}, dims='time')
ao_idx = (ao_idx - ao_idx.mean())
var_frac = solver.varianceFraction()[0].values * 100
period = slice('1303', '1416')
reg_lat, reg_lon = slice(20, 80), slice(-80, 30)

slp_sel = anom.sel(time=period, lat=reg_lat, lon=reg_lon)
ao_sel = ao_idx.sel(time=period)
ao_sel, slp_sel = xr.align(ao_sel, slp_sel)

def get_regression(x, y):
    return np.polyfit(x, y, 1)[0]
reg_map = xr.apply_ufunc(get_regression, ao_sel, slp_sel,
                         input_core_dims=[['time'], ['time']], vectorize=True)

nboot = 800
boot_results = []
for _ in range(nboot):
    idx = np.random.choice(len(ao_sel), len(ao_sel), replace=True)
    res = xr.apply_ufunc(get_regression, ao_sel.isel(time=idx), slp_sel.isel(time=idx),
                         input_core_dims=[['time'], ['time']], vectorize=True)
    boot_results.append(res)
boot_all = xr.concat(boot_results, dim='boot')
sig95 = (boot_all.quantile(0.025, 'boot') > 0) | (boot_all.quantile(0.975, 'boot') < 0)
sig90 = (boot_all.quantile(0.05, 'boot') > 0) | (boot_all.quantile(0.95, 'boot') < 0)

fig = plt.figure(figsize=(10, 12), dpi=150)
gs = GridSpec(2, 1, height_ratios=[1.2, 0.8], hspace=0.3)
proj = ccrs.Orthographic(central_longitude=-25, central_latitude=50)
ax1 = fig.add_subplot(gs[0], projection=proj)
ax1.set_extent([-80, 30, 20, 80], crs=ccrs.PlateCarree())
ax1.set_boundary(make_boundary_path(reg_map.lon.values, reg_map.lat.values), transform=ccrs.PlateCarree())
ax1.coastlines(linewidth=1)
gl = ax1.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.6,
    color='gray',
    alpha=0.6,
    linestyle='--'
)
gl.xlocator = plt.FixedLocator(np.arange(-80, 31, 20))
gl.ylocator = plt.FixedLocator(np.arange(20, 81, 10))
gl.xlabel_style = {'size': 9, 'color': 'black'}
gl.ylabel_style = {'size': 9, 'color': 'black'}
gl.xpadding = 5
gl.ypadding = 5
cf = reg_map.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), levels = np.linspace(-400, 400, 17),
                           cmap='RdBu_r', extend='both', add_colorbar=False)

step = 1
lon2d, lat2d = np.meshgrid(reg_map.lon.values, reg_map.lat.values)
sig95_np = sig95.values
sig90_np = sig90.values
sig_only_90 = sig90_np & (~sig95_np)
lon_sub = lon2d[::step, ::step]
lat_sub = lat2d[::step, ::step]
sig95_sub = sig95_np[::step, ::step]
sig90_sub = sig_only_90[::step, ::step]
ax1.scatter(
    lon_sub[sig95_sub],
    lat_sub[sig95_sub],
    transform=ccrs.PlateCarree(),
    s=0.4,
    c='black',
    marker='o',
    alpha=0.8,
    zorder=10
)
ax1.scatter(
    lon_sub[sig90_sub],
    lat_sub[sig90_sub],
    transform=ccrs.PlateCarree(),
    s=0.4,
    c='blue',
    marker='o',
    alpha=0.8,
    zorder=9
)
ax1.set_title(f'SLP Regression on AO Index ({period.start}-{period.stop})\nExplained Variance: {var_frac:.1f}%', fontsize=12)
plt.colorbar(cf, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.7, label=r'hPa')
ax2 = fig.add_subplot(gs[1])
ax2.plot(ao_idx['time.year'], ao_idx, color='black', lw=0.8)
ax2.axvspan(int(period.start), int(period.stop), color='red', alpha=0.15, label='Collapse Period')
ax2.set_xlim(1200, 1600)
ax2.set_ylabel('Standardized AO Index')
ax2.legend(loc='upper right')
plt.savefig(r'D:\Pycharm\pythonproject\AO_Regression_Map.pdf',
            dpi=400, bbox_inches='tight', format='pdf')
plt.show()
