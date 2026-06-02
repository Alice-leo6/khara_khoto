import xarray as xr
import os

# 1. 定义路径
src_dir = r"D:\CMIP6\PMIP\past1000\ua"
output_dir = r"D:\CMIP6\PMIP\past1000\ua\ua_interpolated_all"
target_file = r"D:\CMIP6\MERRA-2\U_all\MERRA2_100.instM_3d_asm_Np.198001.SUB.nc"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ds_target = xr.open_dataset(target_file)
target_lat = ds_target.lat
target_lon = ds_target.lon
target_plev = ds_target.lev

for filename in os.listdir(src_dir):
    if filename.endswith(".nc"):
        file_path = os.path.join(src_dir, filename)
        # 读取 CMIP6 数据
        ds_src = xr.open_dataset(file_path)
        ds_src['plev'] = ds_src['plev'] / 100.0
        ds_src.plev.attrs['units'] = 'hPa'
        if ds_src.lon.max() > 180:
            ds_src.coords['lon'] = (ds_src.coords['lon'] + 180) % 360 - 180
            ds_src = ds_src.sortby('lon')
        ds_src = ds_src.chunk({'time': 12, 'plev': 1})
        ds_interp = ds_src.interp(lat=target_lat, plev=target_plev, lon=target_lon, method="linear")
        ds_interp.attrs = ds_src.attrs
        output_file = os.path.join(output_dir, f"interp_{filename}")
        ds_interp.to_netcdf(output_file)
        ds_src.close()
        ds_interp.close()