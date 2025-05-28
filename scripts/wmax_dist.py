''' Mathilde Ritman 2025 '''

import cartopy.crs as ccrs
import cartopy.feature as cf
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import logging
import easygems.healpix as egh

# collect paramters
import sys
i = int(sys.argv[-1])
zoom = 6 ## specify here

# directories
import os
savedir = '/home/users/train045/Documents/WCRP25_hackathon/figs/w_dists/'
output_dir = f'/work/scratch-nopw2/train045/wmax_zoom{zoom}/'
os.makedirs(output_dir, exist_ok=True)

# load time step
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
ds = cat['icon_d3hp003aug'](zoom=zoom, time='PT15M').to_dask().pipe(egh.attach_coords).isel(time=i)

# subselect the tropics
domain_extents = {"tropics": (0, 360, -30.1, 30.1),}
def cells_of_domain(ds, domain_name):
    lon_min, lon_max, lat_min, lat_max = domain_extents[domain_name]
    cells = ds.cell
    c1 = cells.where(ds.lon>lon_min).where(ds.lon<lon_max).where(ds.lat>lat_min).where(ds.lat<lat_max)
    return c1.dropna('cell')

trop = ds.sel(cell=cells_of_domain(ds, domain_name='tropics'))

# compute tropospheric max w
w = trop.wa.sel(pressure=slice(10000, 100000)).max('pressure')

# save result
print('saving...')
w.to_zarr(output_dir + f'time_{i}.zarr')
print('saved to', output_dir + f'time_{i}.zarr')
