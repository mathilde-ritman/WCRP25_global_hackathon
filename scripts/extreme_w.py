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
# i = int(sys.argv[-1])
zoom = 10 ## specify here

# directories
import os
output_dir = f'/work/scratch-nopw2/train045/ral3/v2/extreme_wmax_zoom{zoom}/'
# output_dir = f'/work/scratch-nopw2/train045/gal9/v2/extreme_wmax_zoom{zoom}/all/'
os.makedirs(output_dir, exist_ok=True)

# load time step
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
ds = cat['um_glm_n2560_RAL3p3'](zoom=10, time='PT3H').to_dask().pipe(egh.attach_coords)
# ds = cat['um_glm_n1280_GAL9'](zoom=10, time='PT3H').to_dask().pipe(egh.attach_coords).isel(time=i)
ds = ds.sel(pressure=slice(100, 1000))

# compute wmax
wmax = ds.wa.max('pressure')

# extreme wmaxs
# - load percentiles
# percentiles = (.995,.99,.95)
# p_dir = f'/work/scratch-nopw2/train045/ral3/wmax_percentiles_zoom{zoom}/'
# di = {}
# for q in percentiles:
#     file = p_dir + f'pi{(q*100):.1}.zarr'
#     di[q] = xr.open_zarr(f).compute().wa

# - use threshold
di = {5:5}

# - exceedances
for q, val in di.items():
    # count exceedances
    n_exceeded = (wmax > val).sum('time')

    # height of max in column
    level = ds.wa.where(wmax > q).idxmax('pressure').mean('time')

    # save
    print('saving...')
    exdir = output_dir + f'count_exceedances/wmax{val}/'
    levdir = output_dir + f'level_exceedances/wmax{val}/'
    os.makedirs(exdir, exist_ok=True)
    os.makedirs(levdir, exist_ok=True)

    # n_exceeded.to_zarr(exdir + f'time_{i}.zarr')
    # level.to_zarr(levdir + f'time_{i}.zarr')
    n_exceeded.to_zarr(output_dir + f'count_exceedances/wmax{val}_alltime.zarr')
    level.to_zarr(output_dir + f'level_exceedances/wmax{val}_alltime.zarr')

    print('saved to', output_dir, 'for time', i)
