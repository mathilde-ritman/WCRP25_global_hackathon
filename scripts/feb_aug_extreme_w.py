''' Mathilde Ritman 2025 '''

import cartopy.crs as ccrs
import cartopy.feature as cf
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
import logging
import easygems.healpix as egh

# collect paramters
import sys
i = int(sys.argv[-1])
zoom = 10 ## specify here

# directories
import os
output_dir = f'/work/scratch-nopw2/train045/extreme_updrafts-aug_feb/ral3_zoom{zoom}/'
# output_dir = f'/work/scratch-nopw2/train045/gal9/v2/extreme_wmax_zoom{zoom}/all/'
os.makedirs(output_dir, exist_ok=True)

# load time step
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
ds = cat['um_glm_n2560_RAL3p3'](zoom=zoom, time='PT3H').to_dask().pipe(egh.attach_coords)
# ds = cat['um_glm_n1280_GAL9'](zoom=10, time='PT3H').to_dask().pipe(egh.attach_coords).isel(time=i)

# get months
ds = xr.concat((ds.sel(time='2020-8'), ds.sel(time='2021-2')), dim='time').sel(pressure=slice(100, 1000))

# # choose day
# groups = list(ds.resample(time='1D').groups.keys()) # all day starts
# day = pd.to_datetime(groups[i]).date().strftime('%Y-%m-%d')
# ds = ds.sel(time=day)
# logging.warning(f'loaded day {day}')

# choose time
ds = ds.isel(time=i)
logging.warning(f'loaded time {ds.time.dt.strftime("%Y%m%d %H:%M").item()}')

# mask data below land
ds = ds[['wa']].where(~(ds.zg < ds.orog))

# compute wmax
logging.warning(f'computing maximum ')
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
di = {1:1,5:5}

# - exceedances
for q, val in di.items():
    # count exceedances
    logging.warning(f'finding exceedances of {val}')
    n_exceeded = (wmax > val)

    # height of max in column
    logging.warning(f'finding the hieghts of those')
    level = ds.wa.where(wmax > val).idxmax('pressure')

    # save
    countdir = output_dir + f'count_{val}ms/'
    leveldir = output_dir + f'level_{val}ms/'
    os.makedirs(countdir, exist_ok=True)
    os.makedirs(leveldir, exist_ok=True)

    logging.warning(f'{datetime.now()} rechunking...')
    n_exceeded = n_exceeded.chunk((98304,))
    level = level.chunk((98304,))

    logging.warning(f'{datetime.now()} saving...')
    n_exceeded.to_zarr(countdir + f'{i}.zarr')

    logging.warning(f'saving levels...')
    level.to_zarr(leveldir + f'{i}.zarr')

    logging.warning('saved')
