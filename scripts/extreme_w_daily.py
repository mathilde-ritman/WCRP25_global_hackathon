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
zoom = 10 ## specify here

# directories
import os
output_dir = f'/work/scratch-nopw2/train045/ral3/v3/extreme_updrafts_zoom{zoom}/'
# output_dir = f'/work/scratch-nopw2/train045/gal9/v2/extreme_wmax_zoom{zoom}/all/'
os.makedirs(output_dir, exist_ok=True)

# load time step
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
ds = cat['um_glm_n2560_RAL3p3'](zoom=zoom, time='PT3H').to_dask().pipe(egh.attach_coords)
# ds = cat['um_glm_n1280_GAL9'](zoom=10, time='PT3H').to_dask().pipe(egh.attach_coords).isel(time=i)
ds = ds.sel(time=slice('2020-4', None), pressure=slice(100, 1000))

# choose month
groups = list(ds.resample(time='1D').groups.keys()) # all month starts
day = pd.to_datetime(groups[i]).date().strftime('%Y-%m-%d')
ds = ds.sel(time=day)

logging.warning('loaded')

# compute wmax
logging.warning(f'computing maximum (day {i})')
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
di = {8:8,}

# - exceedances
for q, val in di.items():
    # count exceedances
    logging.warning(f'finding exceedances of {val}')
    n_exceeded = (wmax > val).sum('time')

    # height of max in column
    # logging.warning(f'finding the hieghts of those')
    # level = ds.wa.where(wmax > val).idxmax('pressure').mean('time')

    # save
    countdir = output_dir + f'count_{val}ms/'
    # leveldir = output_dir + f'level_{val}ms/'
    os.makedirs(countdir, exist_ok=True)
    # os.makedirs(leveldir, exist_ok=True)

    logging.warning(f'saving...')
    n_exceeded.to_zarr(countdir + f'{day}.zarr')
    logging.warning(f'saving levels...')
    # level.to_zarr(leveldir + f'{day}.zarr')

    logging.warning('saved')
