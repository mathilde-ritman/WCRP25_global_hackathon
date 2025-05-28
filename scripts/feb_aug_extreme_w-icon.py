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
output_dir = f'/work/scratch-nopw2/train045/extreme_updrafts-aug_feb/icon_zoom{zoom}/'
os.makedirs(output_dir, exist_ok=True)

# load time step
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
feb = cat['icon_d3hp003feb'](zoom=zoom, time='PT15M').to_dask().pipe(egh.attach_coords).sel(pressure=slice(10000,100000))
aug = cat['icon_d3hp003aug'](zoom=zoom, time='PT15M').to_dask().pipe(egh.attach_coords).sel(pressure=slice(10000,100000))

# # icon - select 3H times
feb_itr = feb.time[[x in [0,3,6,12,15,18,21] for x in feb.time.dt.hour]][i]
aug_itr = aug.time[[x in [0,3,6,12,15,18,21] for x in aug.time.dt.hour]][i]
feb = feb.sel(time=feb_itr)
aug = aug.sel(time=aug_itr)
logging.warning(f'loaded times')

# mask data below land
aug = aug[['wa']].where(~(aug.zg < feb.orog))
feb = feb[['wa']].where(~(feb.zg < feb.orog))

# compute wmax
logging.warning(f'computing maximum ')
aug_wmax = aug.wa.max('pressure')
feb_wmax = feb.wa.max('pressure')

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
    feb_exceeded = (feb_wmax > val)
    aug_exceeded = (aug_wmax > val)

    # height of max in column
    logging.warning(f'finding the hieghts of those')
    feb_level = feb.wa.where(feb_wmax > val).idxmax('pressure')
    aug_level = aug.wa.where(aug_wmax > val).idxmax('pressure')

    # save
    countdir = output_dir + f'count_{val}ms/'
    leveldir = output_dir + f'level_{val}ms/'
    os.makedirs(countdir, exist_ok=True)
    os.makedirs(leveldir, exist_ok=True)

    logging.warning(f'saving...')
    feb_exceeded.to_zarr(countdir + f'{feb_exceeded.time.dt.strftime("%Y-%m-%dT%H%M").item()}.zarr')
    aug_exceeded.to_zarr(countdir + f'{aug_exceeded.time.dt.strftime("%Y-%m-%dT%H%M").item()}.zarr')
    
    logging.warning(f'saving levels...')
    feb_level.to_zarr(leveldir + f'{feb_exceeded.time.dt.strftime("%Y-%m-%dT%H%M").item()}.zarr')
    aug_level.to_zarr(leveldir + f'{aug_exceeded.time.dt.strftime("%Y-%m-%dT%H%M").item()}.zarr')

    logging.warning('saved')
