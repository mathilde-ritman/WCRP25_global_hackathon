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
# output_dir = f'/work/scratch-nopw2/train045/percentiles-aug_feb/icon_zoom{zoom}/'
output_dir = f'/work/scratch-nopw2/train045/percentiles-aug_feb/ral3_zoom{zoom}/'
os.makedirs(output_dir, exist_ok=True)

# load all data
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
ds = cat['um_glm_n2560_RAL3p3'](zoom=zoom, time='PT3H').to_dask().pipe(egh.attach_coords)
# feb = cat['icon_d3hp003feb'](zoom=zoom, time='PT15M').to_dask().pipe(egh.attach_coords)
# aug = cat['icon_d3hp003aug'](zoom=zoom, time='PT15M').to_dask().pipe(egh.attach_coords)

# select pressure level
ds = ds.sel(pressure=500)
# feb = feb.sel(pressure=50000)
# aug = aug.sel(pressure=50000)

# sample time from aug and feb
logging.warning('loaded data')
ds = xr.concat((ds.sel(time='2020-8'), ds.sel(time='2021-2')), dim='time').isel(time=i)
# ds = xr.concat((aug, feb), dim='time').isel(time=i)

# mask data below land
ds = ds[['wa']].where(~(ds.zg < ds.orog))

# compute histogram
w = ds.wa.data.flatten()
hist, bins = np.histogram(w, bins=10000, range=(-15,15))

# save result
print('saving...')
fname = output_dir + f'hist_time_{i}.npy'
with open(fname, 'wb') as f:
    np.save(f, hist.compute())

if i == 0:
    with open(output_dir + f'bins.npy', 'wb') as f:
        np.save(f, bins)
print('saved to', fname)

# percentiles
percentiles = (.99,)
for q in percentiles:
    logging.warning(f'computing q={q}')
    pi = ds.wa.quantile(q).compute()
    logging.warning('saving...')
    fname = output_dir + f'pi{(q*100):.1f}_time_{i}.npy'
    with open(fname, 'wb') as f:
        np.save(f, pi)
    logging.warning('saved.')
