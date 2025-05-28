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

# load ds
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
ds = cat['um_glm_n2560_RAL3p3'](zoom=zoom, time='PT3H').to_dask().pipe(egh.attach_coords)
# ds = cat['um_glm_n1280_GAL9'](zoom=10, time='PT3H').to_dask().pipe(egh.attach_coords).isel(time=i)
ds = ds.sel(time=slice('2020-4', None))

# choose month
groups = list(ds.resample(time='1MS').groups.keys()) # all month starts
month = pd.to_datetime(groups[i]).date().strftime('%Y-%m')

# load daily results for month
val = 8
rdir = output_dir + f'count_{val}ms/{month}*'
files = glob.glob(rdir)
li = []
for f in files:
    d = xr.open_zarr(f)
    li.append(d)
ds = sum(li)
logging.warning('loaded results')

# save
logging.warning(f'saving (month {month})')
countdir = output_dir + f'count_{val}ms_monthly/'
os.makedirs(countdir, exist_ok=True)
ds.to_zarr(countdir + f'{month}.zarr')
logging.warning('saved')
