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
zoom = 9 ## specify here

# directories
import os
output_dir = f'/work/scratch-nopw2/train045/cmf/I_zoom{zoom}/'
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

ds = ds.sel(cell=cells_of_domain(ds, domain_name='tropics'))
ds = ds.sel(pressure=slice(10000, 100000))

# compute density
def density(ds):
    # thermodynamic variables
    p = ds.pressure # Pa (kg m-1 s-2)
    T = ds.ta # K
    Rd = 287.04 # J kg-1 K-1 (m2 s-2 K-1)
    Rv = 461.4 # J kg-1 K-1 (m2 s-2 K-1)
    
    # specific vapour
    q_v = ds.hus # kg kg-1
    
    # eqn state gives
    q_condensate = ds.qall # kg kg-1
    alpha = ((Rv / Rd) - 1) * q_v - q_condensate
    rho = p / (Rd * T * (1 + alpha)) # kg m-3
    return rho

# compute mass flux
cmf = ds.wa * density(ds)

# vertical (upward) integral

del_p = ds.pressure.diff('pressure') # Pa
g = 9.8 # m/s
cmf_I = 1 / g * (ds.wa * np.abs(del_p)).sum('pressure')

# save result
print('saving...')
# cmf_I.to_zarr(output_dir + f'time_{i}.zarr')
# print('saved to', output_dir + f'time_{i}.zarr')

# histogram of all
print('making histogram...')
data = (-cmf).data.flatten() # positive upwards
hist, bins = np.histogram(cmf, bins=40, range=(-15,20))

# save hist
hist_output_dir = f'/work/scratch-nopw2/train045/cmf/hist_{zoom}/'
os.makedirs(hist_output_dir, exist_ok=True)

fname = hist_output_dir + f'time_{i}.npy'
with open(fname, 'wb') as f:
    np.save(f, hist)

if i == 0:
    with open(hist_output_dir + f'bins.npy', 'wb') as f:
        np.save(f, bins)

print('done.')