# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:21:15 2024

@author: Woerkom
"""
import numpy as np
from pathlib import Path
import imod
import xarray as xr
import geopandas as gpd
import pandas as pd

import utils

import dask
from dask.diagnostics import ProgressBar
dask.config.set({"array.slicing.split_large_chunks": True})
#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_landbouw_snake_debug", 
                                          'calculate_index')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%%calculate indices landbouw
keepcoords = np.array(['x', 'y', 'year'])
da_median = xr.open_dataset(input_fns.ref_output_fn)

print('combine scenarios')
da_list = []
for data_fn in input_fns.basis_sce_fns:
    data_fn = Path(data_fn)
    sub_sce = data_fn.parent.parent.stem
    da = xr.open_dataset(data_fn).chunk(year=30)
    
    coords = np.array(list(da.coords.keys()))
    drop_coords = coords[~np.isin(coords, keepcoords)]
    da = da.drop_vars(drop_coords)
    da['year'] = da['year'].astype(str).str+f'_{sub_sce}'
    
    da_list.append(da)
# item_da_ds = xr.Dataset(da_list)
item_da_ds = xr.merge(da_list)

print('calculate index')
index = utils.calculate_index(item_da_ds, da_median, 180)

item_names =  iter(index.data_vars.keys())

index = index.assign(totaal = index['aantal_dagen_beregening_overschrijding_gewas']+index['dagen_zonder_zwlens'])

save_index_fn = Path(output_fns.index_fn)
if not save_index_fn.parent.is_dir():
    save_index_fn.parent.mkdir(parents=True, exist_ok = True)

print('save output')
with ProgressBar():
    index.to_netcdf(save_index_fn)
