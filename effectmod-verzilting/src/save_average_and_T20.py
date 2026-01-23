# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:09:14 2024

@author: Woerkom
"""
from pathlib import Path
import xarray as xr
import geopandas as gpd
import numpy as np
import imod
import pandas as pd

import numpy as np
from pathlib import Path
import imod
import xarray as xr
import geopandas as gpd
import pandas as pd

import utils

#%%functions
def get_return_time_over_years_da(data_per_xy, return_t):
    rank = data_per_xy.argsort()+1
    max_rank = rank.max()
    prob = (max_rank-rank+1)/(max_rank+1)
    return_years = 1/prob
        
    val = np.interp(return_t, sorted(return_years), data_per_xy[rank-1])
    return val

def get_return_time_over_years_gdf(gdf, return_t):
    rank = gdf.rank(axis = 1)
    max_rank = rank.max()
    prob = (max_rank-rank+1)/(max_rank+1)
    return_years = 1/prob

    val = return_years.T.apply(lambda x: np.interp(20, sorted(x), sorted(gdf.loc[x.name])))
    return val
#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_landbouw_snake", 
                                          'save_characteristic_years')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output
#%% T20 values
fn = Path(input_fns.index_fn)
if fn.suffix == '.nc':
    da = xr.open_dataset(fn).totaal
    
    T20 = xr.apply_ufunc(
        get_return_time_over_years_da,
        da,
        20,
        input_core_dims=[["year"], []],
        vectorize=True
        )
    
    imod.rasterio.save(output_fns.T20_out_file, T20)
    
elif fn.suffix == '.gpkg':    
    da_gdf = gpd.read_file(fn, engine = 'pyogrio')
    idx_cols = da_gdf.columns[da_gdf.columns.str.contains('idx_f')]
    da_idx = da_gdf[idx_cols].astype(float)
    
    T20 = get_return_time_over_years_gdf(da_idx, return_t = 20)
    T20_gdf = gpd.GeoDataFrame(T20, geometry = da_gdf.geometry, columns = ['T20'])
    T20_gdf.to_file(output_fns.T20_out_file, engine = 'pyogrio')
    
#%% save average values too
if fn.suffix == '.nc':   
    da = xr.open_dataset(fn).totaal       
    imod.rasterio.save(output_fns.average_out_file, da.mean('year'))
    
elif fn.suffix == '.gpkg':    
    da_gdf = gpd.read_file(fn, engine = 'pyogrio')
    idx_cols = da_gdf.columns[da_gdf.columns.str.contains('idx_f')]
    da_idx = da_gdf[idx_cols].astype(float)
    
    average = da_idx.mean(1)
    average_gdf = gpd.GeoDataFrame(average, geometry = da_gdf.geometry, columns = ['average'])
    average_gdf.to_file(output_fns.average_out_file, engine = 'pyogrio')
