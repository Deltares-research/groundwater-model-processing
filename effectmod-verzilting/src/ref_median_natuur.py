# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:21:15 2024

@author: Woerkom
"""
import numpy as np
import xarray as xr

import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake", 
                                          'calculate_ref_median')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%%create baseline voor als gemiddelde om mee te vergelijken, dus alleen runnen op referentie scenario
#op basis van emailwisseling met femke en ilja dinsdag 30-4-2024: mediaan van ref

keepcoords = np.array(['x', 'y', 'year'])

da_list = []
for data_fn in input_fns.basis_ref_fns:
    da = xr.open_dataset(data_fn).chunk(year=30)
    
    coords = np.array(list(da.coords.keys()))
    drop_coords = coords[~np.isin(coords, keepcoords)]
    da = da.drop_vars(drop_coords)
    
    da_list.append(da)

da_all = xr.concat(da_list, 'year')        
da_median = da_all.median('year') 
da_median.to_netcdf(output_fns.ref_output_fn)   