# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:10:57 2023

@author: VanWoerkom_T
"""

from pathlib import Path
import xarray as xr
import numpy as np
import gc

import verzilting_indices_basedata

import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake_debug", 
                                          'preprocess_heads_fluxes')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%%preprocess heads and fluxes
scenario_input_folder = Path(input_fns.scenario_input_folder)
scenarios_input_folder = scenario_input_folder.parent
s = scenario_input_folder.stem

#get years from dummy data
dummy = xr.open_dataarray(input_fns.dummy_fn)
# dummy=imod.idf.open(input_fns.dummy_fn, 
#                       pattern = '{name}_{time}_l{layer}')
years = np.unique(dummy.time.dt.year)

#%% Data inladen
Path(output_fns.fn_mfnc).parent.mkdir(exist_ok=True,parents=True)  

verzilting_indices_basedata.write_modflow_output_nc(scenarios_input_folder, 
                                                    s, 
                                                    years, 
                                                    output_fns.fn_mfnc)

gc.collect()