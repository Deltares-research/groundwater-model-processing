# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:51:27 2024

@author: Woerkom
"""
from dask.diagnostics import ProgressBar
from pathlib import Path

import verzilting_indices_basedata#_t2 as verzilting_indices_basedata
import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake_debug", 
                                          'run_rsgem')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%%
chunks = {'y': 100}

zw_lens = verzilting_indices_basedata.zoetwaterlens(input_fns.fn_mfnc,
                                                    input_fns.fn_dldrasvat, 
                                                    input_fns.fn_areasvat, 
                                                    input_fns.fn_solbnd, 
                                                    params.minimum_rsgem_years,
                                                    chunks)

svatsel = zw_lens.svat.where(zw_lens.svat["conc_gw"] > params.concentratie)

ds = zw_lens.calculate_rsgem_lhm(zw_lens.input_rsgem, 
                                zw_lens.L2, 
                                svatsel["area"], 
                                params.porosity, 
                                svatsel["conc_rain"], 
                                svatsel["conc_gw"])

output_folder = Path(output_fns.rsgem_output_fn).parent
if not output_folder.is_dir():
    output_folder.mkdir(parents=True)

with ProgressBar():
    ds[['zeta', 'rwl_thick']].to_netcdf(output_fns.rsgem_output_fn)
