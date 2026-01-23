# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:10:57 2023

@author: VanWoerkom_T
"""
from pathlib import Path
import geopandas as gpd
import imod

import verzilting_indices_basedata
import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake", 
                                          'preprocess_concs')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%% Data inladen
voorbeeld_da = imod.rasterio.open(input_fns.dummy_fn)
lsw_shp = gpd.read_file(input_fns.lsw_fn)
lsw_shp = lsw_shp[["LSWFINAL","DWRN","geometry"]]

#%%################################################################
#### Pre-processing  - om juiste invoer bestanden te maken ##
##################################################################
# Bereken chloride concentratie en maak figuren
# input_fns.mz_input_folder = r'P:/11210039-dpzw-effectmodules/Verzilting/Input/Scenarios/REF2028a/Mozart'
verzilting_indices_basedata.calculate_conc_per_lsw(Path(input_fns.mz_input_folder), 
                                      output_fns.output_conc_fn, 
                                      lsw_shp, 
                                      voorbeeld_da)


