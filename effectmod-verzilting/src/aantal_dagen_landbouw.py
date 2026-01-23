# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:10:57 2023

@author: VanWoerkom_T
"""
#%%

from pathlib import Path
import imod
import xarray as xr
import numpy as np

import verzilting_indices_basedata
from dask.diagnostics import ProgressBar

import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"C:\git_repos\groundwater-model-processing\effectmod-verzilting\src\snake\verzilting_effectmodule_landbouw_snake", 
                                          'calculate_aantal_dagen')
    snakemake = utils.replace_wildcards_in_snakemake(snakemake,
                                                     {'{scenario}': 'REF2028',
                                                      '{x}': 'a'})
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%% open needed files
basemap_files = { 'CONC': input_fns.conc_fn,
                  'beregening_data': Path(input_fns.beregening_data) / 'bdgPssw*.idf',
                  'area': input_fns.ms_area_fn,
                  'lgn': input_fns.lgn_fn,
                  'voorbeeld_da': input_fns.lgn_fn,
                  'shp': input_fns.lsw_fn,
                  'ahn': input_fns.ahn_fn}

print(basemap_files)
    
# chunks = {'x': 10, 'y': 10}
chunks = {'y': 100}
basemap_data = verzilting_indices_basedata.Basemaps(basemap_files)

z_afstand = imod.rasterio.open(input_fns.zcrit_fn)#.compute()
da_lsws =  imod.prepare.rasterize(basemap_data.shp, z_afstand ,'LSWFINAL')

beregening_idf = imod.rasterio.open(input_fns.beregening_idf_fn)#.compute()

dummy = xr.open_dataset(input_fns.fn_mfnc)
years = np.unique(dummy.time.dt.year)
#%% bereken aantal dagen oppervlaktewater
# basemap_data = verzilting_indices_basedata.Basemaps(basemap_files)
basis_beregening = verzilting_indices_basedata.Beregening(basemap_data, years)
   
# Maak threshold maps van scenario
result_maps = basis_beregening.calculate_threshold_maps_for_indices(threshold = params.concentratie)    # concentration in g/L
with ProgressBar():
    result_maps.to_netcdf(output_fns.dagen_beregening_fn)

#%% bereken dagen zoetwaterlens
zw_lens = verzilting_indices_basedata.zoetwaterlens(input_fns.fn_mfnc,
                                                    input_fns.fn_dldrasvat, 
                                                    input_fns.fn_areasvat, 
                                                    input_fns.fn_solbnd, 
                                                    params.minimum_rsgem_years,
                                                    chunks)
        
ds = xr.open_dataset(input_fns.rsgem_fn)
#with ProgressBar():
print('make masks')
mask_risico_gebied, mask_agrarisch_gebied, mask_beregening = zw_lens.create_masks(ds, 
                                                                                  z_afstand, 
                                                                                  params.LGN_numbers,
                                                                                  beregening_idf, 
                                                                                  years, 
                                                                                  basemap_data)
print('calc days over threshold')    
zw_lens.calculate_n_days_above_threshold(params.rwl_thick_threshold, 
                                            years,
                                            ds, 
                                            mask_risico_gebied, 
                                            mask_agrarisch_gebied, 
                                            mask_beregening, 
                                            zw_lens.zeta_max)
    
zw_lens.n_days_over_threshold = zw_lens.n_days_over_threshold.compute()
dagen_LSW = verzilting_indices_basedata.average_per_lsw(zw_lens.n_days_over_threshold, da_lsws)
dagen_LSW = dagen_LSW.to_dataset(name = 'dagen_zonder_zwlens')
print('start saving netcdf')
dagen_LSW.to_netcdf(output_fns.dagen_zw_lens_fn)

