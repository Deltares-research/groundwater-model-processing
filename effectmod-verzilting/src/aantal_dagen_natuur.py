# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:26:42 2024

@author: VanWoerkom_T
"""

from pathlib import Path
import pandas as pd
# from tqdm import tqdm
import imod
import xarray as xr
# import geopandas as gpd
import numpy as np

import verzilting_indices_basedata
from dask.diagnostics import ProgressBar

import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake", 
                                          'calculate_aantal_dagen')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output
#%%datapaths

fn_dict = {'lsw_conc': input_fns.lsw_conc_fn,
            'rsgem': input_fns.rsgem_fn,
            'zcrit': input_fns.zcrit_fn,
            'ahn': input_fns.ahn_fn,
            'natuurtypen': input_fns.natuurtypen_fn,
            'lsw': input_fns.lsw_fn}

output_folder = Path(output_fns.natuur_dagen_fn).parent
if not output_folder.is_dir():
    output_folder.mkdir(parents=True)   

#%%preprocessing
base_data = verzilting_indices_basedata.Basemaps(fn_dict)
base_data.lsw_conc = base_data.lsw_conc.chunk(y=100)/1000. #van mg/L naar g/L  

aquatische_typen = ['N01.01', 'N02.01', 'N03.01', 'N04.01', 
                    'N04.02', 'N04.03', 'N04.04', 'N05.01', 
                    'N05.02', 'N05.04', 'N09.01']
base_data.natuurtypen['aquatisch'] = base_data.natuurtypen.beheerType.isin(aquatische_typen)

base_data.lsw = base_data.lsw[['LSWFINAL', 'geometry']]
#%% interne (terrestrische) verzilting
#kans op verdwijnen zoetwaterlens

comb_days = []
for year, yeardata in base_data.rsgem.groupby(base_data.rsgem.time.dt.year):  
    #with ProgressBar():
    #    yeardata = yeardata.compute()
    print(year)
    dates = pd.to_datetime(yeardata.time)
    
    #calculate ghg_zeta
    zeta_rank = yeardata['zeta'].rank('time')
    max_rank = zeta_rank.max('time')
    ghg_zetas = imod.util.where(zeta_rank > (max_rank - 3), yeardata['zeta'], np.nan)
    ghg_zeta = ghg_zetas.mean('time') #ghg of zz grensvlak in meters NAP
    
    diepte_ghg_zeta = base_data.ahn / 100. - ghg_zeta #ahn in cm > resulteerd in diepte_ghg_zeta in meters -mv (positief is onder mv)
    diepte_cap = diepte_ghg_zeta - (base_data.zcrit / 100) #haalt worteldiepte af van ghg_zeta, oftewel: positieve waarden hebben zeta binnen de wortelzone
    
    rwl_yeardata = imod.util.where(diepte_cap <= 1, yeardata.rwl_thick, np.nan) #extra zekerheidsmarge, alleen als het minder dan 1 onder de wortelzone zit
    
    number_days = xr.zeros_like(rwl_yeardata.isel(time=0))

    # Loop through each time step in the selected data
    for t in range(0, len(dates)):
        if t == 0:
            days = dates[t].dayofyear
        elif t == (len(dates)-1):
            days = int((dates[t] - dates[t-1]).days)+(365-dates[t].dayofyear)
        else:
            # Calculate the number of days between consecutive time steps
            days = int((dates[t] - dates[t-1]).days)
        select_day = rwl_yeardata.isel(time=t)

        # Update the number of days over the threshold
        number_days = imod.util.where(select_day < params.rwl_thick_threshold, number_days+days, number_days)
        # number_days[select_day <= conc_treshhold] = number_days[select_day <= conc_treshhold] + days
        
    # Assign the current year to the result and append it to the list
    number_days['year'] = year
    number_days = number_days.drop_vars('time')
    comb_days.append(number_days)
    
internal_days_over_threshold = xr.concat(comb_days, 'year')#.compute()
#%% externe (aquatische) verzilting
#kans op zoutwater in de sloten

comb_days = []
for year, yeardata in base_data.lsw_conc.groupby(base_data.lsw_conc.time.dt.year):    
    print(year)
    dates = pd.to_datetime(yeardata.time)
     
    number_days = xr.zeros_like(yeardata.isel(time=0))

    # Loop through each time step in the selected data
    for t in range(0, len(dates)):
        if t == 0:
            days = dates[t].dayofyear
        elif t == (len(dates)-1):
            days = int((dates[t] - dates[t-1]).days)+(365-dates[t].dayofyear)
        else:
            # Calculate the number of days between consecutive time steps
            days = int((dates[t] - dates[t-1]).days)
        select_day = yeardata.isel(time=t)

        # Update the number of days over the threshold
        number_days = imod.util.where(select_day >= params.conc_treshold, number_days+days, number_days)
        # number_days[select_day <= conc_treshhold] = number_days[select_day <= conc_treshhold] + days
        
    # Assign the current year to the result and append it to the list
    number_days['year'] = year
    number_days = number_days.drop_vars('time')
    comb_days.append(number_days)
    
external_days_over_threshold = xr.concat(comb_days, 'year')#.compute()

#%%sla aantal dagen op
days_over_threshold = xr.Dataset({'internal_days': internal_days_over_threshold,
                                  'external_days': external_days_over_threshold})

with ProgressBar():
    days_over_threshold.to_netcdf(output_fns.natuur_dagen_fn)
