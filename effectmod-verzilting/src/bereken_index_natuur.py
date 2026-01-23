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

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake", 
                                          'calculate_index')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output
#%% calculate index
# scenarios = np.unique([s.stem[:-1] for s in output_folder.glob('*') if 'REFmedian' not in s.stem and 'Final_idx' not in s.stem])
# s = scenarios[0]

#s = 'REF2028'

keepcoords = np.array(['x', 'y', 'year'])

base_input_folder = Path(r"P:\11210039-dpzw-effectmodules\Verzilting\Input")
statisch_input_folder = base_input_folder / 'Statisch'
natuurtypen_fn = statisch_input_folder / "natuurtypen.shp"
lsw_fn = statisch_input_folder / "lsws.shp"

natuurtypen_gdf = gpd.read_file(natuurtypen_fn)
aquatische_typen = ['N01.01', 'N02.01', 'N03.01', 'N04.01', 
                    'N04.02', 'N04.03', 'N04.04', 'N05.01', 
                    'N05.02', 'N05.04', 'N09.01']
natuurtypen_gdf['aquatisch'] = natuurtypen_gdf.beheerType.isin(aquatische_typen)

da_median = xr.open_dataset(input_fns.ref_output_fn)

lsw_gdf = gpd.read_file(lsw_fn)

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
    
item_da_ds = xr.concat(da_list, dim = 'year')

index = utils.calculate_index(item_da_ds, da_median, 365)
index = index.rename({'internal_days': 'internal_idx', 'external_days': 'external_idx'})

base_rwl = natuurtypen_gdf[natuurtypen_gdf.sensi!=0]

#get xy to shape idx transformation
centerps = base_rwl.centroid
con_idxs = imod.select.points_indices(index.internal_idx, x=centerps.x, y=centerps.y)

internal_idx_shp = index.internal_idx.isel(**con_idxs).to_dataframe()['internal_idx'].unstack('year')
external_idx_shp = index.external_idx.isel(**con_idxs).to_dataframe()['external_idx'].unstack('year')

internal_days_shp = item_da_ds.internal_days.isel(**con_idxs).to_dataframe()['internal_days'].unstack('year')
internal_days_shp.columns = internal_days_shp.columns.astype(str)+'_day_i'

external_days_shp = item_da_ds.external_days.isel(**con_idxs).to_dataframe()['external_days'].unstack('year')
external_days_shp.columns = external_days_shp.columns.astype(str)+'_day_e'

# gevoeligheden per natuurtypen voor intern en extern
#factor bovenop risicokansen
internal_idx_shp_sens = (internal_idx_shp.T*base_rwl.sensi*(~base_rwl.aquatisch)).T
external_idx_shp_sens = (external_idx_shp.T*base_rwl.sensi*base_rwl.aquatisch).T

#gemiddelde: elke feature is nu alleen aquatisch of terrestrisch
#als anders: schaal gevoeligheden gegeven percentage interne (terrestrische) en externe (aquatische) natuur
total_idx_shp_sens = (internal_idx_shp_sens+external_idx_shp_sens)
total_idx_shp_sens = total_idx_shp_sens.loc[:, total_idx_shp_sens.notnull().any()]

external_idx_shp.columns=external_idx_shp.columns.astype(str)+'_idx_e'
internal_idx_shp.columns=internal_idx_shp.columns.astype(str)+'_idx_i'

# total_idx_shp_sens = total_idx_shp_sens[(total_idx_shp_sens!=0).any(1)]
total_idx_shp_sens.columns = total_idx_shp_sens.columns.astype(str)+'_idx_f'

# postprocessing towards shapefile
idx_shp = pd.concat([base_rwl, internal_days_shp, external_days_shp, internal_idx_shp, external_idx_shp, total_idx_shp_sens, ], axis = 1)
idx_shp = gpd.GeoDataFrame(idx_shp, geometry = idx_shp.geometry)
idx_shp = idx_shp[idx_shp.geometry.is_valid]
idx_shp = idx_shp[(idx_shp[total_idx_shp_sens.columns] != 0).any(axis = 1)]
idx_shp = idx_shp[(idx_shp[total_idx_shp_sens.columns].notnull()).any(axis = 1)]


# print('saving', s)
idx_shp.to_file(output_fns.index_fn, engine="pyogrio")