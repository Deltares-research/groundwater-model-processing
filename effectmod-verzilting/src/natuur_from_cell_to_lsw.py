# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:58:35 2024

@author: Woerkom
"""
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

gpd.options.io_engine = "pyogrio"

import utils

#%% functions
def weighted_average_lsw(file, lsw_gdf):
    """
    Calculate the weighted average of a specified attribute from a GeoDataFrame, 
    overlaying it with another GeoDataFrame and grouping by 'LSWFINAL'.

    Parameters:
    file (str): The filename of the GeoDataFrame to be read.
    lsw_gdf (GeoDataFrame): The GeoDataFrame containing 'LSWFINAL' to overlay with.

    Returns:
    GeoDataFrame: A GeoDataFrame with the weighted average of the specified attribute 
                  and the combined geometry for each 'LSWFINAL' group.
    """
    file = Path(file)
    
    # Extract the type from the filename
    typ = file.stem.split('_')[0]
    
    # Read the GeoDataFrame from the file
    natuur_gdf = gpd.read_file(file)
    
    # Overlay the two GeoDataFrames
    lsw_natuur_gdf = gpd.overlay(lsw_gdf, natuur_gdf, how='union')
    # Assign unique 'LSWFINAL' values to null entries
    lsw_natuur_gdf.loc[lsw_natuur_gdf['LSWFINAL'].isnull(), 'LSWFINAL'] = lsw_natuur_gdf['LSWFINAL'].max() + 1 + np.arange(lsw_natuur_gdf['LSWFINAL'].isnull().sum())
    # Fill null values in the specified attribute with 0
    lsw_natuur_gdf[typ] = lsw_natuur_gdf[typ].fillna(0)
    
    # Group by 'LSWFINAL' and calculate the weighted average and combined geometry
    lsw_with_data = lsw_natuur_gdf.groupby('LSWFINAL').apply(
        lambda x: pd.Series([np.average(x[typ], weights=x.area), x.geometry.unary_union],
                            index=[typ, 'geometry']),
        include_groups=True
    )
    
    return lsw_with_data

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake_debug", 
                                          'calculate_per_lsw')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%%
output_folder = Path(r'P:\11210039-dpzw-effectmodules\Verzilting\Output\Final_idx')
files = ['average_natuur_cell.gpkg', 'T20_natuur_cell.gpkg']

lsw_gdf = gpd.read_file(input_fns.lsw_fn)
lsw_gdf = lsw_gdf[['LSWFINAL', 'geometry']]

#for average map
lsw_average_data = weighted_average_lsw(input_fns.average_cell_file, lsw_gdf)
lsw_average_data.to_file(output_fns.average_lsw_file)

#for T20 map
lsw_T20_data = weighted_average_lsw(input_fns.T20_cell_file, lsw_gdf)
lsw_T20_data.to_file(output_fns.T20_lsw_file)
  
