# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:02:04 2024

@author: Woerkom
"""


import imod
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import pandas as pd
import pickle
import xarray as xr

import utils

gpd.options.io_engine = "pyogrio"
plt.rcParams.update({'font.size': 18})

#%% functions
def plot_spatial_index(fn, out_image_fn, deelregio_gdf, ignore_deelregios):
    """
    Plot a spatial index map based on the input file and save the output image.

    Parameters:
    fn (Path): The file path of the input data (either .tif or .gpkg).
    out_image_fn (Path): The file path where the output image will be saved.
    deelregio_gdf (GeoDataFrame): GeoDataFrame containing regional boundaries.
    ignore_deelregios (list): List of region names to ignore in the plot.

    Returns:
    None
    """
    #set figsize
    figsize = (12,10)
    
    #make filename Path items
    fn = Path(fn)
    out_image_fn = Path(out_image_fn)    
    
    # Define categories based on the type of data in the filename
    if 'landbouw' in fn.stem.lower():
        categories = [-0.011,  0.011,  0.056,  0.167]  # Equivalent to -2, 2, 10, 30 days
    if 'natuur' in fn.stem.lower():
        categories = [-0.016,  0.016,  0.082,  0.246]  # Equivalent to -2, 2, 10, 30 days
    klassen = ["Kleiner risico", "Nagenoeg gelijk risico", "Enig groter risico", "Matig groter risico", "Aanzienlijk groter risico"]


    #get colors from cmap
    cmap = ListedColormap(['#0000bf','#0000ff','#6464ff','#b3b3ff','#dfdfff',
                '#c0c0c0','#ffdfdf','#ff9b9b','#ff4f4f','#ff0000','#bf0000'])
    colors = pd.Series({k: c for k, c in zip(klassen, cmap([1,5,6,8,10]))})
    
    # Extract return type and category from the filename
    return_t_type = fn.stem.split('_')[0]
    category = fn.stem.split('_')[1].lower()
    
    # Ensure the output directory exists
    if not out_image_fn.parent.is_dir():
        out_image_fn.parent.mkdir(exist_ok=True, parents=True)
    
    # Handle .tif files
    if fn.suffix == '.tif':
        # Open the raster file
        da = imod.rasterio.open(fn)
        
        # Plot the raster data
        fig, ax, cbar = imod.visualize.spatial.plot_map(da,
                                                        colors.values,
                                                        categories,
                                                        figsize=figsize,
                                                        return_cbar=True)
        # Remove the color bar
        cbar.remove()
        
    # Handle .gpkg files
    elif fn.suffix == '.gpkg':
        # Read the GeoPackage file
        da = gpd.read_file(fn)
        typ = fn.stem.split('_')[0]
        
        # Dissolve the GeoDataFrame by the return type
        da_diss = da.dissolve(return_t_type, as_index=False)
        # Bin the data into categories
        bins = pd.cut(da_diss[typ], bins=np.r_[[-10], categories, [10]], labels=klassen)
        
        # Create a new plot
        fig, ax = plt.subplots(figsize=figsize)
        # Plot the dissolved GeoDataFrame with colors based on bins
        da_diss.plot(ax=ax,
                     color=colors[bins].values,
                     legend=False,
                     aspect=1,
                     edgecolor=colors[bins].values)
    
    # Plot regions to ignore with a specific color
    deelregio_gdf[deelregio_gdf.Naam.isin(ignore_deelregios)].plot(ax=ax,
                                                                   color=colors['Nagenoeg gelijk risico'],
                                                                   linewidth=0)
    # Plot regional boundaries
    deelregio_gdf.boundary.plot(ax=ax,
                                linewidth=0.5,
                                color='k')
    
    # Set plot limits based on the bounding box of the province
    bs = gpd.GeoSeries(deelregio_gdf.unary_union).bounds.iloc[0]
    ax.set_xlim(bs.minx - 5000, bs.maxx + 5000)
    ax.set_ylim(bs.miny - 5000, bs.maxy + 5000)
    
    # Determine the category for the legend title
    if 'natuur' in fn.stem.lower():
        category = 'Natuur'
    if 'landbouw' in fn.stem.lower():
        category = 'Landbouw'

    # Create legend handles
    handles = [Patch(color=c) for c in colors.values]
    ax.legend(handles, colors.index,
              loc='upper left',
              bbox_to_anchor=(1.02, 1),
              title=f'Risicoklassen verzilting {category}')
    
    # Set the title of the plot
    ax.set_title(out_image_fn.stem)
    
    # Save the figure
    fig.savefig(out_image_fn, bbox_inches='tight', dpi=300)
    plt.close()
    #plt.show()

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake_debug", 
                                          'create_spatial_plot')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output
#%%

#%% directe output plots 
deelregio_gdf = gpd.read_file(input_fns.deelregios_fn)

ignore_deelregios = ['Hoge Zandgronden Oost zonder aanvoer', 
                     'Hoge Zandgronden Zuid met aanvoer',
                     'Hoge Zandgronden Zuid zonder aanvoer',
                     'Noord Drenths plateau',
                     'Noord Veenkolonien',
                     'Zuid Limburg']

##################### DIT ZIJN NIET DE GOEDE REGIONAMEN! ##################### nu misschien wel

file_image_dict = {input_fns.average_lsw_file: output_fns.average_img,
                   input_fns.T20_lsw_file: output_fns.T20_img}

for input_fn, output_img_fn in file_image_dict.items():
    plot_spatial_index(input_fn, output_img_fn, deelregio_gdf, ignore_deelregios)



