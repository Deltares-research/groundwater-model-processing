# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:24:01 2024

@author: Woerkom
"""
import snakemake
from pathlib import Path
import geopandas as gpd
import numpy as np

def read_snakemake_rule(path, name: str) -> "snakemake.rules.Rule":
    """
    Parameters
    ----------
    path: str, pathlib.Path
        The path to the snakefile.
    name: str
        Name of the rule in the snakefile that runs this script.
        
    Returns
    -------
    rule: snakemake.rules.Rules
    
    Examples
    --------
    To run an example both interactively and in a workflow, e.g.:
    
    >>> if "snakemake" not in globals():
    >>>     snakemake = read_snakemake_rule("snakefile", rule="my_rule")
    >>> modelname = snakemake.params.modelname
    >>> template = snakemake.input["template"]
    """
    from snakemake.settings.types import ResourceSettings
    from snakemake.api import SnakemakeApi

    with SnakemakeApi() as snakemake_api:
        workflow = snakemake_api.workflow(
            resource_settings=ResourceSettings(),
            snakefile=Path(path),
        )
        rules = {rule.name: rule for rule in workflow._workflow.rules}
    
    rule = rules.get(name)
    if rule is None:
        raise ValueError(
            f"Rule {name} not in snakefile. Available rules: {', '.join(rules.keys())}")
    return rule

def replace_wildcards_in_snakemake(snakemake_obj, replace_dict):
    import snakemake.io as snakeio

    keys = ['input', 'params', 'output']
    for key in keys:
        if not hasattr(snakemake_obj, key):
            continue
        items = getattr(snakemake_obj, key)

        for item_name in items._names.keys():
            item = getattr(items, item_name)
            if isinstance(item, snakeio._IOFile):
                item_str = str(item)
                for replace_key, replace_val in replace_dict.items():
                    item_str = item_str.replace(replace_key, replace_val)
                replace_item = item_str
            elif isinstance(item, snakeio.Namedlist):
                item = list(item)
                for i in range(len(item)):
                    item_str = str(item[i])
                    for replace_key, replace_val in replace_dict.items():
                        item_str = item_str.replace(replace_key, replace_val)
                    item[i] = item_str
                replace_item = type(item)(item)
            else:
                replace_item = item

            setattr(items, item_name, replace_item)
    return snakemake_obj

def calculate_index(scenario, bias, totaal):
    """
    Calculate an index by adjusting the scenario data with a bias and dividing by a total.

    Parameters
    ----------
    scenario : xarray.DataArray
        An xarray DataArray representing the scenario data.
    bias : xarray.DataArray
        An xarray DataArray representing the bias adjustment to be applied to the scenario data.
    totaal : xarray.DataArray
        An xarray DataArray representing the total or scaling factor to normalize the adjusted scenario data.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the calculated index values, where each element is computed as
        (scenario - bias) / totaal.

    Notes
    -----
    This function assumes that `scenario`, `bias`, and `totaal` have compatible dimensions for broadcasting.
    """
    index = (scenario - bias) / totaal
    return index

def faster_overlay(df_zone, df_data, intsect_above_ncounts=3):
    """
    Perform a faster spatial overlay by joining two GeoDataFrames and 
    updating geometries based on intersection counts.

    Parameters:
    df_zone (GeoDataFrame): The GeoDataFrame containing zone geometries.
    df_data (GeoDataFrame): The GeoDataFrame containing data geometries.
    intsect_above_ncounts (int): The threshold for the number of intersections 
                                 above which geometries are considered shared.

    Returns:
    GeoDataFrame: The resulting GeoDataFrame after performing the overlay.
    """
    # Perform a spatial join between the zone and data GeoDataFrames
    data_join = gpd.sjoin(df_zone, df_data, how='right')

    # Get unique indices and their counts from the joined data
    idx, counts = np.unique(data_join.index, return_counts=True)
    # Identify indices with counts above the specified threshold
    shared_idx = idx[counts > intsect_above_ncounts]

    # Select polygons from the joined data that are shared
    shared_pols = data_join.loc[shared_idx]
    # Select corresponding zones from the zone GeoDataFrame
    shared_zones = df_zone.loc[shared_pols.index_left]
    # Align the indices of shared zones with shared polygons
    shared_zones.index = shared_pols.index

    # Calculate the intersection of shared polygons and zones
    intsect_pols = shared_pols.intersection(shared_zones)
    # Update the geometry of the joined data with the intersection geometries
    data_join.loc[intsect_pols.index.unique(), 'geometry'] = intsect_pols.geometry
    # Drop the 'index_left' column from the joined data
    data_join = data_join.drop('index_left', axis=1)
    
    return data_join
