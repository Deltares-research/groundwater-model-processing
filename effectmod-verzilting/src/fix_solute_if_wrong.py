# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:49:06 2024

@author: Woerkom
"""

import verzilting_indices_basedata#_t2 as verzilting_indices_basedata
import utils

#%% get snakemake params if local
if "snakemake" not in globals():
    snakemake = utils.read_snakemake_rule(r"P:\11210039-dpzw-effectmodules\Verzilting\src\workflow\verzilting_effectmodule_natuur_snake_debug", 
                                          'fix_solute_bnd')
    
input_fns = snakemake.input
params  = snakemake.params
output_fns = snakemake.output

#%% check if soluted_bnd is still in old (wrong) format
#change if that is, keep current file if already updated 

verzilting_indices_basedata.fix_fn_solbnd_merge(input_fns.fn_solbnd)