import xarray as xr
import numpy as np
import imod
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numba
from tqdm import tqdm
import dask
from datetime import datetime
import mozart as mz
from numba import float32, guvectorize, njit

dask.config.set({"array.slicing.split_large_chunks": True})

import warnings
from dask.diagnostics import ProgressBar


#%%  Critical concentration values for irrigation - Stuyt 2016
drempel_dict = {}
drempel_dict['gras'] = [2055, 2711]
drempel_dict['mais'] = [498, 371]
drempel_dict['aardappelen'] = [838, 668]
drempel_dict['bieten'] = [1478, 817]
drempel_dict['granen'] = [2626, 2796]
drempel_dict['overig'] = [963, 817]
drempel_dict['boomteelt'] = [233, 62]
drempel_dict['boomgaard'] = [465, 381]
drempel_dict['bollen'] = [475, 468]
drempel_dict['fruitkwekerij'] = [300, 247] #std van fruitkwekerij is geschat door 300*(drempel_df['std']/drempel_df['mean']).mean()

drempel_df = pd.DataFrame(drempel_dict, index = ['mean', 'std']).T
drempel_df['LGN_number'] = [1,2,3,4,5,6,7,9,10,21]

salt_tolerant_limits = drempel_df['mean']

#%% Functions
class Basemaps:
    """
    Class for managing various base map data, including IDF files and shapefiles.

    Parameters:
        filepath_dict (dict): A dictionary mapping names to filepaths for different base map data.

    Attributes:
        Each key-value pair in `filepath_dict` is added as an attribute to the class instance.
        For IDF files, the data is loaded using imod.idf.open or imod.idf.open_subdomains based on the file name.
        For shapefiles, the data is loaded using geopandas.read_file.

        Example:
        basemaps_instance = Basemaps({'idf_data': 'path/to/idf_data.idf', 'shapefile_data': 'path/to/shapefile.shp'})
        print(basemaps_instance.{idf_name}_data)  # Access the IDF data
        print(basemaps_instance.idf_name}_data_path)  # Access the path to the IDF data file
        print(basemaps_instance.{shape_name}_data)  # Access the shapefile data
        print(basemaps_instance.{shape_name}_data_path)  # Access the path to the shapefile data file
    """

    def __init__(self, filepath_dict):
        """
        Initialize the Basemaps class with base map data from provided filepaths.

        Parameters:
            filepath_dict (dict): A dictionary mapping names to filepaths for different base map data.
        """
        # Loop through each key-value pair in the filepath dictionary
        for name, filepath in filepath_dict.items():
            filepath = Path(filepath)
            # Check if the file has a .idf extension
            if filepath.suffix.lower() == '.idf':
                print(filepath.stem)
                # Check if 'p00' is in the filename to determine whether to use open or open_subdomains
                if 'p00' in filepath.stem:
                    data = imod.idf.open_subdomains(filepath)
                else:
                    data = imod.idf.open(filepath)
            # Check if the file has a .shp extension
            elif filepath.suffix.lower() == '.shp':
                data = gpd.read_file(filepath)
            elif filepath.suffix.lower() == '.asc':
                data = imod.rasterio.open(filepath)
            elif filepath.suffix.lower() == '.nc':
                data = xr.open_dataset(filepath)
                if len(data.data_vars) == 1:
                    data = xr.open_dataarray(filepath)
            else:
                # Handle other file types if needed
                raise ValueError(f"Unsupported file type for {name}: {filepath.suffix}")
            
            # Set attributes with the loaded data and file path
            setattr(self, name, data)
            setattr(self, f'{name}_path', filepath)

def write_modflow_output_nc(path_runs, run, years, fn_mfnc):
    """
    Read MODFLOW output IDF files for specified parameters, process the data, and save the result as a NetCDF file.

    Parameters:
        path_runs (Path): Directory path containing MODFLOW run results.
        run (str): Name of the specific MODFLOW run.
        years (list): List of years to include in the output.
        fn_mfnc (Path): Output NetCDF file path.

    Returns:
        None
    """
    # List of MODFLOW parameters to process
    prms = ["bdgriv_sys1", "bdgriv_sys2", "bdgriv_sys3", "bdgdrn_sys1", "bdgdrn_sys3", "bdgflf", "head"]
    
    # Dictionary to store processed MODFLOW results
    mfresults = {}
    
    # Process each parameter
    for prm in prms:
        print(f"- {prm}")
        # Extract the subdirectory from the parameter name
        prmdir = prm.split("_")[0]
        
        # Construct the path pattern for IDF files
        da_path = path_runs / f"{run}/Modflow_{prmdir}/{prm}_??????[12][48]_l1*.idf"
        
        # Open and process the IDF files for the parameter
        da = imod.idf.open(da_path).chunk(time=200)
        da = da.squeeze(dim="layer", drop=True)
        
        # Select data for specified days and years
        da = da.sel(time=da.time.dt.day.isin([14, 28]) & da.time.dt.year.isin(years))
        
        # Store the processed data in the dictionary
        with ProgressBar():   
            mfresults[prm] = da.compute().transpose('y', 'x', 'time').chunk()
    
    # Create a dataset from the processed results
    input_rsgem = xr.Dataset(mfresults)
    # input_rsgem = input_rsgem
    
    with ProgressBar():   
        # Save the dataset as a NetCDF file
        input_rsgem.to_netcdf(fn_mfnc)
        
def calculate_conc_per_lsw(input_dir, output_fn, lsw_shp, voorbeeld_da):
    """
    Calculate concentration of a specific land surface water (LSW) for each date and save the result as IDF files.

    Parameters:
        input_dir (Path): Input directory containing the necessary files.
        output_fn (Path): Output file where nc will be saved.
        lsw_shp (GeoDataFrame): GeoDataFrame containing the shapefile of lsw.
        voorbeeld_da (xarray.DataArray): DataArray for reference, used for rasterization.

    Returns:
        None
    """
    output_dir = Path(output_fn).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    input_dir = Path(input_dir)
    
    all_lsw_concs = []
    # Read the input file containing concentration values
    for fn in input_dir.glob('lswvalue_*.out'):
        print(fn.stem)
        value =  mz.read_mzlswvalue(fn)
        value['LSWNR'] = value['local_surface_water_code']
                               
        # Convert date columns to datetime format
        value['time_start'] = pd.to_datetime(value['time_start'], format='%Y%m%d')
        value['time_end'] = pd.to_datetime(value['time_end'], format='%Y%m%d')
        
        # Extract relevant columns for further processing
        value_small = value[['LSWNR', 'time_start', 'time_end', 'concentration']]
        
        # Save output for every date
        Data = value_small.copy()
        dates = Data['time_start'].unique()
    
        for date in dates:
            # Select data for the specific date
            Data_select = Data[Data['time_start'] == date]
            Data_select = Data_select.set_axis(['LSWFINAL', 'TIMESTART', 'TIMEEND', 'concentration'], axis=1) 
    
            # Merge with land surface water shapefile
            result = lsw_shp.merge(Data_select, on="LSWFINAL").reset_index()
            
            # Select relevant columns for output
            result = result[["LSWFINAL", "DWRN", "concentration", "geometry"]]
            
            # Rasterize the result based on the reference DataArray
            result_grid = imod.prepare.rasterize(result, like=voorbeeld_da, column="concentration")
            result_grid['time'] = date
            
            all_lsw_concs.append(result_grid.chunk())
    
    all_lsw_concs_da = xr.concat(all_lsw_concs, dim = 'time')
    all_lsw_concs_da = all_lsw_concs_da.groupby('time').mean()
    all_lsw_concs_da.to_netcdf(output_fn)
    
def merged_grids_vergroten(da, voorbeeldextent):
    """
    Enlarge the extent of a data array after loading .p00x grids with imod-python.

    Parameters:
        da (xr.DataArray): DataArray to be corrected.
        voorbeeldextent (xr.DataArray): Example extent for correction.

    Returns:
        da_extent (xr.DataArray): DataArray with corrected extent.
    """
    # Replace grid values with NaN
    voorbeeldextent_nan = xr.full_like(voorbeeldextent, np.nan)
    
    # Enlarge the extent of the data array
    da_extent_nan = voorbeeldextent_nan.combine_first(da)
    
    # Replace the NaN values with 0
    da_extent = da_extent_nan.fillna(0)
    
    return da_extent

def average_per_lsw(da, da_lsws):
    """
    Calculate the average per land surface water (LSW) from a given DataArray.

    Parameters:
        da (xr.DataArray): DataArray to calculate the average for.
        da_lsws (xr.DataArray): DataArray representing land surface waters.

    Returns:
        out (xr.DataArray): DataArray containing the average per LSW.
    """
    # Ensure there are no NaN values, as they may interfere with the groupby operation
    try:
        da = da.isel(layer=0)
    except:
        pass

    # Convert land surface water DataArray to integers, filling NaN values with 100
    int_lsw = da_lsws.fillna(100).astype(int)
    
    # Group the original DataArray by land surface water, calculating the mean for each group
    variabele_per_lsw = da.groupby(int_lsw).mean()

    # Create a grid of land surface waters with a value of 1
    da_lsws_een = da_lsws * 0 + 1
    
    # Select the group based on the integer representation of land surface waters
    out = variabele_per_lsw.sel(group=int_lsw)
    
    # Set values to NaN where original land surface waters are null, and multiply by the grid of ones
    out = out.where(da_lsws.notnull()) * da_lsws_een
    
    return out



class Beregening:
    def __init__(self, basemap_data, years):
        """
        Initialize the Beregening class.

        Parameters:
            basemap_data (object): Object containing basemap data, including CONC, beregening_data, transpiratie_pot, area, lgn, voorbeeld_da, shp, and da_lsws.
            years (list): List of years to include in the analysis.
        """
        self.basemaps = basemap_data
        self.preprocess_data(years)
    
    def preprocess_data(self, years):
        """
        Preprocess the basemap data for the specified years.

        Parameters:
            years (list): List of years to include in the analysis.

        Returns:
            None
        """
        # Select data for the specified years and compute if needed
        self.basemaps.CONC = self.basemaps.CONC.sel(time=self.basemaps.CONC.time.dt.year.isin(years)).chunk(y=200)#.compute()
        self.basemaps.beregening_data = self.basemaps.beregening_data.sel(time=self.basemaps.beregening_data.time.dt.year.isin(years))#.compute()
        self.basemaps.transpiratie_pot = self.basemaps.transpiratie_pot.sel(time=self.basemaps.transpiratie_pot.time.dt.year.isin(years))#.compute()
        self.basemaps.area = self.basemaps.area#.compute()
        self.basemaps.lgn = self.basemaps.lgn#.compute()
        self.basemaps.voorbeeld_da = self.basemaps.voorbeeld_da#.compute()

        # Rasterize land surface water shapefile
        self.basemaps.da_lsws = imod.prepare.spatial.rasterize(self.basemaps.shp, like=self.basemaps.voorbeeld_da, column="LSWFINAL")
        self.basemaps.shp = self.basemaps.shp[["LSWFINAL", "DWRN", "geometry"]]

        # Define growing season
        self.groeiseizoen = (self.basemaps.CONC.time.dt.month >= 4) & (self.basemaps.CONC.time.dt.month < 10)

        # Calculate total irrigation and transpiration for the growing season
        # beregening_totaal = self.basemaps.beregening_data.sel(time=self.groeiseizoen)
        # beregening_totaal = beregening_totaal.groupby(beregening_totaal.time.dt.year).sum()
        # self.beregening_totaal = ((beregening_totaal * self.basemaps.area) / 10000) * 1000  # mm/ha/summer-half-year

        # transpiratie_totaal = self.basemaps.transpiratie_pot.sel(time=self.groeiseizoen)
        # transpiratie_totaal = transpiratie_totaal.groupby(transpiratie_totaal.time.dt.year).sum()
        # self.transpiratie_totaal = ((transpiratie_totaal * self.basemaps.area) / 10000) * 1000  # mm/ha/summer-half-year

        # Set time coordinate of beregening_data to match CONC
        self.basemaps.beregening_data['time'] = self.basemaps.CONC.time

        # Initialize and apply regridder
        self.regridder = imod.prepare.Regridder('max_overlap')
        self.basemaps.CONC = self.regridder.regrid(self.basemaps.CONC, self.basemaps.voorbeeld_da)
        self.basemaps.beregening_data = self.regridder.regrid(self.basemaps.beregening_data, self.basemaps.voorbeeld_da)
        self.basemaps.transpiratie_pot = self.regridder.regrid(self.basemaps.transpiratie_pot, self.basemaps.voorbeeld_da)

    
    def calculate_threshold_maps_for_indices(self,threshold):
        """
        Calculate various threshold maps based on salt-tolerant limits.
    
        Parameters:
            OLD method: salt_tolerant_limits (pd.Series): Salt-tolerant limits for different LGN landuse classes.
            Salt-tolerant limit = threshold values
    
        Returns:
            result_maps (xr.Dataset): Dataset containing various threshold maps.
        """
        # Calculate the threshold map based on salt-tolerant limits
        #self.calculate_drempel(salt_tolerant_limits)
    
        # # OLD METHOD -- Calculate damage due to irrigation exceeding the threshold 
        # beregening_schade = xr.where(((self.basemaps.CONC - self.drempel_map) > 0) & self.groeiseizoen,
        #                              self.basemaps.beregening_data, np.nan)
        
        # Calculate damage due to irrigation exceeding 1000 mg/L
        # beregening_schade = xr.where((self.basemaps.CONC > threshold * 1000) & self.groeiseizoen,    # conversion from g/L to mg/l
        #                              self.basemaps.beregening_data, np.nan)
        # beregening_schade = beregening_schade.groupby(beregening_schade.time.dt.year).sum()
        # beregening_schade = ((beregening_schade * self.basemaps.area) / 10000) * 1000  # mm/ha/summer-half-year
    
        # Calculate the number of days the irrigation exceeds the threshold
        day_roll = self.basemaps.CONC.time - self.basemaps.CONC.time.roll({'time': -1})
        number_days_gewas = xr.where((self.basemaps.CONC > (threshold * 1000)) &
                                     (self.basemaps.beregening_data > 0) & self.groeiseizoen, day_roll.dt.days, 0)
        number_days_gewas = number_days_gewas.groupby(number_days_gewas.time.dt.year).sum()
        number_days_gewas = abs(number_days_gewas).sel(layer=1)
    
        # Correct size if too big
        self.basemaps.da_lsws = self.basemaps.da_lsws.sel(x=self.basemaps.area.x, y=self.basemaps.area.y)
        number_days_gewas = number_days_gewas.sel(x=self.basemaps.area.x, y=self.basemaps.area.y)
    
        # Calculate average damage, number of days, total irrigation, and total transpiration per land surface water
        #brakke_beregening = average_per_lsw(beregening_schade, self.basemaps.da_lsws)
        dagen_beregening = average_per_lsw(number_days_gewas, self.basemaps.da_lsws)
        #totaal_beregening = average_per_lsw(self.beregening_totaal, self.basemaps.da_lsws)
        #totaal_transpiratie = average_per_lsw(self.transpiratie_totaal, self.basemaps.da_lsws)
    
        # Create a dataset with the calculated threshold maps
        result_maps = xr.Dataset({#'Beregening_mmha_zomer_overschrijding_gewascriteria': brakke_beregening,
                                  'aantal_dagen_beregening_overschrijding_gewas': dagen_beregening,
                                  #'Beregening_totaal_mmha_zomer': totaal_beregening,
                                  #'Transpiratie_totaal_mmha_zomer': totaal_transpiratie
                                  })
    
        return result_maps
    
    # calculate drempelmap
    def calculate_drempel(self, salt_tolerant_limits):
        """
        Calculate the threshold map based on salt-tolerant limits.
    
        Parameters:
            salt_tolerant_limits (pd.Series): Salt-tolerant limits for different land classes.
    
        Returns:
            None
        """
        # Create a DataArray filled with NaN values having the same dimensions as the reference map
        self.drempel_map = xr.full_like(self.basemaps.voorbeeld_da, np.nan)
        
        # Iterate over salt-tolerant limits for different land classes
        for lgn_num, drempelwaarde in salt_tolerant_limits.iteritems():
            # Use imod.util.where to conditionally set threshold values based on land class
            self.drempel_map = imod.util.where(self.basemaps.lgn == lgn_num, drempelwaarde, self.drempel_map)

    
class zoetwaterlens:
    def __init__(self, fn_mfnc, fn_dldrasvat, fn_areasvat, fn_solbnd, minimum_rsgem_years, chunks):
        """
        Initialize the Zoetwaterlens class by preprocessing RSGEM data and expanding time for warm-up.

        Parameters:
        - fn_mfnc (str): Filepath to the MODFLOW NetCDF dataset.
        - fn_dldrasvat (str): Filepath to the DLD RAS VAT (Variable Anisotropy Tensor) dataset.
        - fn_areasvat (str): Filepath to the RSGEM Area VAT dataset.
        - fn_solbnd (str): Filepath to the MODFLOW solution boundary file.
        - minimum_rsgem_years (int): Minimum number of years for the RSGEM warm-up period.
        - chunks (dict): way to chunk data over dimensions

        Returns:
        None (Initialization of the class with preprocessed data).

        """
        self.chunks = chunks
        
        # Preprocess RSGEM data using provided filepaths
        self.preprocess_rsgem_data(fn_mfnc, fn_dldrasvat, fn_areasvat, fn_solbnd)

        # Expand time for warm-up based on the minimum number of RSGEM years
        self.expand_time_for_warmup(minimum_rsgem_years)
        
        

    #@njit
    def _calc_fz(y, L):
        """
        Calculate the function fz using the provided parameters.
    
        Parameters:
        - y (float or numpy.ndarray): Depth or array of depths.
        - L (float): A characteristic length scale.
    
        Returns:
        - float or numpy.ndarray: The calculated value of the function fz.
    
        """
        # The function fz is calculated using the provided formula
        result = (2. / np.pi * np.arctan(np.exp(2. * np.pi * y / L) / (1 - np.exp(4 * np.pi * y / L))**0.5))
    
        return result
    
    def merged_grids_vergroten(da, voorbeeldextent):
        """
        Enlarge the extent of a data array after loading .p00x grids with imod-python.

        Parameters:
            da (xr.DataArray): DataArray to be corrected.
            voorbeeldextent (xr.DataArray): Example extent for correction.

        Returns:
            da_extent (xr.DataArray): DataArray with corrected extent.
        """
        # Replace grid values with NaN
        voorbeeldextent_nan = xr.full_like(voorbeeldextent, np.nan)
        
        # Enlarge the extent of the data array
        da_extent_nan = voorbeeldextent_nan.combine_first(da)
        
        # Replace the NaN values with 0
        da_extent = da_extent_nan.fillna(0)
        
        return da_extent

    
    @guvectorize("(float32, float32[:], float32[:], float32[:,:], float32[:], float32, float32, float32, float32, float32[:])", "(), (n), (n), (m,n), (m), (), (), (), () -> (n)")
    def _calculate_rsgem_guv(zeta_start, head, bdgflf, fluxes, L, porosity, conc_rch, conc_gw, min_zeta, out_zeta):#, out_zeta, out_fluxwconc, out_fluxconc):
        #@numba.guvectorize("(float32, float32[:], float32[:], float32[:,:], float32[:], float32, float32, float32, float32, float32[:], float32[:], float32[:,:])", "(), (n), (n), (m,n), (m), (), (), (), () -> (n), (n), (m,n)")
        #pass
        """
        Calculate RSGEM groundwater and vadose zone parameters over time.
    
        Parameters:
        - zeta_start (float): Initial value of the water table depth.
        - head (numpy.ndarray): Array of groundwater head values over time.
        - bdgflf (numpy.ndarray): Array of base discharge and flux values.
        - fluxes (numpy.ndarray): Array of various flux values over time.
        - L (float): Characteristic length scale.
        - porosity (float): Porosity value.
        - conc_rch (float): Concentration of recharge water.
        - conc_gw (float): Concentration of groundwater.
        - min_zeta (float): Minimum water table depth threshold.
        - out_zeta (numpy.ndarray): Output array to store water table depth over time.
        - out_fluxwconc (numpy.ndarray): Output array to store flux with concentration over time.
        - out_fluxconc (numpy.ndarray): Output array to store concentration of flux over time.
    
        Returns:
        None (Modifies the provided output arrays in-place).
    
        """
        # Determine the number of time steps
        nt = int(head.shape[0])
    
        # Check if conc_gw is not NaN or less than 0.1, if true, return
        if np.isnan(conc_gw) or conc_gw < 0.1:
            return
    
        # Loop through each time step
        for it in range(nt):
            # Set the previous water table depth (zeta) value
            if it:
                zeta_tmin1 = out_zeta[it - 1]
            else:
                zeta_tmin1 = zeta_start
    
            # Calculate the vertical distance between current and previous zeta
            y = min(zeta_tmin1, head[it]) - head[it]
    
            # Calculate the function fz, ignore functions because guvectorize issues
            fz = (2. / np.pi * np.arctan(np.exp(2. * np.pi * y / L) / (1 - np.exp(4 * np.pi * y / L))**0.5))

            #fz = 1
            
            # Calculate the below-ground flux
            below = bdgflf[it] + (fz * fluxes[:, it]).sum()
    
            # Update the water table depth considering porosity
            out_zeta[it] = min(head[it], zeta_tmin1 + below / 1000. / porosity)
    
            # Calculate the flux concentration using fz, conc_gw, and conc_rch
           # out_fluxconc[:, it] = fz * conc_gw + (1 - fz) * conc_rch
    
            # Calculate the flux with concentration
            #out_fluxwconc[it] = (out_fluxconc[:, it] * fluxes[:, it]).sum() / fluxes[:, it].sum()
    
            # Check if the water table depth is below the minimum threshold, if true, return
            if out_zeta[it] < min_zeta:
                return 
        return

    
    def preprocess_rsgem_data(self, fn_mfnc, fn_dldrasvat, fn_areasvat, fn_solbnd):
        """
        Preprocess RSGEM data from provided files.
    
        Parameters:
            fn_mfnc (str): Filepath to the netCDF file containing RSGEM data.
            fn_dldrasvat (str): Filepath to the DLDRA_SVAT file.
            fn_areasvat (str): Filepath to the AREA_SVAT file.
            fn_solbnd (str): Filepath to the SOL_BND file.
    
        Returns:
            None
        """
        # Define chunks for the time dimension    
        # Open the netCDF file containing RSGEM data and apply chunking
        self.input_rsgem = xr.open_dataset(fn_mfnc).chunk(self.chunks)
    
        # Reading only 14/28 data, so multiply fluxes by elapsed time
        dt = self.input_rsgem.time.to_series().diff().dt.days.fillna(14).to_xarray()  # first NaN is 14
        for prm in self.input_rsgem.data_vars:
            if "bdg" in prm:
                print(prm)
                self.input_rsgem[prm] *= dt
    
        # Load input from AREA_SVAT, DLDRA_SVAT, and SOL_BND files
        area_svat = pd.read_csv(fn_areasvat, delim_whitespace=True, header=None, names=["svat", "area", "elev", "slk", "lu", "dprzk", "nm", "cfpm", "cfE", "x", "y", "_", "irow", "icol"])
        dldra_svat = pd.read_csv(fn_dldrasvat, delim_whitespace=True, header=None, names=["svat"] + [f"D_s{i}" for i in range(1, 6)] + [f"L_s{i}" for i in range(1, 6)])
        self.fix_fn_solbnd_merge(fn_solbnd)
        sol_bnd = pd.read_csv(fn_solbnd, delimiter=",", skiprows=2, header=None, names=["svat", "conc_rain", "conc_gw"])
    
        # Merge dataframes and convert to xarray dataset
        joined = area_svat.merge(dldra_svat, left_on="svat", right_on="svat").merge(sol_bnd, left_on="svat", right_on="svat")
        self.svat = joined.groupby(["y", "x"]).first().to_xarray().reindex_like(self.input_rsgem)
    
        # Extract and assign L values to coordinate "sys"
        L = self.svat[[f"L_s{i}" for i in range(1, 6)]].to_array(dim="sys", name="L").astype(float)
        L = L.assign_coords(sys=["bdgriv_sys1", "bdgriv_sys2", "bdgriv_sys3", "bdgdrn_sys1", "bdgdrn_sys3"])
        
        # Clip L values to a maximum of 100.0
        self.L2 = np.fmin(L, 100.0).chunk(self.chunks)  # According to the manual simgro
        
        self.input_dates = self.input_rsgem.time

    def expand_time_for_warmup(self, minimum_rsgem_years):
        """
        Expand the time dimension of the input_rsgem dataset to meet a minimum number of years for warm-up.
    
        Parameters:
            minimum_rsgem_years (int): The minimum number of years required for the warm-up period and calculation period combined.
    
        Returns:
            None
        """
        
        first_day = self.input_rsgem.time[0].dt.strftime('%d-%m').item()
        last_day = self.input_rsgem.time[-1].dt.strftime('%d-%m').item()
        if first_day != '14-01' or last_day != '28-12':
            raise UserWarning('Input data does not contain complete years, which is needed when expanding the input data')
        
        # Calculate the current number of years in the dataset
        self.dt_year = (self.input_rsgem.time[-1].dt.year - self.input_rsgem.time[0].dt.year).item() + 1
        
        # Calculate the multiplication factor needed to reach the minimum number of years
        mult_factor = int(np.ceil(minimum_rsgem_years /  self.dt_year))
        
        # Create a list to store the expanded datasets
        xrs = [self.input_rsgem.copy()]
        self.translated_input_dates = xrs[0].time

        
        # Iterate to create additional datasets with adjusted time values
        for i in range(1, mult_factor):
            input_rsgem = self.input_rsgem.copy()
            
            ts = [datetime(year=t.dt.year.item()+ self.dt_year*i, month = t.dt.month.item(), day=t.dt.day.item()) for t in input_rsgem.time]
            input_rsgem['time'] = ts
            xrs.append(input_rsgem)
            
            self.translated_input_dates = input_rsgem.time
        
        # Concatenate the datasets along the time dimension
        self.input_rsgem = xr.concat(xrs, dim='time')
        self.input_rsgem.chunk(self.chunks)
        
        
    def calculate_rsgem_lhm(self, input_mf, L, area, porosity, conc_rch, conc_gw):
        """
        Calculate various hydrological parameters using the RSGEM-LHM (Regional Salt Groundwater Model - Land and Hydrological Model) approach.
    
        Parameters:
        - input_mf (xarray.Dataset): Input dataset containing Modflow data.
        - L (xarray.DataArray): Represents the horizontal hydraulic conductance of the model grid.
        - area (float): Area of the model grid.
        - porosity (float): Porosity value.
        - conc_rch (xarray.DataArray): Concentration of recharge.
        - conc_gw (xarray.DataArray): Concentration of groundwater.
    
        Returns:
        - ds (xarray.Dataset): Dataset containing calculated hydrological parameters, including 'zeta' (water table), 'fluxwconc' (flux with concentration), 'fluxconc' (concentration of flux), 'head' (head values), 'rwl_thick' (root zone thickness).
    
        """
    
        # Create an empty array for 'zeta' filled with NaN
        zeta = xr.full_like(input_mf["head"], np.nan).astype(np.float32)
        zeta.name = "zeta"
    
        # Extract relevant fluxes from the input dataset and convert to mm
        fluxes = input_mf[["bdgriv_sys1", "bdgriv_sys2", "bdgriv_sys3", "bdgdrn_sys1", "bdgdrn_sys3"]].to_array(dim="sys", name="flux")
        fluxes = fluxes / area * 1000.  # m3 to mm
        fluxes = fluxes.chunk(sys=1)
    
        # Check if the shape of L matches the shape of fluxes
        assert(L.shape[0] == fluxes.shape[0])
    
        # Create arrays for 'fluxwconc' and 'fluxconc' filled with NaN
        fluxwconc = xr.full_like(input_mf["head"], np.nan).astype(np.float32)
        fluxwconc.name = "fluxwconc"
        fluxconc = xr.full_like(fluxes, np.nan).astype(np.float32)
        fluxconc.name = "conc"
    
        # Convert bdgflf to mm
        bdgflf = input_mf["bdgflf"] / area * 1000.  # m3 to mm
    
        # Create an array for porosity with the same shape as 'head'
        porosity = xr.full_like(input_mf["head"].isel(time=0), porosity)
    
        # Initialize zeta_start and min_zeta based on the first time step
        zeta_start = input_mf["head"].isel(time=0) - 2.5  # initialize zeta @ 2m below first head
        min_zeta = input_mf["head"].isel(time=0) - 30.
    
    
        # Apply the RSGEM-LHM calculation using a ufunc
        zeta = xr.apply_ufunc(
            zoetwaterlens._calculate_rsgem_guv,  # Function to apply
            zeta_start.astype(np.float32),
            input_mf["head"].astype(np.float32),
            bdgflf.astype(np.float32),
            fluxes.astype(np.float32),
            L.astype(np.float32),
            porosity.astype(np.float32),
            conc_rch.astype(np.float32),
            conc_gw.astype(np.float32),
            min_zeta.astype(np.float32),
            zeta,
            input_core_dims=[[], ["time"], ["time"], ["sys", "time"], ["sys"], [], [], [], [], ["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"allow_rechunk": True}
        )
        # # Apply the RSGEM-LHM calculation using a ufunc
        # zeta, fluxwconc, fluxconc = xr.apply_ufunc(
        #     zoetwaterlens._calculate_rsgem_guv,  # Function to apply
        #     zeta_start.astype(np.float32),
        #     input_mf["head"].astype(np.float32),
        #     bdgflf.astype(np.float32),
        #     fluxes.astype(np.float32),
        #     L.astype(np.float32),
        #     porosity.astype(np.float32),
        #     conc_rch.astype(np.float32),
        #     conc_gw.astype(np.float32),
        #     min_zeta.astype(np.float32),
        #     zeta,
        #     fluxwconc,
        #     fluxconc,
        #     input_core_dims=[[], ["time"], ["time"], ["sys", "time"], ["sys"], [], [], [], [], ["time"], ["time"], ["sys", "time"]],
        #     output_core_dims=[["time"], ["time"], ["sys", "time"]],
        #     dask="parallelized",
        #     output_dtypes=[np.float32, np.float32, np.float32],
        #     dask_gufunc_kwargs={"allow_rechunk": True}
        # )
    
        # Create a dataset with the calculated arrays
        ds = xr.Dataset({"zeta": zeta})#, "fluxwconc": fluxwconc, "fluxconc": fluxconc})
    
        # Transpose the dataset to match the desired order of dimensions
        #ds = ds.transpose("sys", "time", "y", "x")
    
        # Add 'head' and 'rwl_thick' to the dataset
        ds["head"] = input_mf["head"]#.chunk(self.chunks)
        ds["rwl_thick"] = (input_mf["head"] - ds["zeta"])#.chunk(ds["zeta"].chunks)
    
    
        #reset expaneded dataset back to original dates
        ds = ds.sel(time=self.translated_input_dates)
        ds['time'] = self.input_dates
        
        # Return the calculated dataset
        return ds

    
    def fix_fn_solbnd_merge(self, fn_solbnd):
        """
        Fix formatting issues in a MODFLOW solution boundary file.
    
        Parameters:
        - fn_solbnd (str): Filepath to the MODFLOW solution boundary file.
    
        Returns:
        None (File is modified in-place).
    
        """
    
        # Read the content of the solution boundary file
        with open(fn_solbnd, 'r') as f:
            lines = f.readlines()
    
        # Initialize a list to store the modified lines
        newlines = []
    
        # Iterate through each line in the file
        for l in lines:
            # Split the line using comma as a delimiter
            d = l.split(',')
    
            # Check if the split result has more than 3 elements
            if len(d) > 3:
                # Keep only the first three elements and truncate the third element to 12 characters
                d = d[:3]
                d[2] = d[2][:12] + '\n'
    
            # Join the modified elements back into a comma-separated string
            d = ','.join(d)
    
            # Append the modified line to the list
            newlines.append(d)
    
        # Write the modified lines back to the solution boundary file
        with open(fn_solbnd, 'w') as f:
            f.writelines(newlines)

    
    def create_masks(self, ds, z_afstand, LGN_numbers, beregening_idf, years, basemap_data):
        """
        Create masks based on specified criteria for risk area, agricultural area, and irrigation.
    
        Parameters:
        - ds (xarray.Dataset): Modflow dataset with a 'zeta' variable.
        - z_afstand (float): Distance for grid expansion in the calc_zeta_max method.
        - drempel_df (pandas.DataFrame): DataFrame containing LGN_number for agricultural area masking.
        - beregening_idf (xarray.DataArray): Irrigation dataset.
        - years (list): List of years for calculating zeta_max.
        - basemap_data (your_datatype): Data containing information for agricultural area masking.
    
        Returns:
        - mask_risico_gebied (xarray.DataArray): Mask for the risk area.
        - mask_agrarisch_gebied (xarray.DataArray): Mask for the agricultural area.
        - mask_beregening (xarray.DataArray): Mask for irrigation.
    
        """
    
        # Calculate zeta_max using the calc_zeta_max method
        self.calc_zeta_max(ds.zeta, years, z_afstand)
    
        # Calculate the depth to cap based on zeta_max and grid expansion distance
        diepte_zetamax = basemap_data.ahn / 100. - self.zeta_max  # ahn in cm
        diepte_cap = diepte_zetamax - (z_afstand / 100)
    
        # Create a mask for the risk area where the depth to cap is less than 1 m
        mask_risico_gebied = xr.where(diepte_cap < 1, 1, np.nan)
    
        # Create a mask for the agricultural area based on LGN_number
        mask_agrarisch_gebied = xr.where(basemap_data.lgn.isin(LGN_numbers), 1, np.nan)
    
        # Create a mask for irrigation where beregening_idf is less than 2
        mask_beregening = xr.where(beregening_idf < 2, 1, np.nan)
    
        # Return the created masks
        return mask_risico_gebied, mask_agrarisch_gebied, mask_beregening

    
    def calc_zeta_max(self, zeta, jaren, z_afstand):
        """
        Calculate zeta_max (maximum head value) over a specified period and expand the grid with a specified distance.
    
        Parameters:
        - zeta (xarray.Dataset): Modflow heads dataset with a 'time' dimension.
        - jaren (list): List of years for the specified period.
        - z_afstand (float): Distance to expand the grid.
    
        Returns:
        None (Result stored in self.zeta_max).
    
        """
    
        # Get the start and end years from the provided list
        startjaar = jaren[0]
        eindjaar = jaren[-1]
    
        # If the start and end years are the same, increment the end year by 1
        if startjaar == eindjaar:
            eindjaar += 1
    
        # Calculate zeta_min and zeta_max using the heads_to_GLG_GHG method
        zeta_max = self.heads_to_GLG_GHG(zeta, startjaar, eindjaar)
    
        # Subtract 1 from zeta_max and expand the grid using merged_grids_vergroten method
        # self.zeta_max = merged_grids_vergroten(zeta_max - 1, z_afstand)
        self.zeta_max = merged_grids_vergroten(zeta_max, z_afstand)

        
    def heads_to_GLG_GHG(self, head, jaar_start, jaar_eind):
        """
        Convert Modflow heads to GHG (Gemiddeld Hoge Grondwaterstand) and GLG (Gemiddeld Lage Grondwaterstand)
        for a specified hydrological year period, in meters above sea level (m+NAP).
    
        Parameters:
        - head (xarray.Dataset): Modflow heads dataset with a 'time' dimension.
        - jaar_start (int): Start year of the hydrological year period.
        - jaar_eind (int): End year of the hydrological year period.
    
        Returns:
        - list: A list containing two xarray datasets - [GLG (mean of lowest 3 ranks), GHG (mean of highest 3 ranks)].
    
        """
    
        # Select heads data for the specified hydrological year period
        selection_hyd_jaar = head.sel(time=(slice(str(jaar_start) + '-04-01', str(jaar_eind) + '-03-31')))
    
        # Select every 14th and 28th day within the hydrological year
        selection = selection_hyd_jaar.sel(time=((selection_hyd_jaar["time.day"] == 14) | (selection_hyd_jaar["time.day"] == 28)))
        
        #GHG = self.self_ghg(selection)
        GHGs = []
        #GLGs = []
        for y in tqdm(selection.y):
            #print(y)
            y_selection = selection.sel(y=y)#.compute()
            #Compute the rank of each time step based on the head values
            rank = y_selection.rank("time")
        
            # Find the maximum rank across all dimensions
            max_rank = rank.max().values
        
            # Calculate GHG by averaging values where rank is in the top 3
            GHG = y_selection.where(rank > (max_rank - 3)).mean("time")
            
            GHGs.append(GHG)
            
            # # Calculate GLG by averaging values where rank is in the bottom 3
            # GLG = y_selection.where(rank <= 3).mean("time")
            # GLGs.append(GLG)
        
        GHG = xr.concat(GHGs, dim = 'y')
        # GLG = xr.concat(GLGs, dim = 'y')
        
        # Return a list containing GLG and GHG
        return GHG

    def self_ghg(self, da_chunk):
        test_da = da_chunk.copy()
        da_min = da_chunk.min()
        da_fill = da_min - 10
        max_vals = []
        for i in range(3):
            da_max_idx = test_da.idxmax('time')
            da_max_values = test_da.max('time')
            da_max_values['time'] = i
            test_da = imod.util.where(da_chunk.time==da_max_idx, da_fill, test_da)
            max_vals.append(da_max_values)
            #print(test_da.max('time'))
            
        max_vals_da = xr.concat(max_vals, dim = 'time')
        da_self_ghg = max_vals_da.mean('time')
        return da_self_ghg

    def calculate_n_days_above_threshold(self, crit, jaren, ds, mask_risico_gebied, mask_agrarisch_gebied, mask_beregening, zeta_max):
        """
        Calculate the number of days above a specified threshold for each year.
    
        Parameters:
        - crit (float): The threshold value.
        - jaren (list): List of years for which the calculation is performed.
        - ds (xarray.Dataset): Input dataset containing the variable 'rwl_thick' and a time dimension.
        - mask_risico_gebied (xarray.DataArray): Mask for the risk area.
        - mask_agrarisch_gebied (xarray.DataArray): Mask for the agricultural area.
        - mask_beregening (xarray.DataArray): Mask for irrigation.
        - zeta_max (xarray.DataArray): Maximum threshold.
    
        Returns:
        None (Results stored in self.n_days_over_threshold).
    
        """
    
        # Initialize an empty list to store results for each year
        comb_days = []
    
        # Loop through each year in the list of years
        for y in jaren:
            print(y)
            # Select data for the current year and apply masks
            select = ds.rwl_thick.sel(time=slice(f'{y}-01-01', f'{y}-12-31')) * mask_risico_gebied * mask_agrarisch_gebied * mask_beregening
    
            # Create a new array with zeros to store the number of days over the threshold
            number_days = xr.full_like(zeta_max, 0)
            #number_days.load()
    
            # Loop through each time step in the selected data
            for t in range(0, len(select.time)):
                # Check if the date is in the growing season (April to September)
                if (int(select.time[t].dt.month) >= 4) & (int(select.time[t].dt.month) < 10):
                    # Calculate the number of days between consecutive time steps
                    days = int((select.time[t + 1] - select.time[t]).dt.days)
    
                    # Enlarge the selected grid using a method merged_grids_vergroten
                    voorbeeldextent_nan = xr.full_like(number_days, np.nan)
                    select_day = voorbeeldextent_nan.combine_first(select.isel(time=t))
                    #select_day = merged_grids_vergroten(select.isel(time=t), zeta_max)
    
                    # Update the number of days over the threshold
                    number_days = xr.where(select_day < crit, number_days + days, number_days)
    
            # Assign the current year to the result and append it to the list
            number_days['year'] = y
            comb_days.append(number_days)
    
        # Concatenate the results for all years along the 'year' dimension
        self.n_days_over_threshold = xr.concat(comb_days, 'year')

            
#%%
# if __name__ == '__main__':
#     years=range(2010,2011)
    
#     path_model_output = Path(r"G:\Deltascenarios\data\4_output\test_delta_light")
#     scenarios = ["lhm_4.3.0b_ref_herstart"]
#     scenarios_output = scenarios
#     path_output = Path(r"g:\Projecten\2022\DPZW_verziltingsrelatie\Beregening_gewasschade")
    
#     scenarios_lgn = [Path(r"E:\LHM_master\Data\2_Model_Input\metaswap\grid\LGN250.IDF")]
    
#     lsw_shp_path = Path(r"e:\LHM_master\Data\2_Model_Input\coupling\lsws.shp")
    
#     drempel_dict = {}
#     drempel_dict['gras'] = [2055, 2711]
#     drempel_dict['mais'] = [498, 371]
#     drempel_dict['aardappelen'] = [838, 668]
#     drempel_dict['bieten'] = [1478, 817]
#     drempel_dict['granen'] = [2626, 2796]
#     drempel_dict['overig'] = [963, 817]
#     drempel_dict['boomteelt'] = [233, 62]
#     drempel_dict['boomgaard'] = [465, 381]
#     drempel_dict['bollen'] = [475, 468]
#     drempel_dict['fruitkwekerij'] = [300, 247] #std van fruitkwekerij is geschat door 300*(drempel_df['std']/drempel_df['mean']).mean()
    
#     drempel_df = pd.DataFrame(drempel_dict, index = ['mean', 'std']).T
#     drempel_df['LGN_number'] = [1,2,3,4,5,6,7,9,10,21]
    
#     #waarom is dat anders dan lgn numbers in drempel_df
#     landbouwgewassen = [1,2,3,4,5,6,7,9,10,21,23,24,25] #8=glastuinbouw:doet niet mee
    
#     z_afstand = imod.rasterio.open(r"g:\Projecten\2022\DPZW_verziltingsrelatie\regenwaterlenzen\zcrit.asc").compute()
#     beregening_idf = imod.idf.open(r'E:\LHM_master\Data\2_Model_Input\metaswap\grid\BEREGEN.IDF').compute()
    
#     porosity = 0.4
    
#     for i in range(len(scenarios)):
#         s = scenarios[i]
#         n = scenarios_output[i]
#         output_dir = path_output / rf"{scenarios_output[i]}/Beregening"
#         output_dir.mkdir(exist_ok=True,parents=True)   
        
    
#         basemap_files = {#'CONC': output_dir / r"../concentratie_LSW/Concentratie_LSW_*_mgL.idf",
#                               'CONC': Path(r'G:\Projecten\2022\DPZW_verziltingsrelatie\Conc_lsw') / s /'Concentratie_LSW*.IDF',
#                               'beregening_data': path_model_output / rf"{s}/metaswap/bdgPssw/bdgPssw*_L1.IDF",
#                               'transpiratie_pot': path_model_output / rf"{s}/metaswap/msw_Tpot/msw_Tpot*L1.IDF",
#                               'area': path_model_output / rf"{s}/metaswap/bdgPssw/area*L1.IDF",
#                               'lgn': Path(r"E:\LHM_master\Data\2_Model_Input\metaswap\grid\LGN250.IDF"),
#                               'voorbeeld_da': Path(r"E:\LHM_master\Data\2_Model_Input\metaswap\grid\LGN250.IDF"),
#                               'shp': Path(r"e:\LHM_master\Data\2_Model_Input\coupling\lsws.shp")}
            
#         basemap_data = basemaps(basemap_files)
    
#         test = Beregening(basemap_data, years)
        
#         salt_tolerant_limits = pd.Series(drempel_df['mean'].values, drempel_df.LGN_number)
#         result_maps, index_maps = test.calculate_indices_with_threshold(salt_tolerant_limits)
        
        
#         path_msinput = path_model_output / s / 'metaswap'
#         fn_dldrasvat = path_msinput / "dldra_svat.inp"
#         fn_areasvat = path_msinput / "area_svat.inp"
#         fn_solbnd = path_msinput / "solute_bnd.csv.inp"     
#         fn_mfnc = path_model_output / s / "imodflow/results" / "head_and_fluxes.nc"
        
#         if not fn_mfnc.is_file():
#             write_modflow_output_nc(path_model_output, s, years, fn_mfnc)
        
#         zw_lens = zoetwaterlens(fn_mfnc, fn_dldrasvat, fn_areasvat, fn_solbnd)        

#         zw_lens.input_rsgem = zw_lens.input_rsgem.isel(time=range(2*12))
#         svatsel = zw_lens.svat.where(zw_lens.svat["conc_gw"] > 0.5)
    
#         print(f"Running RSGEM for {len(zw_lens.input_rsgem.time)} timesteps")#, from {input_rsgem.time[0].dt:%d-%m-%Y} to {input_rsgem.time[-1].dt:%d-%m-%Y}")
#         ds = zw_lens.calculate_rsgem_lhm(zw_lens.input_rsgem, zw_lens.L2, svatsel["area"], porosity, svatsel["conc_rain"], svatsel["conc_gw"])
#         ds = ds.transpose("sys","time","y","x")
#         ds["head"] = zw_lens.input_rsgem["head"].chunk(ds["zeta"].chunks)
#         ds["rwl_thick"] = (zw_lens.input_rsgem["head"] - ds["zeta"]).chunk(ds["zeta"].chunks)
    
        
#         # ## zeta_max
#         zw_lens.calc_zeta_max(ds.zeta, years, z_afstand)    
#         diepte_cap =  zw_lens.zeta_max - (z_afstand/100)
#         mask_risico_gebied = xr.where(diepte_cap<1,1,np.nan)
#         mask_agrarisch_gebied = xr.where(basemap_data.lgn.isin(landbouwgewassen),1,np.nan)
#         mask_beregening = xr.where(beregening_idf<2,1,np.nan)
        
#         criterium = 0.2
#         zw_lens.calculate_n_days_above_threshold(criterium, 
#                                                    years,
#                                                    ds, 
#                                                    mask_risico_gebied, 
#                                                    mask_agrarisch_gebied, 
#                                                    mask_beregening, 
#                                                    zw_lens.zeta_max)
         
    
#     # lens_test.n_days_over_threshold.to_netcdf('n_days_class.nc')
#     zw_lens.n_days_over_threshold = zw_lens.n_days_over_threshold.compute()
    
    
#     dagen_LSW = zw_lens.average_per_lsw(zw_lens.n_days_over_threshold, test.basemaps.da_lsws)
