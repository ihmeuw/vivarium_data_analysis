import pandas as pd
import os

def load_by_location_and_rundate(base_directory: str, locations_run_dates: dict) -> pd.DataFrame:
    """Load output.hdf files from folders namedd with the convention 'base_directory/location/rundate/output.hdf'"""

    # Use dictionary to map countries to the correct path for the Vivarium output to process
    # E.g. /share/costeffectiveness/results/sqlns/bangladesh/2019_06_21_00_09_53
    locactions_paths = {location: f'{base_directory}/{location.lower()}/{run_date}/output.hdf'
                       for location, run_date in locations_run_dates.items()}

    # Read in data from different countries
    locations_outputs = {location: pd.read_hdf(path) for location, path in locactions_paths.items()}

    for location, output in locations_outputs.items():
        output['location'] = location

    return pd.concat(locations_outputs.values(), copy=False, sort=False)

def print_location_output_shapes(locations, all_output):
    """Print the shapes of outputs for each location to check whether all the same size or if some data is missing"""
    for location in locations:
        print(location, all_output.loc[all_output.location==location].shape)

def load_transformed_count_data(directory):
    """
    Loads each transformed "count space" .hdf output file into a dataframe,
    and returns a dictionary whose keys are the file names and values are
    are the corresponding dataframes.
    """
    dfs = {}
    for entry in os.scandir(directory):
        filename_root, extension = os.path.splitext(entry.name)
        if extension == '.hdf':
#             print(filename_root, type(filename_root), extension, entry.path)
            dfs[filename_root] = pd.read_hdf(entry.path)
    return dfs

def path_for_transformed_count_data(directory, location, rundate, subdirectory='count_data'):
    """
    Gets the directory path for a given location and rundate.
    
    The returned path is 
    f'{directory}/{location.lower()}/{rundate}/{subdirectory}'
    """
    return os.path.join(directory, location.lower(), rundate, subdirectory)

def load_all_data(directory, locations_rundates):
    """
    Loads data from all locations into a dictionary of dataframes,
    indexed by (location, filename).
    
    Assumes data files are in a directory called
    f'{directory}/{location.lower()}/{rundate}/{subdirectory}/'
    """
    dfs = {}
    for location in locations_rundates:
        path = path_for_transformed_count_data(directory, location, locations_rundates[location])
        location_dfs = load_transformed_count_data(path)
        
        for key in location_dfs:
            dfs[(location.lower(), key)] = location_dfs[key]
            
    return dfs