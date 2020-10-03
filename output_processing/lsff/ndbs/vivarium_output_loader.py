import pandas as pd
import os

def locaction_paths_from_rundates(base_directory: str, locations_rundates: dict) -> dict:
    """
    Returns a dictionary of the form {'location': 'pathname'} from a dictionary of the form
    {'location': 'rundate'} by constructing each pathname as
    pathname = f'{base_directory}/{location.lower()}/{rundate}/'
    """
    return {location: os.path.join(base_directory, location.lower(), rundate)
            for location, rundate in locations_rundates.items()}

def load_output_by_location(locations_paths: dict, output_filename='output.hdf') -> dict:
    """
    Loads output.hdf file from the path for each location in the dictionary locations_paths,
    and returns a dictionary of output DataFrames indexed by location.
    """
    # Read in data from different countries
    locations_outputs = {location: pd.read_hdf(os.path.join(path, output_filename))
                                               for location, path in locations_paths.items()}
    return locations_outputs

def merge_location_outputs(locations_outputs: dict, copy=True) -> pd.DataFrame:
    """
    Concatenate the output DataFrames for all locations stored in locations_outputs into a single DataFrame,
    with a 'location' column added to specify which location each row came from.
    """
    if copy:
        # Use a dictionary comprhension to avoid overwriting the input dictionary.
        # Use DataFrame.assign to avoid overwriting the dataframes in the dictionary.
        # Note that in the statement `output.assign(location=location)`, the first 'location' is
        # the column name, and the second is the `location` variable, which is the
        # value to assign to the 'location' column.
        locations_outputs = {location: output.assign(location=location)
                             for location, output in locations_outputs.items()}
    else:
        # Modify dictionary and dataframes in place
        for location, output in locations_outputs.items():
            output['location'] = location

    # The concatenated dataframe is not intended to be modified, so we don't
    # need to copy the sub-dataframes.
    return pd.concat(locations_outputs.values(), copy=False, sort=False)

def load_and_merge_location_outputs(locations_paths: dict, output_filename='output.hdf') -> pd.DataFrame:
    """
    Loads output.hdf files from the paths for the locations in the dictionary locations_paths into a single,
    concatenated DataFrame, with a 'location' column added to specify which location each row came from.
    """
    locations_outputs = load_output_by_location(locations_paths, output_filename)
    return merge_location_outputs(locations_outputs, copy=False)

# # Old version, replaced with above functions on 2020-09-24:
#
# def load_output_by_location(locations_paths: dict, output_filename='output.hdf') -> pd.DataFrame:
#     """
#     Loads output.hdf files for the locations in the dictionary locations_paths into a single,
#     concatenated DataFrame, with a 'location' column added.
#     """
#     # Read in data from different countries
#     locations_outputs = {location: pd.read_hdf(os.path.join(path, output_filename))
#                                                for location, path in locations_paths.items()}
    
#     for location, output in locations_outputs.items():
#         output['location'] = location

#     return pd.concat(locations_outputs.values(), copy=False, sort=False)

def load_transformed_count_data(directory: str) -> dict:
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

def load_count_data_by_location(locations_paths: dict, subdirectory='count_data') -> dict:
    """
    Loads data from all locations into a dictionary of dictionaries of dataframes,
    indexed by location. Each dictionary in the outer dictionary is
    indexed by filename
    
    For each location, reads data files from a directory called f'{path}/{subdirectory}/',
    where `path` is the path for the location specified in the `locations_paths` dictionary.
    """
    locations_count_data = {location: load_transformed_count_data(os.path.join(path, subdirectory))
                        for location, path in locations_paths.items()}
    return locations_count_data

def merge_location_count_data(locations_count_data: dict, copy=True) -> dict:
    """
    Concatenate the count data tables from all locations into a single dictionary of dataframes
    indexed by table name, with a column added to the begininning of each table specifying the
    location for each row of data.
    """
    if copy:
        # Use a temporary variable and a for loop instead of a dictionary comprehension
        # so we can access the `data` variable later.
        locations_count_data_copy = {}
        for location, data in locations_count_data.items():
            locations_count_data_copy[location] = {
                table_name:
                # Use DataFrame.reindex() to simultaneously make a copy of the dataframe and
                # assign a new location column at the beginning.
                table.reindex(columns=['location', *table.columns], fill_value=location)
                for table_name, table in  data.items()
            }
        locations_count_data = locations_count_data_copy
#         # Alternate version using dictionary comprehension; this version would require a different
#         # method of iterating through the table names below, because `data` is inaccessible afterwards.
#         locations_count_data = {
#             location: {
#                 table_name:
#                 table.reindex(columns=['location', *table.columns], fill_value=location)
#                 for table_name, table in  data.items()
#             }
#             for location, data in locations_count_data.items()
#         }
    else:
        # Modify the dictionaries and dataframes in place
        for location, data in locations_count_data.items():
            for table in data.values():
                # Insert a 'location' column at the beginning of each table
                table.insert(0, 'location', location)

    # `data` now refers to the dictionary of count_data tables for the last location
    # encountred in the above for loop. We will use the keys stored in this dictionary
    # to iterate through all the table names and concatenate the tables across all locations.
    all_data = {table_name: pd.concat([locations_count_data[location][table_name]
                                      for location in locations_count_data], copy=False, sort=False)
                for table_name in data}
    return all_data

def load_and_merge_location_count_data(locations_paths: dict, subdirectory='count_data') -> dict:
    locations_count_data = load_count_data_by_location(locations_paths, subdirectory)
    return merge_location_count_data(locations_count_data, copy=False)

# # Old version, replaced with above functions on 2020-09-24:
#
# def load_count_data_by_location(locations_paths: dict, subdirectory='count_data') -> dict:
#     """
#     Loads data from all locations into a dictionary of dataframes,
#     indexed by filename, with a column added to each table specifying
#     the location.
    
#     For each location, reads data files from a directory called f'{path}/{subdirectory}/',
#     where `path` is the path for the location specified in the `locations_paths` dictionary.
#     """
#     location_dictionaries = []
#     for location, path in locations_paths.items():
#         location_dfs = load_transformed_count_data(os.path.join(path, subdirectory))
#         for table in location_dfs.values():
#             # Insert the 'location' column at the beginning
#             table.insert(0, 'location', location)
        
#         location_dictionaries.append(location_dfs)
        
#     data_dict = location_dictionaries[0]
   
#     for location_dfs in location_dictionaries[1:]:
#         for table_name, table in location_dfs.items():
#             data_dict[table_name] = data_dict[table_name].append(table)
            
#     return data_dict

##### 2020-09-24: The following functions are old and should be superceded by the above versions,
##### which do a better job of separating concerns. Leaving them in for now because some old
##### notebooks may use them.

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

def print_location_output_shapes(all_output, locations=None):
    """Print the shapes of outputs for each location to check whether all the same size or if some data is missing"""
    if locations is None:
        locations = all_output.location.unique()
    for location in locations:
        print(location, all_output.loc[all_output.location==location].shape)

def path_for_transformed_count_data(directory, location, rundate, subdirectory='count_data'):
    """
    Gets the directory path for a given location and rundate.
    
    The returned path is 
    f'{directory}/{location.lower()}/{rundate}/{subdirectory}'
    """
    return os.path.join(directory, location.lower(), rundate, subdirectory)

def load_all_transformed_count_data(directory, locations_rundates):
    """
    Loads data from all locations into a dictionary of dataframes,
    indexed by (location, filename).
    
    Assumes data files are in a directory called
    f'{directory}/{location.lower()}/{rundate}/{subdirectory}/'
    """
    dfs = {}
    for location, rundate in locations_rundates.items():
        path = path_for_transformed_count_data(directory, location, rundate)
        location_dfs = load_transformed_count_data(path)
        
        # Keys in location_dfs are filenames that describe what measures are stored in the dataframe
        for table_name in location_dfs:
            dfs[(location.lower(), table_name)] = location_dfs[table_name]
            
    return dfs
    
def load_transformed_count_data_and_merge_locations(directory, locations_rundates):
    """
    Loads data from all locations into a dictionary of dataframes,
    indexed by filename, with a column added to each table specifying
    the location.
    
    Assumes data files are in a directory called
    f'{directory}/{location.lower()}/{rundate}/{subdirectory}/'
    """
    location_dictionaries = []
    for location, rundate in locations_rundates.items():
        path = path_for_transformed_count_data(directory, location, rundate)
        location_dfs = load_transformed_count_data(path)
        for table in location_dfs.values():
            # Insert the 'location' column at the beginning
            table.insert(0, 'location', location)
        
        location_dictionaries.append(location_dfs)
        
    data_dict = location_dictionaries[0]
   
    for location_dfs in location_dictionaries[1:]:
        for table_name, table in location_dfs.items():
            data_dict[table_name] = data_dict[table_name].append(table)
            
    return data_dict

