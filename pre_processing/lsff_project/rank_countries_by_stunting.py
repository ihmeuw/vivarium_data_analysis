"""Module to rank countries by decreasing population of stunted children in GBD 2019."""
import pandas as pd
import sys, os.path
sys.path.append(os.path.abspath("../.."))
from pre_processing.id_helper import *
# from db_queries import get_ids # get_ids is included in the above import *

def clean_orig_country_data(filepath):
    """Clean the original country data from BGMF in Paulina's spreadsheet saved as a .csv.
    
    - Filters to the relevant columns [country, survey year, percent stunting]
    - Adds a column recording the country rank in order of decending stunting prevalence
    - Makes all column names valid Python variables in lower case
    
    Returns a pandas DataFrame.
    """
    orig_countries = pd.read_csv(filepath)
    orig_countries = orig_countries.iloc[:,:3]
    orig_countries.drop(index=0, inplace=True)
    orig_countries.index.rename('rank_2013', inplace=True)
    orig_countries.columns = orig_countries.columns.str.replace('%', 'percent')
    orig_countries.columns = orig_countries.columns.str.replace(' ', '_')
    orig_countries.columns = orig_countries.columns.str.lower()
    orig_countries = orig_countries.reset_index()
    return orig_countries

def add_location_ids(orig_countries, location_ids):
    """Adds location id's and GBD location names to the cleaned country data returned by `clean_orig_country_data()`."""
    # Load location id table and filter to id and name columns
    locids = get_ids('location')[['location_id', 'location_name']]
    # Get id's for country names
    orig_countries = orig_countries.merge(locids, left_on='country', right_on='location_name', how='left')
    # The names Tanzania and Cote d'Ivoire aren't in the location database,
    # so search for a matching name to get the correct id's and names.
    # (I pre-verified that these searches give exactly one result)
    orig_countries.loc[orig_countries.country=='Tanzania', ['location_id', 'location_name']]=(
        search_id_table(locids, 'Tanzania')[['location_id', 'location_name']].values
    )
    orig_countries.loc[orig_countries.country=="Cote d'Ivoire", ['location_id', 'location_name']]=(
        search_id_table(locids, 'Ivoire')[['location_id', 'location_name']].values
    )
    # Filter out duplicate country names by inner joining with correct id's
    location_ids=pd.Series(location_ids, name='location_id')
    orig_countries = orig_countries.merge(location_ids, how='inner')
    return orig_countries