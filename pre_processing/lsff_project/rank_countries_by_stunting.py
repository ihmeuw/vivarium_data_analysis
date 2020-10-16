"""Module to rank countries by decreasing population of stunted children in GBD 2019."""
import pandas as pd
import sys, os.path
sys.path.append(os.path.abspath("../.."))
from pre_processing.id_helper import *
# get_ids is no longer included in the above import * statement. Woo hoo!
from db_queries import get_ids, get_population
from get_draws.api import get_draws

def get_locations_for_stunting_prevalence(category='admin0'):
    """Returns location id table filtered to the specified location category.
    Currently the only supported category is 'admin0' (the default), which finds locations 
    with location_type=="admin0" (returing 299 id's).
    """
    locations = get_ids('location')
    if category=='admin0':
        locations = locations.query('location_type=="admin0"')
    else:
        raise ValueError(
            f"I'm sorry, Dave, I can't do that. I don't know how to find locations in the specified category {category}."
        )
    return locations

def pull_stunting_prevalence_for_locations(location_ids):
    """Calls `get_draws()` to pull Child stunting exposure for the location id's in the locations_ids iterable."""
    stunting = get_draws(
        'rei_id',
        gbd_id=list_ids('rei', 'Child stunting'),
        source='exposure',
        measure_id=list_ids('measure', 'prevalence'),
        location_id=list(location_ids),
        year_id=2019,
        age_group_id=list_ids('age_group', 'Under 5'),
        sex_id=list_ids('sex', 'Both'),
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step4',
    )
    return stunting

def compute_mean_stunting_prevalence_by_location(stunting_df):
    """Calculates mean stunting prevalence over 1000 draws for all locations in stunting_df,
    where stunting is defined as "height-for-age Z-score < -2 standard deviations".
    """
    draw_cols = stunting_df.filter(regex=r'^draw_\d{1,3}$').columns # regex for 'draw_' followed by 1-3 digits
    stunting_cat_cols = ['modelable_entity_id','parameter'] # each of these columns encodes the stunting category
    index_cols = stunting_df.columns.difference(draw_cols).difference(stunting_cat_cols) # columns to group by
    unique_location_ids = stunting_df.location_id.unique() # record this to check shape later

    # Fiind meid's for stunting exposure < 2 SD's below mean (paramter column is cat1 or cat2)
    stunted_meids = list_ids('modelable_entity',
                             'Severe Stunting, < -3 SD (post-ensemble)', # cat1
                             'Stunting Between -3 SD and -2 SD (post-ensemble)', # cat2
                             )
    # Filter stunting dataframe to rows found above (filters out cat3 and cat4)
    stunting_df = stunting_df.query(f'modelable_entity_id in {stunted_meids}')

    # Sum prevalences of cat1 and cat2 in each draw to get total stunting prevalence for each location
    stunting_df = stunting_df.groupby(index_cols.to_list())[draw_cols].sum() # fails if pd.Index not converted to list
    assert stunting_df.shape == (len(unique_location_ids), len(draw_cols))

    # Take mean over draws to get a single prevalence estimate for each location
    stunting_df = stunting_df.mean(axis=1) # stunting_df is now a pd.Series
    stunting_df.rename('stunting_prevalence', inplace=True) # changes Series.name

    return stunting_df.reset_index() # converts back to DataFrame, with column named 'stunting_prevalence'

def add_location_names_and_populations(stunting_df):
    """Adds location names and populations for each location in `stunting_df`."""
    locations = ids_to_names('location', *stunting_df.location_id.unique()).reset_index()
    stunting_df = stunting_df.merge(locations)
    population = get_population(
        age_group_id=list(stunting_df.age_group_id.unique()),
        location_id=ids_in(locations),
        year_id=2019,
        gbd_round_id=list_ids('gbd_round', '2019'),
        decomp_step='step4',
        with_ui=False,
    )
    stunting_df = stunting_df.merge(population.drop(columns='run_id'))
    return stunting_df

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