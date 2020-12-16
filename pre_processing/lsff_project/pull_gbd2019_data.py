import pandas as pd
import sys, os.path
sys.path.append(os.path.abspath("../.."))
from pre_processing.id_helper import *
import rank_countries_by_stunting as rbs

from db_queries import get_ids, get_population, get_outputs, get_best_model_versions
from get_draws.api import get_draws

def get_locations(key):
    """Reads the location file for the specified description key."""
    location_files = {
        'all': 'all_countries_with_ids.csv',
        'original': 'bgmf_countries_with_ids.csv',
        'top25': 'bmgf_top_25_countries_20201203.csv',
    }
    data_dir = 'data'
    filepath = f'{data_dir}/{location_files[key]}'
    return pd.read_csv(filepath)

def find_best_model_versions(search_string, entity='modelable_entity', decomp_step='step4', **kwargs_for_contains):
    """Searches for entity id's with names matching search_string using pandas.Series.str.contains,
    and calls get_best_model_versions with appropriate arguments to determine if decomp_step is correct.
    """
    best_model_versions = get_best_model_versions(
        entity,
        ids = find_ids('modelable_entity', search_string, **kwargs_for_contains),
        gbd_round_id = list_ids('gbd_round', '2019'),
        status = 'best',
        decomp_step = decomp_step,
    )
    return best_model_versions

def pull_vad_prevalence_for_locations(location_ids):
    """Calls `get_draws()` to pull vitamin A deficiency exposure for 
    the location id's in the locations_ids iterable.
    """
    # I got an error when I tried to pass measure_id or metric_id; the default is to return all available
    vad = get_draws(
        'rei_id',
        gbd_id=list_ids('rei', 'Vitamin A deficiency'),
        source='exposure',
        location_id=list(location_ids),
        year_id=2019,
        age_group_id=list_ids('age_group', 'Under 5'),
        sex_id=list_ids('sex', 'Both'),
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step4',
    )
    return vad

def pull_vad_daly_burden_for_locations(location_ids):
    """Pulls all-cause DALYs attributable to vitamin A deficiency for the specified location id's."""
    vad_burden = get_draws(
        gbd_id_type=['rei_id', 'cause_id'], # Types must match gbd_id's
        gbd_id=[list_ids('rei', 'Vitamin A deficiency'), list_ids('cause', 'All causes')],
        source='burdenator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=list_ids('metric', 'Number'), # Only available metrics are Number and Percent
        location_id=list(location_ids),
        year_id=2019,
        age_group_id=list_ids('age_group', # Actually, burden exists for all age groups because of VAD cause
                             'Early Neonatal', 'Late Neonatal', 'Post Neonatal', '1 to 4'), # Age group aggregates not available
        sex_id=list_ids('sex', 'Male', 'Female'), # Sex aggregates not available
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    return vad_burden

def pull_dalys_attributable_to_risk_for_locations(risk_factor_name, location_ids):
    """Calls get_draws to pull all-cause DALYs attributable to the specified risk for the specified locations.
    The call does not specify the age_group_id, so it will pull DALYs for all age groups that contriibute DALYs.
    """
    burden = get_draws(
        gbd_id_type=['rei_id', 'cause_id'], # Types must match gbd_id's
        gbd_id=[list_ids('rei', risk_factor_name), list_ids('cause', 'All causes')],
        source='burdenator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=list_ids('metric', 'Number'), # Only available metrics are Number and Percent
        location_id=list(location_ids),
        year_id=2019,
        sex_id=list_ids('sex', 'Male', 'Female'), # Sex aggregates not available
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    return burden

def aggregate_draws_over_columns(df, groupby_cols):
    """Aggregates (by summing) over the specified columns in the passed dataframe, draw by draw."""
    draw_cols = df.filter(regex=r'^draw_\d{1,3}$').columns
    index_cols = df.columns.difference([*draw_cols, *groupby_cols])
    return df.groupby(index_cols.to_list())[draw_cols.to_list()].sum()

def calculate_proportion_global_vad_burden(vad_burden_for_locations, global_vad_burden):
    """Calculates percent of the global burden (in DALYs) of vitamin A deficiency for each of the locations
    in `vad_burden_for_locations`.
    """
    groupby_cols = ['age_group_id', 'sex_id']
    vad_burden_for_locations = aggregate_draws_over_columns(vad_burden_for_locations, groupby_cols)
    global_vad_burden = aggregate_draws_over_columns(global_vad_burden, groupby_cols)
    # Reset the location_id level in denominator to broadcast over global instead of trying to match location
    return vad_burden_for_locations / global_vad_burden.reset_index('location_id', drop=True)

