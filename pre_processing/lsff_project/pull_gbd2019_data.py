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

def pull_dalys_for_cause_for_locations(cause_name, location_ids):
    dalys = get_draws(
        gbd_id_type='cause_id',
        gbd_id=[list_ids('rei', cause_name), list_ids('cause', cause_name)],
        source='dalynator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=list_ids('metric', 'Number'),
        location_id=list(location_ids),
        year_id=2019,
        sex_id=list_ids('sex', 'Male', 'Female'),
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    return dalys

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

def calculate_proportion_global_burden(burden_for_locations, global_burden):
    """Calculates percent of the global burden of a risk or cause for each of the locations
    in `burden_for_locations`.
    """
    groupby_cols = ['age_group_id', 'sex_id']
    burden_for_locations = aggregate_draws_over_columns(burden_for_locations, groupby_cols)
    global_burden = aggregate_draws_over_columns(global_burden, groupby_cols)
    # Reset the location_id level in denominator to broadcast over global instead of trying to match location
    return burden_for_locations / global_burden.reset_index('location_id', drop=True)

def calculate_all_proportion_global_burdens(locations_key):
    """Claculate"""
    locations = get_locations(locations_key)[['location_name', 'location_id']]
    global_loc = pd.Series(['Global', 1], index=locations.columns)
    locations = locations.append(global_loc, ignore_index=True)
    
    risks = ['Vitamin A deficiency', 'Zinc deficiency', 'Iron deficiency']
    risk_burdens = [pull_dalys_attributable_to_risk_for_locations(risk, locations.location_id) for risk in risks]
#     vad_burden = pull_dalys_attributable_to_risk_for_locations('Vitamin A deficiency', locations.location_id)
#     zinc_burden = pull_dalys_attributable_to_risk_for_locations('Zinc deficiency', locations.location_id)
#     iron_burden = pull_dalys_attributable_to_risk_for_locations('Iron deficiency', locations.location_id)

    risk_burden_proportions = [
        calculate_proportion_global_burden(
            risk_burden.query(f"location_id != {global_loc.location_id}"),
            risk_burden.query(f"location_id == {global_loc.location_id}")
        ) for risk_burden in risk_burdens
    ]
    
    return pd.concat(risk_burden_proportions) # return pd.concat of this?

def summarize_risk_proportions(risk_burden_proportions):
    # Sum proportion over all locations for each risk
    risk_burden_proportions = risk_burden_proportions.groupby(['rei_id']).sum()
    risk_burden_proportions['mean'] = risk_burden_proportions.mean(axis=1)
    risk_burden_proportions['lower'] = risk_burden_proportions.quantile(0.025, axis=1)
    risk_burden_proportions['upper'] = risk_burden_proportions.quantile(0.975, axis=1)
    risk_burden_proportions['mean_lb_ub'] = risk_burden_proportions.apply(
        lambda row: f"{row['mean']:.2%} ({row['lower']:.2%}, {row['upper']:.2%})", axis=1)
    risk_burden_proportions.index = ids_to_names('rei', *risk_burden_proportions.index)
    return risk_burden_proportions[[ 'mean_lb_ub', 'mean', 'lower', 'upper']]
    
    

