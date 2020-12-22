import pandas as pd, numpy as np
import sys, os.path
from collections import namedtuple
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

def get_or_append_global_location(locations=None):
    """Returns a dataframe with the name and id of the 'Global' location,
    or appends this data to the passed locations dataframe if not None.
    """
    global_loc = pd.DataFrame({'location_name':'Global', 'location_id':1}, index=[0])
    return global_loc if locations is None else locations.append(global_loc, ignore_index=True)

def split_global_from_other_locations(df):
    """Splits df into two subdataframes and returns the pair of dataframes:
    The first is where df.location_id is NOT the 'Global' id (1);
    The second is where df.location_id IS the 'Global' id (1).
    """
    SubDataFramesByLocation = namedtuple('SubDataFramesByLocation', 'other_locations, global_location')
    return SubDataFramesByLocation(df.query("location_id !=1"), df.query("location_id==1"))

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

def pull_dalys_attributable_to_risk_for_locations(location_ids, *risk_factor_names):
    """Calls get_draws to pull all-cause DALYs attributable to the specified risk for the specified locations.
    The call does not specify the age_group_id, so it will pull DALYs for all age groups that contriibute DALYs.
    Estimates from age group aggregates are not available from the burdenator.
    """
    risk_ids = list_ids('rei', *risk_factor_names)
    if isinstance(risk_ids, int): risk_ids = [risk_ids]
    burden = get_draws(
        gbd_id_type=['rei_id']*len(risk_ids) + ['cause_id'], # Types must match gbd_id's
        gbd_id=[*risk_ids, list_ids('cause', 'All causes')],
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

def pull_dalys_due_to_cause_for_locations(location_ids, *cause_names):
    dalys = get_draws(
        gbd_id_type='cause_id',
        gbd_id=list_ids('cause', *cause_names),
        source='dalynator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=list_ids('metric', 'Number'),
        location_id=list(location_ids),
        year_id=2019,
        sex_id=list_ids('sex', 'Male', 'Female'),
        age_group_id=list_ids('age_group', 'All Ages'), #22,#
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    return dalys

def concatenate_risk_and_cause_burdens(risk_burdens, cause_burdens):
    """Concatenates the risk and cause DALY dataframes returned by get_draws,
    adding a name column for the risk or cause, and dos some renaming and dropping of unncessary columns. 
    """
    risk_burdens = add_entity_names(risk_burdens, 'rei')
    risk_burdens = risk_burdens.drop(columns='cause_id').rename(
        columns={'rei_id':'gbd_id', 'rei_name': 'gbd_entity_name'})
    cause_burdens = add_entity_names(cause_burdens, 'cause')
    cause_burdens = cause_burdens.rename(
        columns={'cause_id':'gbd_id',  'cause_name': 'gbd_entity_name'})
    risk_burdens['gbd_id_type'] = 'rei'
    cause_burdens['gbd_id_type'] = 'cause'
    all_data = pd.concat([risk_burdens, cause_burdens], ignore_index=True, copy=False)
    all_data = replace_ids_with_names(all_data, 'measure', 'metric')
    return all_data

def aggregate_draws_over_columns(df, marginalized_cols):
    """Aggregates (by summing) over the specified columns in the passed dataframe, draw by draw."""
    draw_cols = df.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    index_cols = df.columns.difference([*draw_cols, *marginalized_cols]).to_list()
    return df.groupby(index_cols)[draw_cols].sum()

def calculate_proportion_global_vad_burden(vad_burden_for_locations, global_vad_burden):
    """Calculates percent of the global burden (in DALYs) of vitamin A deficiency for each of the locations
    in `vad_burden_for_locations`.
    """
    marginalized_cols = ['age_group_id', 'sex_id']
    vad_burden_for_locations = aggregate_draws_over_columns(vad_burden_for_locations, marginalized_cols)
    global_vad_burden = aggregate_draws_over_columns(global_vad_burden, marginalized_cols)
    # Reset the location_id level in denominator to broadcast over global instead of trying to match location
    return vad_burden_for_locations / global_vad_burden.reset_index('location_id', drop=True)

def calculate_proportion_global_burden(burden_for_locations, global_burden):
    """Calculates percent of the global burden of a risk or cause for each of the locations
    in `burden_for_locations`.
    """
    marginalized_cols = ['age_group_id', 'sex_id']
    burden_for_locations = aggregate_draws_over_columns(burden_for_locations, marginalized_cols)
    global_burden = aggregate_draws_over_columns(global_burden, marginalized_cols)
    # Reset the location_id level in denominator to broadcast over global instead of trying to match location
    proportion = burden_for_locations / global_burden.reset_index('location_id', drop=True)
    # I thought it was confusing to have 'Number' as the metric for a proportion, but on the other hand,
    # leaving the metric_id alone tells what data was used in the computation
#     if 'metric_id' in proportion.index.names:
#         proportion.index.set_levels([list_ids('metric', 'Rate')], level='metric_id', inplace=True)
    return proportion

# def add_global_location(locations):
#     locations = locations[['location_name', 'location_id']]
#     global_loc = pd.Series(['Global', 1], index=locations.columns)
#     return locations.append(global_loc, ignore_index=True)

# def calculate_all_proportion_global_burdens(locations_key):
#     """Claculate"""
#     locations = get_locations(locations_key)[['location_name', 'location_id']]
# #     global_loc = pd.Series(['Global', 1], index=locations.columns)
# #     locations = locations.append(global_loc, ignore_index=True)
#     locations = add_global_location(locations)
    
#     risks = ['Vitamin A deficiency', 'Zinc deficiency', 'Iron deficiency']
#     risk_burdens = [pull_dalys_attributable_to_risk_for_locations(risk, locations.location_id) for risk in risks]
# #     vad_burden = pull_dalys_attributable_to_risk_for_locations('Vitamin A deficiency', locations.location_id)
# #     zinc_burden = pull_dalys_attributable_to_risk_for_locations('Zinc deficiency', locations.location_id)
# #     iron_burden = pull_dalys_attributable_to_risk_for_locations('Iron deficiency', locations.location_id)

#     risk_burden_proportions = [
#         calculate_proportion_global_burden(
#             risk_burden.query(f"location_id != {global_loc.location_id}"),
#             risk_burden.query(f"location_id == {global_loc.location_id}")
#         ) for risk_burden in risk_burdens
#     ]
    
#     return pd.concat(risk_burden_proportions) # return pd.concat of this?

def summarize_risk_proportions(risk_burden_proportions):
    # Sum proportion over all locations for each risk
    risk_burden_proportions = risk_burden_proportions.groupby(['rei_id']).sum()
    risk_burden_proportions['mean'] = risk_burden_proportions.mean(axis=1)
    risk_burden_proportions['lower'] = risk_burden_proportions.quantile(0.025, axis=1)
    risk_burden_proportions['upper'] = risk_burden_proportions.quantile(0.975, axis=1)
    risk_burden_proportions['mean_lb_ub'] = risk_burden_proportions.apply(
        lambda row: f"{row['mean']:.2%} ({row['lower']:.2%}, {row['upper']:.2%})", axis=1)
    risk_burden_proportions.index = ids_to_names('rei', *risk_burden_proportions.index)
    return risk_burden_proportions[['mean_lb_ub', 'mean', 'lower', 'upper']]

def summarize_burden_proportions_across_locations(burden_proportions):
    # Sum proportion over all locations for each risk and cause
#     burden_proportions = burden_proportions.groupby(['gbd_id_type', 'gbd_id']).sum()
    burden_proportions = aggregate_draws_over_columns(burden_proportions.reset_index(), ['location_id'])
    burden_proportions['mean'] = burden_proportions.mean(axis=1)
    burden_proportions['lower'] = burden_proportions.quantile(0.025, axis=1)
    burden_proportions['upper'] = burden_proportions.quantile(0.975, axis=1)
    burden_proportions['mean_lb_ub'] = burden_proportions.apply(
        lambda row: f"{row['mean']:.2%} ({row['lower']:.2%}, {row['upper']:.2%})", axis=1)
#     burden_proportions.loc['rei', 'gbd_entity_name'] = list(
#         ids_to_names('rei', *burden_proportions.loc['rei'].index)
#     )
#     burden_proportions.loc['cause', 'gbd_entity_name'] = list(
#         ids_to_names('cause', *burden_proportions.loc['cause'].index)
#     )
#     burden_proportions.set_index('gbd_entity_name', append=True, inplace=True)
    return burden_proportions[['mean_lb_ub', 'mean', 'lower', 'upper']]

def get_iron_dalys_by_subpopulation1(iron_burden):
#     wra_age_groups = range(list_ids('age_group', '10 to 14'), list_ids('age_group', '55 to 59'))
#     under5_age_groups = range(list_ids('age_group', 'Early Neonatal'), list_ids('age_group', '5 to 9'))
#     male_id = list_ids('sex', 'Male')
    
#     wra_query = f"sex_id==2 and age_group_id in {wra_age_groups}"
#     under5_query = f"age_group_id in {under5_age_groups}"
#     # rest of population = At least 5 years old and (Male or not between ages 10 and 54)
#     other_pop_query = f"(age_group_id not in {under5_age_groups}) and " \
#                       f"(sex_id == {male_id} or age_group_id not in {wra_age_groups})"
    
    wra_query = "sex_id==2 and 7 <= age_group_id <= 15" # Female and '10 to 15' to '50 to 54'
    under5_query = "age_group_id <= 5" # id 5 is '1 to 4' age group
    # rest of population = At least 5 years old and (Male or not between ages 10 and 54)
    other_pop_query = "age_group_id > 5 and (sex_id == 1 or age_group_id < 7 or age_group_id > 15)"
    
    iron_wra = iron_burden.query(wra_query)
    iron_under5 = iron_burden.query(under5_query)
    iron_other = iron_burden.query(other_pop_query)
    
    subpops = ('WRA', 'Under 5', 'Other')
    dfs = (iron_wra, iron_under5, iron_other)
    totals = [aggregate_draws_over_columns(df, ['age_group_id', 'sex_id', 'location_id']) for df in dfs]
    totals = [df.T.describe(percentiles=[0.025,0.975]) for df in totals]
    for df in totals:
        df.columns=[0]
    d = {subpop: df for subpop, df in zip(subpops, totals)}
    
#     return iron_wra, iron_under5, iron_other
    return {'WRA':totals[0], 'Under 5': totals[1], 'Other': totals[2]}

def get_iron_data(risk_burdens):
    """Selects and formats the iron deficiency data from the risk_burdens dataframe."""
    iron_deficiency_id = list_ids('rei', 'Iron deficiency')
    iron_burden = risk_burdens.query('rei_id == @iron_deficiency_id')
    iron_burden = add_entity_names(iron_burden)
    iron_burden = drop_id_columns(iron_burden, 'rei', 'location', keep=True) # Drop all id columns except rei and location
    return iron_burden

def get_iron_dalys_by_subpopulation(iron_burden):
    # Female and '10 to 15' to '50 to 54'
    wra = (iron_burden.sex_id == 2) & (iron_burden.age_group_id >=7) & (iron_burden.age_group_id <= 15)
    under5 = iron_burden.age_group_id <=5 # id 5 is '1 to 4' age group
    five_to_nine = iron_burden.age_group_id == 6 # id 6 is '5 to 9' age group
    
#     iron_burden = iron_burden.assign(subpopulation = np.select([wra, under5], ['WRA', 'Under 5'], default='Other'))
    iron_burden = iron_burden.assign(subpopulation = np.select(
        [wra, under5, five_to_nine], ['WRA', 'Under 5', '5 to 9'], default='Other'))
    return aggregate_draws_over_columns(iron_burden, ['age_group_id', 'sex_id'])

def summarize_iron_dalys(iron_dalys_by_subpopulation):
#     dalys = iron_dalys_by_subpopulation.groupby('subpopulation').sum()
    dalys = aggregate_draws_over_columns(iron_dalys_by_subpopulation.reset_index(), ['location_id'])
    return dalys.T.describe(percentiles=[0.025, 0.975]).T

def summarize_draws_across_locations(df):
    """Aggregates draws over locations in df, then calls .describe() to compute statistics for all draws.
    The dataframe df is assumed to have all relevant data stored in columns, not in its index.
    """
    df = aggregate_draws_over_columns(df, ['location_id'])
    return df.T.describe(percentiles=[0.025, 0.975]).T

def format_summarized_data(summary, number_format='', multiplier=1):
    """Format the mean, lower, and upper values from the summarized data."""
    if number_format == 'percent':
        number_format = '.2%'
    elif number_format == 'count':
        number_format = ',.0f'
    
    units = f'_per_{multiplier}' if multiplier != 1 else ''

    def print_number(x):
#         return eval(f'f"{{x*{multiplier}:{number_format}}}"')
        return f"{x*multiplier:{number_format}}"

    summary = summary.rename(columns={'2.5%':'lower', '97.5%':'upper'})
    cols = ['mean', 'lower', 'upper']
    summary = summary[cols]
    for col in cols:
        summary[f'{col}{units}_formatted'] = summary[col].apply(print_number)
    summary['mean_lower_upper'] = summary.apply(
        lambda row: f"{row[f'mean{units}_formatted']} ({row[f'lower{units}_formatted']}, {row[f'upper{units}_formatted']})",
        axis=1
    )
#     fstring = f"""f'{{row["mean"]:{number_format}}} ({{row["lower"]:{number_format}}}, {{row["upper"]:{number_format}}})'"""
#     def print_mean_lower_upper(row):
#         return eval(fstring)
#     summary['mean_lb_ub'] = summary.apply(print_mean_lower_upper, axis=1)
    return summary

