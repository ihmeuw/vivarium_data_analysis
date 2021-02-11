import pandas as pd, numpy as np
import sys, os.path
# from pandas.api.types import CategoricalDtype
from collections import namedtuple

sys.path.append(os.path.abspath("../.."))
from pre_processing.id_helper import *
import rank_countries_by_stunting as rbs

from db_queries import get_ids, get_population, get_outputs, get_best_model_versions
from get_draws.api import get_draws

def get_locations(key, data_directory='data'):
    """Reads the location file for the specified description key."""
    location_files = {
        'all': 'all_countries_with_ids.csv',
        'original': 'bgmf_countries_with_ids.csv',
        'top25': 'bmgf_top_25_countries_20201203.csv',
    }
    filepath = f'{data_directory}/{location_files[key]}'
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
    return df.groupby(index_cols, observed=True)[draw_cols].sum() # observed=True needed for Categorical data

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

def get_iron_data(risk_burdens):
    """Selects and formats the iron deficiency data from the risk_burdens dataframe."""
    iron_deficiency_id = list_ids('rei', 'Iron deficiency')
    iron_burden = risk_burdens.query('rei_id == @iron_deficiency_id')
    iron_burden = add_entity_names(iron_burden, 'rei')
    iron_burden = replace_ids_with_names(iron_burden, 'measure', 'metric')
#     iron_burden = drop_id_columns(iron_burden, 'rei', 'location', keep=True) # Drop all id columns except rei and location
    return iron_burden

def get_iron_dalys_by_subpopulation(iron_burden, subpopulations='under5_wra'):
    """
    """
    pops_to_masks = {
        # Female and '10 to 14' to '50 to 54'
        'WRA (10-54)': (iron_burden.sex_id==2) & (iron_burden.age_group_id>=7) & (iron_burden.age_group_id<=15),
        'Females 15-54': (iron_burden.sex_id==2) & (iron_burden.age_group_id>=8) & (iron_burden.age_group_id<=15),
        'Under 5': iron_burden.age_group_id<=5, # id 5 is '1 to 4' age group
        '5-9': iron_burden.age_group_id==6, # id 6 is '5 to 9' age group
        'Males 10-14': (iron_burden.sex_id==1) & (iron_burden.age_group_id==7),
        'Females 10-14': (iron_burden.sex_id==2) & (iron_burden.age_group_id==7),
    }
    
    if subpopulations == 'under5_wra':
        subpopulations = ['Under 5', 'WRA (10-54)']
    elif subpopulations == 'under5_5to9_wra':
        subpopulations = ['Under 5', '5-9', 'WRA (10-54)']
    elif subpopulations == 'under5_5to9_m10to14_wra':
        subpopulations = ['Under 5', '5-9', 'Males 10-14', 'WRA (10-54)']
    elif subpopulations == 'under5_5to9_mf10to14_f15to54':
        subpopulations = ['Under 5', '5-9', 'Males 10-14', 'Females 10-14', 'Females 15-54']

    pops_to_masks = {pop: mask for pop, mask in pops_to_masks.items() if pop in subpopulations}

    # Check that subpopulations are mutually exclusive
    masks = list(pops_to_masks.values())
    assert all((~(m1 & m2)).all() for m1,m2 in zip(masks[:-1], masks[1:]))

#     iron_burden = iron_burden.assign(subpopulation = np.select([wra, under5], ['WRA', 'Under 5'], default='Other'))
#     iron_burden = iron_burden.assign(subpopulation = np.select(
#         [wra, under5, five_to_nine], ['WRA', 'Under 5', '5 to 9'], default='Other'))
    iron_burden = iron_burden.assign(
        subpopulation = pd.Categorical(
            np.select(list(pops_to_masks.values()), list(pops_to_masks.keys()), default='Other'),
            categories=['Under 5', '5-9', 'Males 10-14', 'Females 10-14', 'Females 15-54', 'WRA (10-54)', 'Other'],
            ordered=True
        )
    )
    return aggregate_draws_over_columns(iron_burden, ['age_group_id', 'sex_id'])

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
    return summary

def summarize_percent_global_burdens(risk_dalys=None, cause_dalys=None, location_key=None, save_filepath=None):
    """Returns a dataframe summarizing the percent global burdens of risks and causes, optionally saving the file.
    If either risk_dalys or cause_dalys is None, then location_key must not be None.
    """
    # Process arguments
    # TODO: Probably edit this to pass location ids (like below) instead of a location key
    if location_key is not None:
        locations = get_locations(location_key)
        locations = get_or_append_global_location(locations)

    if risk_dalys is None:
        risks = ['Vitamin A deficiency', 'Zinc deficiency', 'Iron deficiency']
        risk_dalys = pull_dalys_attributable_to_risk_for_locations(locations.location_id, *risks)
    elif location_key is not None:
        risk_dalys = risk_dalys.loc[risk_dalys.location_id.isin(locations.location_id)]

    if cause_dalys is None:
        cause_dalys = pull_dalys_due_to_cause_for_locations(locations.location_id, 'Neural tube defects')
    elif location_key is not None:
        cause_dalys = cause_dalys.loc[cause_dalys.location_id.isin(locations.location_id)]

    # Concatenate risks and causes, calculate proportion global burden, summarize over draws, format, and save
    all_dalys = concatenate_risk_and_cause_burdens(risk_dalys, cause_dalys)

    proportion_global_burden = calculate_proportion_global_burden(
        *split_global_from_other_locations(all_dalys)
    )

    burden_summary = (proportion_global_burden
                      .reset_index()
                      .pipe(summarize_draws_across_locations)
                      .pipe(format_summarized_data, number_format='percent')
                      .sort_values(['gbd_id_type', 'gbd_id'], ascending=[False, True])
                     )
    if save_filepath is not None:
        burden_summary.to_csv(save_filepath)
    return burden_summary

def summarize_iron_burden_by_subpopulation(risk_dalys=None, location_ids=None, save_filepaths=None):
    """Summarizes DALY burden due to Iron Deficiency by subpopulation for the specified locations,
    and calculates the proportion of global DALY burden by subpopulation.
    """
    # Process arguments (I think it might be better to pass location ids instead of a location key as above)
    if location_ids is not None:
        all_location_ids = list(location_ids)
        global_id = get_or_append_global_location().at[0,'location_id']
        if global_id not in all_location_ids:
            all_location_ids.append(global_id)

    if risk_dalys is None:
        risk_dalys = pull_dalys_attributable_to_risk_for_locations(all_location_ids, 'Iron deficiency')
    elif location_ids is not None:
        risk_dalys = risk_dalys.loc[risk_dalys.location_id.isin(all_location_ids)]

    if save_filepaths is None:
        save_filepaths = [None, None, None]

    # Calculate iron deficiency DALYs by subpopulation, summarize across draws, format and save results
    iron_dalys_by_subpop = (risk_dalys
                            .pipe(get_iron_data)
                            .pipe(get_iron_dalys_by_subpopulation, subpopulations='under5_wra')
                            .pipe(split_global_from_other_locations)
                           ) # Result is namedtuple with fields 'other_locations' and 'global_location'

    iron_subpop_dalys_other_summary = (iron_dalys_by_subpop.other_locations
                                        .reset_index()
                                        .pipe(summarize_draws_across_locations)
                                        .pipe(format_summarized_data, number_format='count')
                                       )
    if save_filepaths[0] is not None:
        iron_subpop_dalys_other_summary.to_csv(save_filepaths[0])

    iron_subpop_dalys_global_summary = (iron_dalys_by_subpop.global_location
                                        .reset_index()
                                        .pipe(summarize_draws_across_locations) # (Only location is global, but that's fine)
                                        .pipe(format_summarized_data, number_format='count')
                                       )
    if save_filepaths[1] is not None:
        iron_subpop_dalys_global_summary.to_csv(save_filepaths[1])

    # Calculate percent of global DALYS for each subpopulation, summarize across draws, format and save results
    proportion_global_iron_dalys_by_subpop = calculate_proportion_global_burden(
        iron_dalys_by_subpop.other_locations.reset_index(),
        iron_dalys_by_subpop.global_location.reset_index()
    )

    percent_global_iron_dalys_by_subpop_summary = (proportion_global_iron_dalys_by_subpop
                                               .reset_index()
                                               .pipe(summarize_draws_across_locations)
                                               .pipe(format_summarized_data, number_format='percent')
                                              )
    if save_filepaths[2] is not None:
        percent_global_iron_dalys_by_subpop_summary.to_csv(save_filepaths[2])

    return iron_subpop_dalys_other_summary, iron_subpop_dalys_global_summary, percent_global_iron_dalys_by_subpop_summary

