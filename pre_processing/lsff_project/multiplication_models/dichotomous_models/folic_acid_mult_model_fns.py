import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats 
import scipy.integrate as integrate

def generate_rr_deficiency_nofort_draws(mean, std, location_ids):
    """This function takes a distribution for the relative risk
    for lack of fortification of a particular nutrient and generates
    1,000 draws based on that distribution. The data is the duplicated
    so that it is the same for each location ID so that it can be easily
    used later in the calculations."""
    data = pd.DataFrame()    
    np.random.seed(7)
    data['rr'] = np.random.lognormal(mean, std, size=1000)
    draws = []
    for i in list(range(0,1000)):
        draws.append(f'draw_{i}')
    data['draws'] = draws
    data = pd.DataFrame.pivot_table(data, values='rr', columns='draws').reset_index().drop(columns=['index'])
    df = pd.DataFrame(np.repeat(data.values,len(location_ids),axis=0))
    df.columns = data.columns
    df['location_id'] = location_ids
    df = df.set_index('location_id')
    return df

def generate_coverage_parameter_draws(df):
    """This function is used to generate 1000 draws of nutrient/vehicle coverage parameters based on
    the mean value and confidence intervals. This function assumes a normal distribution of uncertainty
    within the confidence interval centered around the mean and is truncated at the bounds of 0 and 100%"""
    data_frame = df.copy()
    np.random.seed(11)
    for i in list(range(0,1000)):
        data_frame[f'draw_{i}'] = scipy.stats.truncnorm.rvs(data_frame.a,
                                                            data_frame.b,
                                                            data_frame.value_mean,
                                                            data_frame.value_std) / 100
    data_frame = (data_frame
                  .set_index(['location_id'])
                  .drop(columns=[c for c in data_frame.columns if 'draw' not in c
                                and c not in ['location_id','value_description']]))
    return data_frame


def generate_overall_coverage_rates(nutrient, vehicle, coverage_levels, years, location_ids):
    """This function generates baseline and counterfactual coverage rates of fortification for a specified
    nutrient and vehicle pair. The baseline coverage rates are assumed to remain constant from 2021 to 2025.
    The alternative coverage rates are assumed to jump from the baseline rate in 2021 to either 20/50/80 percent
    of the difference between the baseline rate (proportion of population eating fortified vehicle) and the
    current maximum coverage potential (proportion of population eating industrially produced vehicle) in 2022
    and then remains constant at that level through 2025."""
    coverage_data_path ='/ihme/homes/beatrixh/vivarium_data_analysis/pre_processing/lsff_project/data_prep/outputs/nigeria_ethiopia_india_coverage_data.csv'
    data = pd.read_csv(coverage_data_path)
    data = data.loc[data.location_id.isin(location_ids)].loc[data.sub_population!='women of reproductive age'].drop_duplicates()
    # the following is a transformation for a potential data issue and should be removed when resolved
    data['value_mean'] = data['value_mean'].replace(100, 100 - 0.00001 * 2)
    data['value_025_percentile'] = data['value_025_percentile'].replace(100, 100 - 0.00001 * 3)
    data['value_975_percentile'] = data['value_975_percentile'].replace(100, 100 - 0.00001)
    data = data.loc[data.vehicle == vehicle].loc[data.nutrient.isin([nutrient, 'na'])]
    data['value_std'] = (data.value_975_percentile - data.value_mean) / 1.96
    data['a'] = (0 - data.value_mean) / data.value_std
    data['b'] = (100 - data.value_mean) / data.value_std
    cov_a = data.loc[data.value_description == 'percent of population eating fortified vehicle'].drop(
        columns='value_description')
    cov_b = data.loc[data.value_description == 'percent of population eating industrially produced vehicle'].drop(
        columns='value_description')
    #cov_c = data.loc[data.value_description == 'percent of population eating vehicle'].drop(columns='value_description')
    cov_a = generate_coverage_parameter_draws(cov_a)
    cov_b = generate_coverage_parameter_draws(cov_b)
    #cov_c = generate_coverage_parameter_draws(cov_c)
    #assert np.all(cov_a <= cov_b) & np.all(cov_b <= cov_c), "Error: coverage parameters are not logically ordered"
    baseline_coverage = pd.DataFrame()
    for year in years:
        temp = cov_a.copy()
        temp['year'] = year
        baseline_coverage = pd.concat([baseline_coverage, temp])
    baseline_coverage = baseline_coverage.reset_index().set_index(['location_id', 'year']).sort_index()
    counterfactual_coverage = pd.DataFrame()
    for level in coverage_levels:
        cov = cov_a.copy()
        cov['year'] = years[0]
        for year in years[1:len(years)]:
            temp = cov_b * level
            temp['year'] = year
            cov = pd.concat([cov, temp])
        cov['coverage_level'] = level
        counterfactual_coverage = pd.concat([counterfactual_coverage, cov])
    counterfactual_coverage = (counterfactual_coverage.reset_index()
                               .set_index(['location_id', 'year', 'coverage_level']).sort_index())
    return baseline_coverage, counterfactual_coverage

def pull_dalys(cause_ids, nonfatal_cause_ids, location_ids, ages, sexes, index_cols):
    """This function pulls dalys for specified cause IDs from GBD"""

    ylds = get_draws(
        gbd_id_type='cause_id',
        gbd_id=cause_ids,
        source='como',
        measure_id=3,
        metric_id=3,  # only available as rate
        location_id=location_ids,
        year_id=2019,
        age_group_id=ages,
        sex_id=sexes,
        gbd_round_id=6,
        status='best',
        decomp_step='step5',
    ).set_index(index_cols + ['cause_id'])
    ylds = ylds.drop(columns=[c for c in ylds.columns if 'draw' not in c])
    pop = get_population(
        location_id=location_ids,
        year_id=2019,
        age_group_id=ages,
        sex_id=sexes,
        gbd_round_id=6,
        decomp_step='step4').set_index(index_cols)
    for i in list(range(0, 1000)):
        ylds[f'draw_{i}'] = ylds[f'draw_{i}'] * pop['population']
    ylls = get_draws(
        gbd_id_type='cause_id',
        gbd_id=cause_ids,
        source='codcorrect',
        measure_id=4,
        metric_id=1,
        location_id=location_ids,
        year_id=2019,
        age_group_id=ages,
        sex_id=sexes,
        gbd_round_id=6,
        status='latest',
        decomp_step='step5',
    ).set_index(index_cols + ['cause_id']).replace(np.nan, 0)
    ylls = ylls.drop(columns=[c for c in ylls.columns if 'draw' not in c])

    dalys = ylls + ylds
    
    return dalys


def age_split_dalys(dalys):
    """
    DALYs only available in multi-year bins; split ages 1-4 by population weight
    """
    location_ids = list(dalys.reset_index().location_id.unique())
    sexes = list(dalys.reset_index().sex_id.unique())
    
    age_split_pop_count = get_population(
        location_id=location_ids,
        year_id=2019,
        age_group_id=[49,50,51,52],
        single_year_age=True,
        sex_id=sexes,
        gbd_round_id=6,
        decomp_step='step4')
    age_split_pop_count['denom'] = age_split_pop_count.groupby('location_id').transform('sum').population
    age_split_pop_count['prop_1_4'] = age_split_pop_count.population / age_split_pop_count.denom
    age_split_pop_count['child_age_group_id'] = age_split_pop_count.age_group_id
    age_split_pop_count['age_group_id'] = 5 #age 1 to 4
    
    merge_cols = ['location_id','sex_id','age_group_id']
    dalys = dalys.reset_index().merge(age_split_pop_count[merge_cols + ['child_age_group_id','prop_1_4']], on = merge_cols, how = 'left')

    dalys.loc[(dalys.child_age_group_id.notna()),'age_group_id'] = dalys.child_age_group_id
    for c in [f'draw_{i}' for i in range(1_000)]:
        dalys.loc[dalys.child_age_group_id.notna(),c] = dalys[c] * dalys.prop_1_4

    dalys = dalys.drop(columns=['child_age_group_id','prop_1_4'])

    dalys = dalys.set_index(['location_id', 'sex_id', 'age_group_id', 'cause_id'])
    return dalys

def calc_dalys_averted(dalys, prop_averted_ntds):
    """
    INPUT:
    - absolute dalys df, with index =  (location_id, sex_id, age_group_id, cause_id), cols = draws
    - prop averted df, with index = (location_id, coverage_level), cols = draws
        - this is the percentage by which TOTAL dalys or birth_prev decreases, by strat level
    -------
    @requires: location_ids of two dfs match
    @returns: absolute DALYs averted df, with index = 
            (location_id, sex_id, age_group_id, cause_id, coverage_level), cols = draws
    """
    prop_averted_ntds = prop_averted_ntds.reset_index().set_index('location_id')
    out = pd.DataFrame()
    for level in prop_averted_ntds.coverage_level.unique():
        s = prop_averted_ntds[prop_averted_ntds.coverage_level==level].drop(columns='coverage_level')
        t = dalys * s
        t['coverage_level'] = level
        t = t.reset_index().set_index(['location_id', 'sex_id', 'age_group_id', 'cause_id','coverage_level'])
        out = out.append(t)
    return out

def is_affected(df, coverage_start_year = 2022):
    """
    INPUT:
    - df with index containing (age_group_id, year_id)
    - start year for fortification of wheat flour with folic acid
    ------
    @recommended: age_group_ids should be disjoint
    @returns: df with appended column, 'is_affected'\
                - TRUE if the specified age_group_id/year will receive be affected by the fortification
    """
    df = df.reset_index()
    age_map = {49:1,50:2,51:3,52:4,53:4,1:np.nan,2:0,3:0.02,4:0.08,5:np.nan}
    df['age_val'] = df.age_group_id.map(age_map)
    df['birth_year'] = df.year_id - (df.age_val)
    df['is_affected'] = df.birth_year > coverage_start_year
    
    df = df.drop(columns=['age_val','birth_year'])
    
    return df