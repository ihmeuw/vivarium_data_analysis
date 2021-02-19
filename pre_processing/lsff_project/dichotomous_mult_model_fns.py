import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats 
import scipy.integrate as integrate
import matplotlib.pyplot as plt

## GLOBALS

location_ids = [163, 214, 205, 190, 189]
location_ids = [163, 214]

"""Note: full set of location IDs is shown below, but subset used here
was selected because they are the locations with non-missing coverage data
for the nutrient and vehicle of interest (vitamin A/oil)

[168, 161, 201, 202, 6, 205, 171, 141, 179, 207, 163, 11, 180, 181,
184, 15, 164, 213, 214, 165, 196, 522, 190, 189, 20]"""

ages = [1,2,3,4,5]
sexes = [1,2]

index_cols=['location_id','sex_id','age_group_id']

# define alternative scenario coverage levels (low, medium, high)
    # this parameter represents the proportion of additional coverage achieved in the
    # alternative scenario, defined as the difference between the proportion of the population
    # that eats the fortified vehicle and the proportion of the population that eats 
    # the industrially produced vehicle
alternative_scenario_coverage_levels = [0.2, 0.5, 0.8]

## FNS

def calculate_paf_deficiency_nofort(rr_deficiency_nofort, alpha):
    """This function calculates the population attributable fraction of UNfortified food
    on the fortification outcome of interest (outcome defined in the fortification 
    effect size, which is generally nutrient deficiency)
    
    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""
       
    paf_deficiency_nofort = ((rr_deficiency_nofort - 1) * (1 - alpha)) / ((rr_deficiency_nofort - 1) * (1 - alpha) + 1)
    return paf_deficiency_nofort

def calculate_pif_deficiency_nofort(paf_deficiency_nofort, alpha, alpha_star):
    """This function calculates the population impact fraction for UNfortified 
    foods and nutrient deficiency based on the location-specific coverage
    levels of fortified foods; specifically, p (1 - proportion of population
    that eats fortified vehicle) and p_start (1 - proportion of population that 
    eats industrially produced vehicle).
    
    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""
    pif_deficiency_nofort = paf_deficiency_nofort * (alpha_star - alpha) / (1 - alpha)
    return pif_deficiency_nofort

def calculate_daly_reduction_by_cause(pif_deficiency_nofort, paf_dalys_deficiency, dalys):
    """This functionc calculates the population impact fraction for UNfortified 
    food and DALYs due to specific causes as well as the total number of DALYs
    averted by cause, sex, and age
    
    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""
    
    df = pd.DataFrame()
    
    for level in ['low','medium','high','full']:
        pif_deficiency_nofort_level = (pif_deficiency_nofort.reset_index()
                                     .loc[pif_deficiency_nofort.reset_index().coverage_level == level]
                                     .drop(columns='coverage_level')
                                     .set_index('location_id'))
        pif_dalys_nofort = pif_deficiency_nofort_level * paf_dalys_deficiency
        pif_dalys_nofort['measure'] = 'pif'
        dalys_reduction = pif_dalys_nofort * dalys
        dalys_reduction['measure'] = 'dalys averted'
        dalys_reduction_overall = dalys_reduction.reset_index().groupby(index_cols + ['measure']).sum().reset_index()
        dalys_reduction_overall['cause_id'] = 294
        data = (pd.concat([pif_dalys_nofort.reset_index(), dalys_reduction.reset_index(), dalys_reduction_overall], ignore_index=True))
        data['coverage_level'] = level
        data = data.set_index(index_cols + ['measure','cause_id','coverage_level']).dropna().sort_index()
        df = pd.concat([df,data])
        
    return df

def load_alternative_coverage_data(nutrient, vehicle):
    data = pd.read_csv('/ihme/homes/beatrixh/vivarium_data_analysis/pre_processing/lsff_project/data_prep/outputs/LSFF_extraction_clean_data_rich_locations_01_11_2021.csv')
    alpha = (data.loc[data.vehicle == vehicle]
             .loc[data.nutrient == nutrient]
             .loc[data.value_description == 'percent of population eating fortified vehicle'])
    alpha_star = (data.loc[data.vehicle == vehicle]
                  .loc[data.value_description == 'percent of population eating industrially produced vehicle'])

    
    # generate draws
    """This currently relies on two major assumptions:
    1. Truncated normal distribution
    2. The same percentile from the eats_fortified and eats_fortifiable distributions sampled for each draw
    
    Assumption number two is likely overly restrictive, but was chosen such that eats_fortified will 
    always be less than eats_fortifiable at the draw level (this is consistent with methodology described
    in 2017 concept model, but is achieved by setting the same random seed to sample each of these
    parameters)"""
      
    for data in [alpha, alpha_star]:
              
        data['value_std'] = (data.value_975_percentile - data.value_025_percentile) / 2 / 1.96
        data['a'] = (data.value_025_percentile - data.value_mean) / data.value_std
        data['b'] = (data.value_975_percentile - data.value_mean) / data.value_std       
        np.random.seed(1246)
        for i in list(range(0,1000)):
            data[f'draw_{i}'] = scipy.stats.truncnorm.rvs(data.a, data.b, data.value_mean, data.value_std) / 100
            
    alpha = (alpha.set_index('location_id')
         .drop(columns=[c for c in alpha.columns if 'draw' not in c and c != 'location_id']))
    alpha_star = (alpha_star.set_index('location_id')
         .drop(columns=[c for c in alpha_star.columns if 'draw' not in c and c != 'location_id']))
    alpha_star_low = (alpha_star) * alternative_scenario_coverage_levels[0]
    alpha_star_low['coverage_level'] = 'low'
    alpha_star_med = (alpha_star) * alternative_scenario_coverage_levels[1]
    alpha_star_med['coverage_level'] = 'medium'
    alpha_star_high = (alpha_star) * alternative_scenario_coverage_levels[2]
    alpha_star_high['coverage_level'] = 'high'
    
    alpha_star = pd.concat([alpha_star_low.reset_index(), 
                            alpha_star_med.reset_index(), 
                            alpha_star_high.reset_index()], 
                           ignore_index=True)
    alpha_star = alpha_star.set_index([c for c in alpha_star.columns if 'draw' not in c])
    
    #p = 1 - alpha
    #p_star = 1 - alpha_star
    
    return alpha, alpha_star

def load_coverage_data(nutrient, vehicle):
    data = pd.read_csv('/ihme/homes/beatrixh/vivarium_data_analysis/pre_processing/lsff_project/data_prep/outputs/LSFF_extraction_clean_data_rich_locations_01_11_2021.csv')
    alpha = (data.loc[data.vehicle == vehicle]
             .loc[data.nutrient == nutrient]
             .loc[data.value_description == 'percent of population eating fortified vehicle'])
    alpha_star = (data.loc[data.vehicle == vehicle]
                  .loc[data.value_description == 'percent of population eating industrially produced vehicle'])

    
    # generate draws
    """This currently relies on two major assumptions:
    1. Truncated normal distribution
    2. The same percentile from the eats_fortified and eats_fortifiable distributions sampled for each draw
    
    Assumption number two is likely overly restrictive, but was chosen such that eats_fortified will 
    always be less than eats_fortifiable at the draw level (this is consistent with methodology described
    in 2017 concept model, but is achieved by setting the same random seed to sample each of these
    parameters)"""
      
    for data in [alpha, alpha_star]:
              
        data['value_std'] = (data.value_975_percentile - data.value_025_percentile) / 2 / 1.96
        data['a'] = (data.value_025_percentile - data.value_mean) / data.value_std
        data['b'] = (data.value_975_percentile - data.value_mean) / data.value_std       
        np.random.seed(11)
        for i in list(range(0,1000)):
            data[f'draw_{i}'] = scipy.stats.truncnorm.rvs(data.a, data.b, data.value_mean, data.value_std) / 100
            
    alpha = (alpha.set_index('location_id')
         .drop(columns=[c for c in alpha.columns if 'draw' not in c and c != 'location_id']))
    alpha_star = (alpha_star.set_index('location_id')
         .drop(columns=[c for c in alpha_star.columns if 'draw' not in c and c != 'location_id']))
    alpha_star_low = (alpha_star - alpha) * alternative_scenario_coverage_levels[0] + alpha
    alpha_star_low['coverage_level'] = 'low'
    alpha_star_med = (alpha_star - alpha) * alternative_scenario_coverage_levels[1] + alpha
    alpha_star_med['coverage_level'] = 'medium'
    alpha_star_high = (alpha_star - alpha) * alternative_scenario_coverage_levels[2] + alpha
    alpha_star_high['coverage_level'] = 'high'
    alpha_star_full = alpha_star.copy()
    alpha_star_full['coverage_level'] = 'full'
    
    alpha_star = pd.concat([alpha_star_low.reset_index(), 
                            alpha_star_med.reset_index(), 
                            alpha_star_high.reset_index(),
                            alpha_star_full.reset_index()], 
                           ignore_index=True)
    alpha_star = alpha_star.set_index([c for c in alpha_star.columns if 'draw' not in c])
    
    #p = 1 - alpha
    #p_star = 1 - alpha_star
    
    return alpha, alpha_star

def generate_rr_deficiency_nofort_draws(mean, std):
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
    
    ## age-split dalys
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

#     dalys = dalys.drop(columns=['child_age_group_id','prop_1_4'])

    dalys = dalys.set_index(['location_id', 'sex_id', 'age_group_id', 'cause_id'])
    
    return dalys