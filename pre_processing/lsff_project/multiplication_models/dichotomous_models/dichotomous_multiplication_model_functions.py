import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

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


def generate_overall_coverage_rates(nutrient, vehicle, coverage_levels, years):
    """This function generates baseline and counterfactual coverage rates of fortification for a specified
    nutrient and vehicle pair. The baseline coverage rates are assumed to remain constant from 2021 to 2025.
    The alternative coverage rates are assumed to jump from the baseline rate in 2021 to either 20/50/80 percent
    of the difference between the baseline rate (proportion of population eating fortified vehicle) and the
    current maximum coverage potential (proportion of population eating industrially produced vehicle) in 2022
    and then remains constant at that level through 2025."""

    data = pd.read_csv(
        '/ihme/homes/alibow/notebooks/vivarium_data_analysis/pre_processing/lsff_project/data_prep/outputs/LSFF_extraction_clean_data_rich_locations_01_11_2021.csv')

    # the following is a transformation for a potential data issue and should be removed when resolved
    data['value_mean'] = data['value_mean'].replace(100, 100 - 0.00001 * 2)
    data['value_025_percentile'] = data['value_025_percentile'].replace(100, 100 - 0.00001 * 3)
    data['value_975_percentile'] = data['value_975_percentile'].replace(100, 100 - 0.00001)

    data = data.loc[data.vehicle == vehicle].loc[data.nutrient.isin([nutrient, 'na'])]
    data['value_std'] = (data.value_975_percentile - data.value_025_percentile) / 2 / 1.96
    data['a'] = (0 - data.value_mean) / data.value_std
    data['b'] = (100 - data.value_mean) / data.value_std

    cov_a = data.loc[data.value_description == 'percent of population eating fortified vehicle'].drop(
        columns='value_description')
    cov_b = data.loc[data.value_description == 'percent of population eating industrially produced vehicle'].drop(
        columns='value_description')
    cov_c = data.loc[data.value_description == 'percent of population eating vehicle'].drop(columns='value_description')

    cov_a = generate_coverage_parameter_draws(cov_a)
    cov_b = generate_coverage_parameter_draws(cov_b)
    cov_c = generate_coverage_parameter_draws(cov_c)

    assert np.all(cov_a <= cov_b) & np.all(cov_b <= cov_c), "Error: coverage parameters are not logically ordered"

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


def apply_vitamin_a_age_related_effective_coverage_restrictions(data,
                                                                sex_ids,
                                                                age_group_ids,
                                                                effective_fractions):
    """This function takes an dataframe of population coverage and generates a dataframe of *effective* coverage
    rates by age group, using the effective coverage assumptions for vitamin A by age (no effect of fortification under
    six months of age)"""
    final = pd.DataFrame()
    for n in list(range(0, len(sex_ids))):
        out_data = pd.DataFrame()
        for i in list(range(0, len(age_group_ids))):
            temp = (data * effective_fractions[i]).reset_index()
            temp['age_group_id'] = age_group_ids[i]
            out_data = pd.concat([out_data, temp], ignore_index=True)
        out_data['sex_id'] = sex_ids[n]
        final = pd.concat([final, out_data], ignore_index=True)
    final = (final.set_index(
        ['location_id', 'age_group_id', 'sex_id', 'year'] + [c for c in final.columns if c == 'coverage_level'])
             .sort_index())
    return final


def calculate_vitamin_a_time_lag_effective_fraction(df, years):
    """This function calculates the proportion of individuals covered by vitamin a fortification who
    are recieving an effect from the fortification based on the time lag assumptions (5 month delay
    from the start of new coverage until vitamin a fortification has an effect on vitamin a deficiency).
    This function also assumes that everyone who is covered at baseline has been covered for at least five
    months and therefore 100% of covered individuals are effectively covered at baseline."""
    final = pd.DataFrame()
    data = df.reset_index()
    for i in list(range(0, len(years))):
        current = (data.loc[data.year == years[i]]
                   .set_index([c for c in data.columns if 'draw' not in c and c != 'year'])
                   .drop(columns='year'))
        if i == 0:
            for draw in list(range(0, 1000)):
                current[f'draw_{draw}'] = 1
        else:
            prior = (data.loc[data.year == years[i - 1]]
                     .set_index([c for c in data.columns if 'draw' not in c and c != 'year'])
                     .drop(columns='year'))
            current = 1 - ((current - prior) * 5 / 12 / current)
        current['year'] = years[i]
        final = pd.concat([final, current])
    final = final.reset_index().set_index([c for c in data.columns if 'draw' not in c]).sort_index()
    return final


def get_effective_vitamin_a_coverage(df, sex_ids, age_group_ids, effective_fractions, years):
    """This function takes a total population coverage dataframe and applies age and time lag
    effective coverage restrictions for population levels of effective vitamin a fortification
    coverage by sex, age group, and year"""
    effective_coverage_by_age = apply_vitamin_a_age_related_effective_coverage_restrictions(df,
                                                                                            sex_ids,
                                                                                            age_group_ids,
                                                                                            effective_fractions)
    effective_fraction_by_time_lag = calculate_vitamin_a_time_lag_effective_fraction(df, years)
    effective_coverage = effective_coverage_by_age * effective_fraction_by_time_lag
    effective_coverage = (effective_coverage.reset_index()
                          .set_index(['location_id', 'sex_id', 'age_group_id', 'year'] +
                                     [c for c in effective_coverage.reset_index().columns if c == 'coverage_level'])
                          .sort_index())

    return effective_coverage


def generate_rr_deficiency_nofort_draws(mean, std, location_ids):
    import pandas as pd, numpy as np
    """This function takes a distribution for the relative risk
    for lack of fortification of a particular nutrient and generates
    1,000 draws based on that distribution. The data is the duplicated
    so that it is the same for each location ID so that it can be easily
    used later in the calculations."""
    data = pd.DataFrame()
    np.random.seed(7)
    data['rr'] = np.random.lognormal(mean, std, size=1000)
    draws = []
    for i in list(range(0, 1000)):
        draws.append(f'draw_{i}')
    data['draws'] = draws
    data = pd.DataFrame.pivot_table(data, values='rr', columns='draws').reset_index().drop(columns=['index'])
    df = pd.DataFrame(np.repeat(data.values, len(location_ids), axis=0))
    df.columns = data.columns
    df['location_id'] = location_ids
    df = df.set_index('location_id')
    return df


def pull_cause_specific_dalys_deficiency_pafs(rei_id,
                                              cause_ids,
                                              location_ids,
                                              ages,
                                              sexes,
                                              index_cols):
    """This function pulls PAF data from GBD for specified
    risk outcome pairs. Note that the risk in this context
    will/should be nutrient *deficiencies*, not the lack of
    nutrient fortification"""

    data = pd.DataFrame()
    for cause_id in cause_ids:
        temp = get_draws(
            gbd_id_type=['rei_id', 'cause_id'],
            gbd_id=[rei_id, cause_id],
            source='burdenator',
            measure_id=2,  # dalys
            metric_id=2,  # percent
            location_id=location_ids,
            year_id=2019,
            age_group_id=ages,
            sex_id=sexes,
            gbd_round_id=6,
            status='best',
            decomp_step='step5',
        )
        data = pd.concat([data, temp], ignore_index=True)
    data = data.set_index(index_cols + ['cause_id'])
    data = data.drop(columns=[c for c in data.columns if 'draw' not in c]).sort_index()
    return data


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
    for nf in nonfatal_cause_ids:
        nonfatal = ylls.groupby(index_cols).sum()
        nonfatal['cause_id'] = nf
        for i in list(range(0, 1000)):
            nonfatal[f'draw_{i}'] = 0
    ylls = pd.concat([ylls.reset_index(), nonfatal.reset_index()]).set_index(index_cols + ['cause_id'])

    dalys = ylls + ylds
    return dalys


def calculate_paf_deficiency_nofort(rr_deficiency_nofort, effective_baseline_coverage):
    """This function calculates the population attributable fraction of UNfortified food
    on the fortification outcome of interest (outcome defined in the fortification
    effect size, which is generally nutrient deficiency)

    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""

    paf_deficiency_nofort = ((rr_deficiency_nofort - 1) * (1 - effective_baseline_coverage)) / (
            (rr_deficiency_nofort - 1) * (1 - effective_baseline_coverage) + 1)
    return paf_deficiency_nofort


def calculate_pif_deficiency_nofort(paf_deficiency_nofort, effective_baseline_coverage, effective_alternative_coverage):
    """This function calculates the population impact fraction for UNfortified
    foods and nutrient deficiency based on the location-specific coverage
    levels of fortified foods; specifically, p (1 - proportion of population
    that eats fortified vehicle) and p_start (1 - proportion of population that
    eats industrially produced vehicle).

    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""
    pif_deficiency_nofort = paf_deficiency_nofort * (effective_alternative_coverage - effective_baseline_coverage) / (
            1 - effective_baseline_coverage)
    return pif_deficiency_nofort


def duplicate_over_simulation_years(df, years):
    data = df.reset_index()
    data_years = pd.DataFrame()
    for year in years:
        temp = data.copy()
        temp['year'] = year
        data_years = pd.concat([data_years, temp], ignore_index=True)
    data_years = data_years.set_index(['location_id', 'sex_id', 'age_group_id', 'year', 'cause_id']).sort_index()
    return data_years


def calculate_final_pifs_and_daly_reductions(pif_deficiency_nofort,
                                             paf_dalys_deficiency,
                                             dalys,
                                             coverage_levels, years):
    """This function calcualtes the PIF for fortification on DALYs as well as the
    overall reduction in the number of DALYs at the location, age group, sex,
    year, cause, draw, and coverage level specific level"""

    dalys_prepped = duplicate_over_simulation_years(dalys, years)
    paf_dalys_deficiency_prepped = duplicate_over_simulation_years(paf_dalys_deficiency, years)

    pif_dalys_nofort = pd.DataFrame()
    dalys_reduction = pd.DataFrame()
    for coverage_level in coverage_levels:
        pif_deficiency_nofort_level = (pif_deficiency_nofort.reset_index()
                                       .loc[pif_deficiency_nofort.reset_index().coverage_level == coverage_level]
                                       .drop(columns='coverage_level')
                                       .set_index(['location_id', 'sex_id', 'age_group_id', 'year']))
        pif_dalys_nofort_level = pif_deficiency_nofort_level * paf_dalys_deficiency_prepped
        daly_reduction_level = pif_dalys_nofort_level * dalys_prepped

        pif_dalys_nofort_level['coverage_level'] = coverage_level
        daly_reduction_level['coverage_level'] = coverage_level
        pif_dalys_nofort = pd.concat([pif_dalys_nofort, pif_dalys_nofort_level])
        dalys_reduction = pd.concat([dalys_reduction, daly_reduction_level])

    pif_dalys_nofort = (pif_dalys_nofort.reset_index()
                        .set_index([c for c in pif_dalys_nofort.reset_index().columns if 'draw' not in c]))
    daly_reduction = (dalys_reduction.reset_index()
                      .set_index([c for c in dalys_reduction.reset_index().columns if 'draw' not in c])
                      .replace(np.nan, 0))

    return pif_dalys_nofort, daly_reduction