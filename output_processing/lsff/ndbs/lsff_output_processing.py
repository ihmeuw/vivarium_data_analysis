import pandas as pd

INDEX_COLUMNS = ['input_draw', 'scenario']
VALUE_COLUMN = 'value'

def conditional_risk_of_ntds(
    births_with_ntd: pd.DataFrame, 
    live_births: pd.DataFrame, 
    conditioned_on: list,
    multiplier=1) -> pd.DataFrame:
    """
    Returns a dataframe with the contitional risk of neural tube defects
    (measured by birth prevalence) in each subgroup in the specified
    categories.
    """
    # Columns in both dataframes are:
    # ['year', 'sex', 'fortification_group', 'measure', 'input_draw', 'scenario', 'value']
    
    # The index columns will NOT be aggregated over
    index_columns = INDEX_COLUMNS + conditioned_on
    
    # In both dataframes, group by the index columns, and aggregate
    #'value' column over remaining columns by summing
    births_with_ntd = births_with_ntd.groupby(index_columns).value.sum()
    live_births = live_births.groupby(index_columns).value.sum()
    
    # Divide the two pandas Series to get birth prevalence
    # in each subgroup we conditioned on.
    # Multiply by the multiplier to get desired units (e.g. per 1000 live births)
    ntd_risk = multiplier * births_with_ntd / live_births
    
    # Drop any rows where we divided by 0 because there were no births
    ntd_risk.dropna(inplace=True)
    
    return ntd_risk.reset_index()

def rate_or_ratio(numerator, denominator,
                  numerator_strata, denominator_strata,
                  multiplier=1
                 ):
    index_cols = INDEX_COLUMNS
    
    # When we divide, the numerator strata must contain the denominator strata,
    # and the difference is the columns to broadcast over.

    broadcast_cols = sorted(
        set(numerator_strata) - set(denominator_strata),
        key=numerator_strata.index
    )
    
    numerator = numerator.groupby(denominator_strata+index_cols+broadcast_cols).value.sum()
    denominator = denominator.groupby(denominator_strata+index_cols).value.sum()
    
    rate_or_ratio = multiplier * numerator / denominator
    
    return rate_or_ratio.reset_index()

def averted(measure, scenario_col, baseline_value):
    baseline = measure[measure[scenario_col] == baseline_value]
    intervention = measure[measure[scenario_col] != baseline_value]
    
    index_columns = list(set(baseline.columns) - set([scenario_col, 'value']))
#     print(index_columns)
    
    baseline = baseline.set_index(index_columns).value
    intervention = intervention.set_index(index_columns).value
    
    # This will broadcast over different interventions if there are more than one
    averted = baseline - intervention
    
    return averted.reset_index()