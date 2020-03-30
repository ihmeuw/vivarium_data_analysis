import pandas as pd

INDEX_COLUMNS = ['input_draw', 'scenario']

def conditional_risk_of_ntds(
    births_with_ntd: pd.DataFrame, 
    live_births: pd.DataFrame, 
    conditioned_on=['fortification_group']: list,
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
