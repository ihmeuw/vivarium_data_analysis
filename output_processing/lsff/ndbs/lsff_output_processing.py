import pandas as pd

VALUE_COLUMN = 'value'
DRAW_COLUMN  = 'input_draw'
SCENARIO_COLUMN = 'scenario'

INDEX_COLUMNS = [DRAW_COLUMN, SCENARIO_COLUMN]

def set_global_index_columns(index_columns:list)->None:
    """
    Set INDEX_COLUMNS to a custom list of columns for the Vivarium model output.
    For example, if tables for different locations have been concatenated with
    a new column called 'location', then use the following to get the correct
    behavior for the functions in this module:
    
    set_global_index_columns(['location']+lsff_output_processing.INDEX_COLUMNS)
    """
    global INDEX_COLUMNS
    INDEX_COLUMNS = index_columns

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
                  multiplier=1,
                  broadcast_cols=None,
                  dropna=False
                 ):
    """
    Compute a rate or ratio by dividing the numerator by the denominator.
    
    Parameters
    ----------
    
    numerator : DataFrame
        The numerator data for the rate or ratio.
        
    denominator : DataFrame
        The denominator data for the rate or ratio.
        
    numerator_strata : list of column names in numerator, or an empty list
        Stratification variables to include in the numerator.
        Strata in the denominator are automatically added to this list
        since the population in the numerator must always be a subset of
        the population in the denominator. (Thus an empty list [] defaults
        to denominator_strata.)
        
     denominator_strata : list of column names in denominator
         Stratification variables to include in the denominator.
         These will automatically be added to numerator_strata.
         
     multiplier : int or float, default 1
         Multiplier for the numerator, typically a power of 10,
         to adjust the units of the result. For example, if computing a ratio,
         some multipliers with corresponding units are:
         1 - proportion
         100 - percent
         1000 - per 1000
         100_000 - per hundred thousand
         
     broadcast_cols : list of column names in numerator
         Columns in the numerator over which to broadcast. E.g. 'cause' to
         compute a rate or ratio for multiple causes at once, or 'measure'
         to compute a rate or ratio for multiple measures at once (like
         deaths, ylls, and ylds).
         
     dropna : boolean, default False
         Whether to drop rows with NaN values in the result, namely
         if division by 0 occurs because of an empty stratum in the denominator.
         
     Returns
     -------
     rate_or_ratio : DataFrame
         The rate or ratio data = numerator / denominator.
    """
    index_cols = INDEX_COLUMNS
    
    if broadcast_cols is None:
        broadcast_cols = []

    # When we divide, the numerator strata must contain the denominator strata,
    # and the difference is the columns to broadcast over.
    broadcast_cols = sorted(
        set(numerator_strata) - set(denominator_strata),
        key=numerator_strata.index
    ) + broadcast_cols
    
    numerator = numerator.groupby(denominator_strata+index_cols+broadcast_cols)[VALUE_COLUMN].sum()
    denominator = denominator.groupby(denominator_strata+index_cols)[VALUE_COLUMN].sum()
    
    rate_or_ratio = multiplier * numerator / denominator
    
    # If dropna is True, drop rows where we divided by 0
    if dropna:
        rate_or_ratio.dropna(inplace=True)
    
    return rate_or_ratio.reset_index()

def averted(measure, baseline_scenario, scenario_col=None):
    """
    Compute an "averted" measure (e.g. DALYs) or measures by subtracting
    the intervention value from the baseline value.
    
    Parameters
    ----------
    
    measure : DataFrame
        DataFrame containing both the baseline and intervention data.
        
    baseline_scenario : scalar, typically str
        The name or other identifier for the baseline scenario in the
        `scenario_col` column of the `measure` DataFrame.
        
    scenario_col : str, default None
        The name of the scenario column in the `measure` DataFrame.
        Defaults to the global parameter SCENARIO_COLUMN if None is passed.
        
    Returns
    -------
    
    averted : DataFrame
        The averted measure(s) = baseline - intervention
    """
    
    scenario_col = SCENARIO_COLUMN if scenario_col is None else scenario_col
    
    # Filter to create separate dataframes for baseline and intervention
    baseline = measure[measure[scenario_col] == baseline_scenario]
    intervention = measure[measure[scenario_col] != baseline_scenario]
    
    # Columns to match when subtracting intervention from baseline
    index_columns = sorted(set(baseline.columns) - set([scenario_col, VALUE_COLUMN]),
                           key=baseline.columns.get_loc)
#     print(index_columns)
    
    # Put the scenario column in the index of intervention but not baseline.
    # When we subtract, this will broadcast over different interventions if there are more than one.
    baseline = baseline.set_index(index_columns)
    intervention = intervention.set_index(index_columns+[scenario_col])
    
    # Get the averted values
    averted = baseline[VALUE_COLUMN] - intervention[VALUE_COLUMN]
    
    # Insert a column after the scenario column to record what the baseline scenario was
    averted = averted.reset_index()
    averted.insert(averted.columns.get_loc(scenario_col)+1, 'relative_to', baseline_scenario)
    
    return averted

def describe(data, **describe_kwargs):
    """Wrapper function for .describe() with `data` grouped by everything except draw and value."""
    groupby_cols = [col for col in data.columns if col not in [DRAW_COLUMN, VALUE_COLUMN]]
    return data.groupby(groupby_cols)[VALUE_COLUMN].describe(**describe_kwargs)
