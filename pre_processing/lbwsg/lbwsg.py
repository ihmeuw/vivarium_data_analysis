"""
This file will probably be a lightweight version of some of the things from the
LBWSG component in Vivarium Public Health:

https://github.com/ihmeuw/vivarium_public_health/blob/master/src/vivarium_public_health/risks/implementations/low_birth_weight_and_short_gestation.py
"""

import pandas as pd, numpy as np
import re
from typing import Tuple#, Dict, Iterable

import demography
# from demography import get_age_group_data, get_sex_id_map

# import gbd_mapping as gbd
import importlib
if importlib.util.find_spec('gbd_mapping') is not None:
    gbd_mapping = importlib.import_module('gbd_mapping')
if importlib.util.find_spec('db_queries') is not None:
    get_ids = importlib.import_module('db_queries').get_ids
if importlib.util.find_spec('get_draws') is not None:
    get_draws = importlib.import_module('get_draws').api.get_draws

LBWSG_REI_ID = 339 # GBD's "risk/etiology/impairment" id for Low birthweight and short gestation
GBD_2019_ROUND_ID = 6

# The support of the LBWSG distribution is nonconvex, but adding this one category makes it convex,
# which makes life easier when shifting the birthweights or gestational ages.
# I think the category number was just arbitrarily chosen from those that weren't already taken.
# Note: In GBD 2019, this category is `cat124`, meid=20224.
MISSING_CATEGORY_GBD_2017 = {'cat212': 'Birth prevalence - [37, 38) wks, [1000, 1500) g'}

# Category to indicate that birthweight and gestational age are outside the domain of the risk distribution
OUTSIDE_BOUNDS_CATEGORY = 'cat_outside_bounds'

# The dictionary below was created with the following code:
# CATEGORY_TO_MEID_GBD_2019 = (
#     lbwsg_exposure_nigeria_birth_male
#      .dropna()
#      .set_index('parameter')
#      ['modelable_entity_id']
#      .astype(int)
#      .sort_index(key=lambda s: s.str.strip('cat').astype(int))
#      .to_dict()
#     )
# where `lbwsg_exposure_nigeria_birth_male` was exposure data from `get_draws`.

# TODO: Perhaps store the category descriptions here as well
# Modelable entity IDs for LBWSG categories
CATEGORY_TO_MEID_GBD_2019 = {
 'cat2': 10755,
 'cat8': 10761,
 'cat10': 10763,
 'cat11': 10764,
 'cat14': 10767,
 'cat15': 10768,
 'cat17': 10770,
 'cat19': 10772,
 'cat20': 10773,
 'cat21': 10774,
 'cat22': 10775,
 'cat23': 10776,
 'cat24': 10777,
 'cat25': 10778,
 'cat26': 10779,
 'cat27': 10780,
 'cat28': 10781,
 'cat29': 10782,
 'cat30': 10783,
 'cat31': 10784,
 'cat32': 10785,
 'cat33': 10786,
 'cat34': 10787,
 'cat35': 10788,
 'cat36': 10789,
 'cat37': 10790,
 'cat38': 10791,
 'cat39': 10792,
 'cat40': 10793,
 'cat41': 10794,
 'cat42': 10795,
 'cat43': 10796,
 'cat44': 10797,
 'cat45': 10798,
 'cat46': 10799,
 'cat47': 10800,
 'cat48': 10801,
 'cat49': 10802,
 'cat50': 10803,
 'cat51': 10804,
 'cat52': 10805,
 'cat53': 10806,
 'cat54': 10807,
 'cat55': 10808,
 'cat56': 10809,
 'cat80': 20203,
 'cat81': 20204,
 'cat82': 20205,
 'cat88': 20209,
 'cat89': 20210,
 'cat90': 20211,
 'cat95': 20214,
 'cat96': 20215,
 'cat106': 20221,
 'cat116': 20227,
 'cat117': 20228,
 'cat123': 20232,
 'cat124': 20224
}

####################################################
# READING AND PROCESSING GBD DATA FOR LBWSG #
####################################################

def read_lbwsg_data_by_draw_from_gbd_2017_artifact(artifact_path, measure, draw, rename=None):
    """
    Reads one draw of LBWSG data from an artifact.
    
    measure should be one of:
    'exposure'
    'relative_risk'
    'population_attributable_fraction'
    rename should be a string or None (default)
    """
    key = f'risk_factor/low_birth_weight_and_short_gestation/{measure}'
    with pd.HDFStore(artifact_path, mode='r') as store:
        index = store.get(f'{key}/index')
        draw = store.get(f'{key}/draw_{draw}')
    if rename is not None:
        draw.rename(rename)
    data = pd.concat([index, draw], axis=1)
    return data

def read_lbwsg_data_from_gbd_2017_artifact(artifact_path, measure, *filter_terms, draws=None):
    """
    Reads multiple draws from the artifact.
    """
    key = f'risk_factor/low_birth_weight_and_short_gestation/{measure}'
    query_string = ' and '.join(filter_terms)
    # NOTE: If draws is a numpy array, the line `if draws=='all':` threw a warning:
    #  "FutureWarning: elementwise comparison failed; returning scalar instead,
    #   but in the future will perform elementwise comparison"
    # So I changed default from 'all' to None.
    if draws is None:
        draws = range(1000)
    
    with pd.HDFStore(artifact_path, mode='r') as store:
        index_cols = store.get(f'{key}/index')
        if query_string != '':
            index_cols = index_cols.query(query_string)
        draw_data_dfs = [index_cols]
        for draw in draws:
            draw_data = store.get(f'{key}/draw_{draw}') # draw_data is a pd.Series
            draw_data = draw_data[index_cols.index] # filter to query on index columns
            draw_data_dfs.append(draw_data)

#     print(index_cols.columns)
    return pd.concat(draw_data_dfs, axis=1, copy=False).set_index(index_cols.columns.to_list())

def pull_lbwsg_exposure_from_gbd_2019(location_ids, year_ids=2019, save_to_hdf=None, hdf_key=None):
    """Calls get_draws to pull LBWSG exposure data from GBD 2019."""
    # Make sure type(location_ids) is list
    location_ids = [location_ids] if isinstance(location_ids, int) else list(location_ids)
    lbwsg_exposure = get_draws(
        'rei_id',
        gbd_id=LBWSG_REI_ID,
        source='exposure',
        location_id=location_ids,
        year_id=year_ids,
#         sex_id=sex_ids, # Default is [1,2], but 3 also exists
        gbd_round_id=GBD_2019_ROUND_ID,
        status='best',
        decomp_step='step4',
    )
    if save_to_hdf is not None:
        if hdf_key is None:
            description = f"location_ids_{'_'.join(location_ids)}"
            hdf_key = f"/gbd_2019/exposure/{description}"
        lbwsg_exposure.to_hdf(save_to_hdf, hdf_key)
    return lbwsg_exposure

def pull_lbwsg_relative_risks_from_gbd_2019(cause_ids=None, year_ids=2019, save_to_hdf=None, hdf_key=None):
    """Calls get_draws to pull LBWSG relative risk data from GBD 2019."""
    global_location_id = 1 # RR's are the same for all locations - they all propagate up to location_id==1
    if cause_ids is None:
        cause_ids = []
    elif isinstance(cause_ids, int):
        cause_ids = [cause_ids]
    lbwsg_rr = get_draws(
            gbd_id_type=['rei_id']+['cause_id']*len(cause_ids), # Types must match gbd_id's
            gbd_id=[LBWSG_REI_ID, *cause_ids],
            source='rr',
            location_id=global_location_id,
            year_id=year_ids,
            gbd_round_id=GBD_2019_ROUND_ID,
            status='best',
            decomp_step='step4',
        )
    if save_to_hdf is not None:
        if hdf_key is None:
            description = 'all' #if len(cause_ids)==0 else f"cause_ids_{'_'.join(cause_ids)}"
            hdf_key = f"/gbd_2019/relative_risk/{description}"
        lbwsg_rr.to_hdf(save_to_hdf, hdf_key)
    return lbwsg_rr

def rescale_prevalence(exposure):
    """Rescales prevalences to add to 1 in LBWSG exposure data pulled from GBD 2019 by get_draws."""
    # Drop residual 'cat125' parameter with meid==NaN, and convert meid col from float to int
    exposure = exposure.dropna().astype({'modelable_entity_id': int})
    # Define some categories of columns
    draw_cols = exposure.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    category_cols = ['modelable_entity_id', 'parameter']
    index_cols = exposure.columns.difference(draw_cols)
    sum_index = index_cols.difference(category_cols)
    # Add prevalences over categories (indexed by meid and/or parameter) to get denominator for rescaling
    prevalence_sum = exposure.groupby(sum_index.to_list())[draw_cols].sum()
    # Divide prevalences by total to rescale them to add to 1, and reset index to put df back in original form
    exposure = exposure.set_index(index_cols.to_list()) / prevalence_sum
    exposure.reset_index(inplace=True)
    return exposure

def preprocess_gbd_data(df, draws=None):
    """df can be exposure or rr data?
    Note that location_id for rr data will always be 1 (Global), so it won't
    match location_id for exposure data.
    """
    # Note: walrus operator := requires Python 3.8 or higher
    if 'cause_id' in df and len(cause_ids:=df['cause_id'].unique())>1:
        # If df is RR data, filter to a single cause id - all affected causes have the same RR's
        df = df.loc[df['cause_id']==cause_ids[0]]
    elif 'measure_id' in df and list(df['measure_id'].unique()) == [5]: # measure_id 5 is Prevalence
        # If df is exposure data, rescale prevalence
        # TODO: Perhaps check if df contains NaN or sum of prevalence is not 1
        df = rescale_prevalence(df)

    sex_id_to_sex = demography.get_sex_id_to_sex_map()
    df = df.join(sex_id_to_sex, on='sex_id').rename(columns={'parameter': 'lbwsg_category'})
    index_cols = ['location_id', 'year_id', 'sex', 'age_group_id', 'lbwsg_category']
    if draws is None:
        draw_cols = df.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    else:
        draw_cols = [f"draw_{i}" for i in draws]
    return df.set_index(index_cols)[draw_cols]

def preprocess_artifact_data(df):
    pass

def convert_draws_to_long_form(data, name='value', copy=True):
    """
    Converts GBD data stored with one column per draw to "long form" with a 'draw' column wihch specifies the draw.
    """
    if copy:
        data = data.copy()
#     draw_cols = data.filter(like='draw').columns
#     index_columns = data.columns.difference(draw_cols)
#     data.set_index(index_columns.to_list())
    # Check whether columns are named draw_### in case we already took a mean over all draws
    if len(data.filter(regex=r'^draw_\d{1,3}$').columns) == data.shape[1]:
        data.columns = data.columns.str.replace('draw_', '').astype(int)
    data.columns.rename('draw', inplace=True)
    data = data.stack()
    data.rename(name, inplace=True)
    return data.reset_index()

##########################################################
# DATA ABOUT LBWSG RISK CATEGORIES #
##########################################################

def get_intervals_from_name(name: str) -> Tuple[pd.Interval, pd.Interval]:
    """Converts a LBWSG category name to a pair of intervals.

    The first interval corresponds to gestational age in weeks, the
    second to birth weight in grams.
    """
    numbers_only = [int(n) for n in re.findall(r'\d+', name)] # The regex \d+ matches 1 or more digits
    return (pd.Interval(numbers_only[0], numbers_only[1], closed='left'),
            pd.Interval(numbers_only[2], numbers_only[3], closed='left'))

def get_category_descriptions(source='gbd_mapping'):
    # The "description" is the modelable entity name for the category
    if source=='get_ids':
        descriptions = get_ids('modelable_entity')
    else:
        if source=='gbd_mapping':
            descriptions = gbd_mapping.risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
        # Assume source is a dictionary of categories to descriptions (i.e. modelable entity names)
        descriptions = (pd.Series(descriptions, name='modelable_entity_name')
                        .rename_axis('lbwsg_category').reset_index())

    cats = (pd.Series(CATEGORY_TO_MEID_GBD_2019, name='modelable_entity_id')
            .rename_axis('lbwsg_category').reset_index()
            .merge(descriptions) # merge on 'modelable_entity_id' if source=='get_ids', on 'lbwsg_category' if source=='gbd_mapping'
           )
    return cats

def get_category_data(source='gbd_mapping'):
    # Get the interval descriptions (modelable entity names) for the categories
    cat_df = get_category_descriptions(source)

    # Extract the endpoints of the gestational age and birthweight intervals into 4 separate columns
    extraction_regex = r'Birth prevalence - \[(?P<ga_start>\d+), (?P<ga_end>\d+)\) wks, \[(?P<bw_start>\d+), (?P<bw_end>\d+)\) g'
    cat_df = cat_df.join(cat_df['modelable_entity_name'].str.extract(extraction_regex).astype(int,copy=False))

    @np.vectorize
    def interval_width_midpoint(left, right):
        interval = pd.Interval(left=left, right=right, closed='left')
        return interval, interval.length, interval.mid

    # Create 2 new columns of pandas.Interval objects for the gestational age and birthweight intervals,
    # and 2 more new columns for the interval widths
    cat_df['ga'], cat_df['ga_width'], cat_df['ga_midpoint'] = interval_width_midpoint(cat_df.ga_start, cat_df.ga_end)
    cat_df['bw'], cat_df['bw_width'], cat_df['bw_midpoint'] = interval_width_midpoint(cat_df.bw_start, cat_df.bw_end)
    return cat_df

##########################################################
# CLASS FOR LBWSG RISK DISTRIBUTION #
##########################################################

# TODO: Move this function to prob_utils.py, and come up with a better name for it
def sample_from_propensity(propensity, categories, category_cdf):
    """Sample categories using the propensities.
    propensity is a number between 0 and 1
    categories is a list of categories
    category_cdf is a mapping of categories to cumulative probabilities.
    """
    condlist = [propensity <= category_cdf[cat] for cat in categories]
    return np.select(condlist, choicelist=categories)

def sample_from_propensity_as_arrays(propensity, categories, category_cdf):
    """Sample categories using the propensities.
    propensity is a number (or list/array of numbers) between 0 and 1
    categories is a list of categories
    category_cdf is a 2d array of shape (len(propensity), len(categories)).
    """
    category_index = (np.asarray(propensity).reshape((-1,1)) > np.asarray(category_cdf)).sum(axis=1)
    return np.asarray(categories)[category_index]

class LBWSGDistribution:
    """
    Class to assign and adjust birthweights and gestational ages of a simulated population.
    """
    def __init__(self, exposure_data):
        # TODO: Should we NOT convert draws to long form since we want them wide for the CDF?
        self.exposure_dist = convert_draws_to_long_form(exposure_data, name='prevalence')
#         self.exposure_dist.rename(columns={'parameter': 'lbwsg_category'}, inplace=True)
#         self.cat_df = get_category_data_by_interval()
        cat_df = get_category_data()
        cat_data_cols = ['ga_start', 'ga_end', 'bw_start', 'bw_end', 'ga_width', 'bw_width']
        self.interval_data_by_category = cat_df.set_index('lbwsg_category')[cat_data_cols]
        self.categories_by_interval = cat_df.set_index(['ga','bw'])['lbwsg_category']
#         self.age_to_id = get_age_to_id_map(self.exposure_dist['age_group_id'].unique())

#     def get_propensity_names(self):
#         """Get the names of the propensities used by this object."""
#         return ['lbwsg_category_propensity', 'ga_propensity', 'bw_propensity']

    def assign_propensities(self, pop):
        """Assigns propensities relevant to this risk exposure to the population."""
        # TODO: Fix this to use the same propensities across draws
        propensities = np.random.uniform(size=(len(pop),3))
        pop['lbwsg_category_propensity'] = propensities[:,0] # Not actually used yet...
        pop['ga_propensity'] = propensities[:,1]
        pop['bw_propensity'] = propensities[:,2]

#     def get_age_groups_for_population(self, pop):
#         age_groups = self.age_to_id.reindex(pop['age'])
#         age_groups.index = pop.index
#         return age_groups

    def get_exposure_cdf(self):
        # TODO: Perhaps move some of this to the preprocessing step, e.g. setting the index to
        # everything except the prevalence column, and unstacking (or never stacking in the first place)
        index_cols = self.exposure_dist.columns.difference(['prevalence']).to_list()
        exposure_cdf = self.exposure_dist.set_index(index_cols).unstack('lbwsg_category').cumsum(axis=1)
       # QUESTION: Is there any situation where we will need 'location_id' or 'year_id'?
        exposure_cdf = exposure_cdf.droplevel(['location_id','year_id']).droplevel(0, axis=1)
        return exposure_cdf

    def get_exposure_cdf_for_population(self, pop):
        exposure_cdf = self.get_exposure_cdf()
        extra_index_cols = ['age_group_id', 'sex']
        pop_exposure_cdf = (
            pop[extra_index_cols]
            .set_index(extra_index_cols, append=True)
            .join(exposure_cdf)
            .droplevel(extra_index_cols)
            .reindex(pop.index)
        )
        return pop_exposure_cdf
    
    def get_exposure_cdf_for_population2(self, pop):
        # TODO: Make this the version that actually gets used
        exposure_cdf = self.get_exposure_cdf()
#         index_cols = exposure_cdf.index.names # Should be ['age_group_id', 'draw', 'sex'], but order not guaranteed
        extra_index_cols = ['age_group_id', 'sex']
        pop_exposure_cdf = pop[extra_index_cols].join(exposure_cdf, on=exposure_cdf.index.names)
        pop_exposure_cdf.drop(columns=extra_index_cols, inplace=True)
        pop_exposure_cdf.rename_axis(columns=exposure_cdf.columns.name, inplace=True, copy=False)
        return pop_exposure_cdf

    def get_exposure_cdf_for_population1(self, pop):
        exposure_cdf = self.get_exposure_cdf()
        age_sex_draw = pop[['age_group_id', 'sex']].reset_index('draw')[['age_group_id', 'draw', 'sex']]
        pop_exposure_cdf = exposure_cdf.reindex(age_sex_draw)
        pop_exposure_cdf.index = pop.index
        return pop_exposure_cdf

    def assign_category_from_propensity(self, pop):
        # TODO: allow specifying the category column name, and add an option to not modify pop in place
        """Assigns LBWSG categories to the population based on simulant propensities."""
        pop_exposure_cdf = self.get_exposure_cdf_for_population(pop)
        lbwsg_cat = sample_from_propensity(pop['lbwsg_category_propensity'], pop_exposure_cdf.columns, pop_exposure_cdf)
        pop['lbwsg_category'] = lbwsg_cat

    def assign_exposure(self, pop):
        """
        Assign birthweights and gestational ages to each simulant in the population based on
        this object's distribution.
        """
        # Based on simulant's age and sex, assign a random LBWSG category from GBD distribution
        # Index levels: location  sex  age_start  age_end   year_start  year_end  parameter
#         idxs = []
#         for (draw, sex, age_group_id), group in pop.groupby(['draw', 'sex', 'age_group_id'], observed=True):
#             exposure_mask = (self.exposure_dist.draw==draw) & (self.exposure_dist.sex==sex) & (self.exposure_dist.age_group_id==age_group_id)
#             cat_dist = self.exposure_dist.loc[exposure_mask]
# #             cat_dist = self.exposure_dist.query("draw==@draw and sex==@sex and age_start==@age_start")
#             pop_mask = (pop.sex == sex) & (pop.age_group_id == age_group_id) & (pop.index.get_level_values('draw') == draw)
# #             pop.loc[group.index, 'lbwsg_category'] = \ # This line is really slow!!!
#             pop.loc[pop_mask, 'lbwsg_category'] = \
#                 np.random.choice(cat_dist['lbwsg_category'], size=len(group), p=cat_dist['prevalence'])

        self.assign_category_from_propensity(pop)
        # Use propensities for ga and bw to assign a ga and bw within each category
        self.assign_ga_bw_from_propensities_within_cat(pop, 'lbwsg_category')

    def assign_ga_bw_from_propensities_within_cat(self, pop, category_column):
        """Assigns birthweights and gestational ages using propensities.
        If the propensities are uniformly distributed in [0,1], the birthweights and gestational ages
        will be uniformly distributed within each LBWSG category.
        """
        category_data = self.get_category_data_for_population(pop, category_column)
        pop['gestational_age'] = category_data['ga_start'] + pop['ga_propensity'] * category_data['ga_width']
        pop['birthweight'] = category_data['bw_start'] + pop['bw_propensity'] * category_data['bw_width']

    def get_category_data_for_population(self, pop, category_column):
        """Returns a dataframe indexed by pop.index, where each row contains data about
        the corresponding simulant's LBWSG category.
        """
        interval_data = self.interval_data_by_category.loc[pop[category_column]]
        # This verifies that the correct population indexes will be assigned to the category data below
        assert pd.Series(interval_data.index, index=pop.index).equals(pop[category_column]), \
            'Categories misaligned with population index!'
        interval_data.index = pop.index
        return interval_data
    
    def get_category_data_for_population1(self, pop, category_column):
        interval_data = pop.join(self.interval_data_by_category, on=category_column)
        return interval_data

    def apply_birthweight_shift(self, pop, shift, bw_col='birthweight', ga_col='gestational_age',
                                 cat_col='lbwsg_category', shifted_col_prefix='shifted', inplace=True):
        """Applies the specified birthweight shift to the population. Modifies the population table in place
        unless inplace=False, in which case a new population table is returned.
        """
        if not inplace:
            pop = pop[[ga_col, bw_col, cat_col]].copy()
        shifted_bw_col = f'{shifted_col_prefix}_{bw_col}'
        shifted_cat_col = f'{shifted_col_prefix}_{cat_col}'
        # Apply the shift in the new birthweight column
        pop[shifted_bw_col] = pop[bw_col] + shift
        # Assign the new category and mark where (ga,bw) is out of bounds
        self.assign_category_for_bw_ga(pop, shifted_bw_col, ga_col, shifted_cat_col,
                                       fill_outside_bounds=OUTSIDE_BOUNDS_CATEGORY, inplace=True)
        pop['valid_shift'] = pop[shifted_cat_col] != OUTSIDE_BOUNDS_CATEGORY
        # Reset out-of-bounds birthweights and categories back to their original values
        pop.loc[~pop['valid_shift'], shifted_bw_col] = pop.loc[~pop['valid_shift'], bw_col].array
        pop.loc[~pop['valid_shift'], shifted_cat_col] = self.assign_category_for_bw_ga(
            pop.loc[~pop['valid_shift']], shifted_bw_col, ga_col, shifted_cat_col, inplace=False)
        pop[f'{cat_col}_changed'] = pop[shifted_cat_col] != pop[cat_col]
        if not inplace:
            pop.drop(columns=[ga_col, bw_col, cat_col], inplace=True)
            return pop

    def assign_category_for_bw_ga(self, pop, bw_col, ga_col, cat_col, fill_outside_bounds=None, inplace=True):
        """Assigns the correct LBWSG category to each simulant in the population,
        given birthweights and gestational ages.
        Modifies the population table in place if inplace=True (default), otherwise returns a pandas Series
        containing each simulant's LBWSG category, indexed by pop.index.
        If `fill_outside_bounds` is None (default), an indexing error (KeyError) will be raised (by IntervalIndex) if any
        (birthweight, gestational age) pair does not correspond to a valid LBWSG category.
        If `fill_outside_bounds` is not None, its value will be used to fill the category value for invalid
        (birthweight, gestational age) pairs.
        """
        # Need to convert the ga and bw columns to a pandas Index to work with .get_indexer below
        ga_bw_for_pop = pd.MultiIndex.from_frame(pop[[ga_col, bw_col]])
        # Default is to raise an indexing error if bw and gw are outside bounds
        if fill_outside_bounds is None:
            # Must convert cats to a pandas array to avoid trying to match differing indexes
            cats = self.categories_by_interval.loc[ga_bw_for_pop].array
            # TODO: See if doing the following instead will result in the same behavior as below for invalid bw, ga:
            # cats = self.categories_by_interval.reindex(ga_bw_for_pop).array
        # Otherwise, the category for out-of-bounds (ga,bw) pairs will be assigned the value `fill_outside_bounds`
        else:
            # Get integer index of category, to check for out-of-bounds (ga,bw) pairs (iidx==-1 if (ga,bw) not in index)
            iidx = self.categories_by_interval.index.get_indexer(ga_bw_for_pop)
            cats = np.where(iidx != -1, self.categories_by_interval.iloc[iidx], fill_outside_bounds)
        # We have to cast cats to a pandas array to avoid trying to match differing indexes
        if inplace:
            pop[cat_col] = cats
        else:
            return pd.Series(cats, index=pop.index, name=cat_col)

##########################################################
# CLASSES FOR LBWSG RISK EFFECTS (RELATIVE RISKS) #
##########################################################

class LBWSGRiskEffect:
    def __init__(self, rr_data, paf_data=None):
        self.rr_data = convert_draws_to_long_form(rr_data, name='relative_risk')
#         self.rr_data.rename(columns={'parameter': 'lbwsg_category'}, inplace=True)
        self.paf_data = paf_data
        
    def assign_relative_risk(self, pop, cat_colname):
        # TODO: Fix tthis because it doesn't work with GBD data - see version below
        # TODO: Figure out better method of dealing with category column name...
        cols_to_match = ['sex', 'age_start', 'draw', cat_colname]
        df = pop.reset_index().merge(
            self.rr_data.rename(columns={'lbwsg_category': cat_colname}), on=cols_to_match
        ).set_index(pop.index.names)
#         return df
        pop['lbwsg_relative_risk'] = df['relative_risk']

    def get_relative_risks_for_populatation(self, pop, cat_colname):
        # TODO: Maybe this should just be called `assign_relative_risk`, with an option for inplace or not
        rr_map = self.get_rr_mapper()
        # Rename the category column so it matches that in the RR data
        extra_index_cols = ['age_group_id', 'sex', 'lbwsg_category']
        pop = pop.rename(columns={cat_colname: 'lbwsg_category'})[extra_index_cols]
        pop_rrs = pop.join(rr_map, on=rr_map.index.names).drop(columns=extra_index_cols)
        return pop_rrs

    def get_rr_mapper(self):
        # TODO: Set the correct index in preprocessing instead of here, and perhaps just have this function
        # drop the location and year levels. Also, rename this function to somthing better.
        index_cols = self.rr_data.columns.difference(['relative_risk']).to_list()
        rr_map = self.rr_data.set_index(index_cols)
       # QUESTION: Is there any situation where we will need 'location_id' or 'year_id'?
        rr_map = rr_map.droplevel(['location_id','year_id'])#.droplevel(0, axis=1)
        return rr_map

