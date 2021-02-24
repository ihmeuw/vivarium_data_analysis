"""
This file will probably be a lightweight version of some of the things from the
LBWSG component in Vivarium Public Health:

https://github.com/ihmeuw/vivarium_public_health/blob/master/src/vivarium_public_health/risks/implementations/low_birth_weight_and_short_gestation.py
"""

import pandas as pd, numpy as np
import re
from typing import Tuple#, Dict, Iterable

# import gbd_mapping as gbd
import importlib
if importlib.util.find_spec('gbd_mapping') is not None:
    gbd_mapping = importlib.import_module('gbd_mapping')
if importlib.util.find_spec('db_queries') is not None:
    get_ids = importlib.import_module('db_queries').get_ids

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

def read_lbwsg_data_by_draw_from_gbd2017_artifact(artifact_path, measure, draw, rename=None):
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

def read_lbwsg_data_from_gbd2017_artifact(artifact_path, measure, *filter_terms, draws=None):
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
    elif source=='gbd_mapping':
        descriptions = gbd_mapping.risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
        descriptions = (pd.Series(descriptions, name='modelable_entity_name')
                        .rename_axis('lbwsg_category').reset_index())
    else:
        raise ValueError(f"Unknown source: {source}")

    cats = (pd.Series(CATEGORY_TO_MEID_GBD_2019, name='modelable_entity_id')
            .rename_axis('lbwsg_category')
            .reset_index()
            .merge(descriptions) # merge on 'modelable_entity_id' if source=='get_ids', on 'lbwsg_category' if source=='gbd_mapping'
           )
    return cats

def get_category_data(source='gbd_mapping'):
    # Get the interval descriptions (modelable entity names) for the categories
    cat_df = get_category_descriptions(source)

    # Extract the endpoints of the gestational age and birthweight intervals into 4 separate columns
    extraction_regex = r'Birth prevalence - \[(?P<ga_start>\d+), (?P<ga_end>\d+)\) wks, \[(?P<bw_start>\d+), (?P<bw_end>\d+)\) g'
    cat_df = cat_df.join(cat_df['modelable_entity_name'].str.extract(extraction_regex).astype(int,copy=False))

    def make_interval(left, right):
        return pd.Interval(left=left, right=right, closed='left')

    # Create 2 new columns of pandas.Interval objects for the gestational age and birthweight intervals
    cat_df['ga'] = np.vectorize(make_interval)(cat_df.ga_start, cat_df.ga_end)
    cat_df['bw'] = np.vectorize(make_interval)(cat_df.bw_start, cat_df.bw_end)

    # Store the width of the intervals
    cat_df['ga_width'] = cat_df['ga_end'] - cat_df['ga_start']
    cat_df['bw_width'] = cat_df['bw_end'] - cat_df['bw_start']
    return cat_df

# def merge_keep_index(df, *args, **kwargs):
#     """Performs df.merge(*args, **kwargs), but keeps the index from df in the result instead of resetting it.
#     Note that this may result in duplicate or missing index entries, depending on the results of the merge.
#     """
#     return df.reset_index().merge(*args, **kwargs).set_index(df.index.names)

class LBWSGDistribution:
    """
    Class to assign and adjust birthweights and gestational ages of a simulated population.
    """
    def __init__(self, exposure_data):
        self.exposure_dist = convert_draws_to_long_form(exposure_data, name='prevalence')
        self.exposure_dist.rename(columns={'parameter': 'lbwsg_category'}, inplace=True)
#         self.cat_df = get_category_data_by_interval()
        cat_df = get_category_data()
        cat_data_cols = ['ga_start', 'ga_end', 'bw_start', 'bw_end', 'ga_width', 'bw_width']
        self.interval_data_by_category = cat_df.set_index('lbwsg_category')[cat_data_cols]
        self.categories_by_interval = cat_df.set_index(['ga','bw'])['lbwsg_category']
    
    def get_propensity_names(self):
        """Get the names of the propensities used by this object."""
        return ['lbwsg_cat_propensity', 'ga_propensity', 'bw_propensity']

    def assign_propensities(self, pop):
        """Assigns propensities relevant to this risk exposure to the population."""
        # TODO: Fix this to use the same propensities across draws
        propensities = np.random.uniform(size=(len(pop),3))
        pop['lbwsg_cat_propensity'] = propensities[:,0] # Not actually used yet...
        pop['ga_propensity'] = propensities[:,1]
        pop['bw_propensity'] = propensities[:,2]
        
    def assign_category_from_propensity(self, pop):
        pass

    def assign_ga_bw_from_propensities_within_cat(self, pop, category_column):
        # Merge pop with cat_df to get function composition pop.index -> category -> category data
        # Unfortunately merging erases the index, so we have to manually reset it to pop.index
#         df = pop.reset_index().merge(
#             self.interval_data_by_category, left_on=category_column, right_on='lbwsg_category').set_index(pop.index.names)
#         category_data = merge_keep_index(pop, self.interval_data_by_category, left_on=category_column, right_on='lbwsg_category')
        category_data = self.get_category_data_for_population(pop, category_column)
        pop['gestational_age'] = category_data['ga_start'] + pop['ga_propensity'] * category_data['ga_width']
        pop['birthweight'] = category_data['bw_start'] + pop['bw_propensity'] * category_data['bw_width']

    def get_category_data_for_population(self, pop, category_column):
        interval_data = self.interval_data_by_category.loc[pop[category_column]].reset_index()
        interval_data.index = pop.index
        # This should pass if you call .reset_index() on interval_data before reassigning its index
        assert interval_data[category_column].equals(pop[category_column])
        return interval_data

    def assign_exposure(self, pop):
        """
        Assign birthweights and gestational ages to each simulant in the population based on
        this object's distribution.
        """
        # Based on simulant's age and sex, assign a random LBWSG category from GBD distribution
        # Index levels: location  sex  age_start  age_end   year_start  year_end  parameter
#         idxs = []
        for (draw, sex, age_start), group in pop.groupby(['draw', 'sex', 'age_start']):
            exposure_mask = (self.exposure_dist.draw==draw) & (self.exposure_dist.sex==sex) & (self.exposure_dist.age_start==age_start)
            cat_dist = self.exposure_dist.loc[exposure_mask]
#             cat_dist = self.exposure_dist.query("draw==@draw and sex==@sex and age_start==@age_start")
            pop_mask = (pop.sex == sex) & (pop.age_start == age_start) & (pop.index.get_level_values('draw') == draw)
#             pop.loc[group.index, 'lbwsg_category'] = \ # This line is really slow!!!
            pop.loc[pop_mask, 'lbwsg_category'] = \
                np.random.choice(cat_dist['lbwsg_category'], size=len(group), p=cat_dist['prevalence'])
        
#         enn_male = (pop.age_start == 0) & (pop.sex == 'Male')
#         enn_female = (pop.age_start == 0) & (pop.sex == 'Female')
#         lnn_male = (pop.age_start > 0) & (pop.age_start < 1) & (pop.sex == 'Male')
#         lnn_female = (pop.age_start > 0) & (pop.age_start < 1) & (pop.sex == 'Female')
        
#         cat_dist = self.exposure_dist.query("draw==@draw and sex==@sex and age_start==@age_start")
#         pop.loc[enn_male, 'lbwsg_category'] = \
#                 np.random.choice(cat_dist['lbwsg_category'], size=len(group), p=cat_dist['prevalence'])

        # Use propensities for ga and bw to assign a ga and bw within each category
        self.assign_ga_bw_from_propensities_within_cat(pop, 'lbwsg_category')
        
#         ga_bw_propensity = pd.DataFrame(np.random.uniform(size=(len(pop),2)),
#                                         index=pop.index, columns=['ga_propensity','bw_propensity'])
#         ga_bw_propensity['lbwsg_category'] = pop['lbwsg_category']
#         self.assign_ga_bw_from_propensities_within_cat(ga_bw_propensity, 'lbwsg_category')
#         # Now need to copy columns to pop...

    def apply_birthweight_shift(self, pop, shift, bw_col='birthweight', ga_col='gestational_age'):
        """
        Applies the specified birthweight shift to the population, and finds the new LBWSG category.
        If a simulant would be shifted out of range of the valid categories, they remain at their original
        birthweight and category (note that this strategy is only reasonable for small shifts).
        """
        index_cols = pop.index.names
        pop = pop[[ga_col, bw_col]]
        # Name the new column 'new_birthweight', but rename the existing column if that's what it's called already
        # TODO: Maybe just prepend (or append?) 'shifted' instead, so that the original column name stays the same.
        # Or allow passing a prefix or suffix.
        if bw_col == 'new_birthweigh':
            pop = pop.rename(columns={bw_col: 'old_new_birthweight'})
        new_bw_col = 'new_birthweight'
        # Throws warning: "A value is trying to be set on a copy of a slice from a DataFrame."
#         pop.loc[:, new_bw_col] = pop[bw_col] + shift 
        pop = pop.assign(**{new_bw_col: pop[bw_col] + shift})
        # TODO: Factor out some of the below code into a new function,
        # e.g. assign_category_for_ga_bw(pop, bw_col, ga_col, cat_col)
        # TODO: Also, return additional columns, e.g. the original category, and whether the category has changed
        # Categories should be computed using the proposed function above, rather than relying on what's already
        # stored in pop
        pop = pop.reset_index().set_index([ga_col, new_bw_col])
        in_bounds = self.categories_by_interval.index.get_indexer(pop.index) != -1
        pop['valid_shift'] = in_bounds
        pop.loc[in_bounds, 'new_lbwsg_category'] = self.categories_by_interval.loc[pop.loc[in_bounds].index].array
        pop = pop.reset_index(new_bw_col).set_index(bw_col, append=True)
        pop.loc[~in_bounds, 'new_lbwsg_category'] = self.categories_by_interval.loc[pop.loc[~in_bounds].index].array
#         pop.loc[~in_bounds, 'new_lbwsg_category'] = pop.loc[~in_bounds, 'lbwsg_category']
        pop = pop.reset_index()
        pop.loc[~in_bounds, new_bw_col] = pop.loc[~in_bounds, bw_col].values
        return pop.set_index(index_cols)

    def apply_birthweight_shift2(self, pop, shift, bw_col='birthweight', ga_col='gestational_age'):
#         index_cols = pop.index.names
        pop = pop[[ga_col, bw_col]].copy()
        # Assign existing categories
        pop['lbwsg_category'] = self.get_category_for_bw_ga(pop, bw_col, ga_col)
#         pop['lbwsg_category'] = self._get_categories_for_indexer(self._get_category_indexer(pop, bw_col, ga_col))
        # Prepend 'shifted', so that the original column name stays the same.
        # (Or allow passing a prefix or suffix?)
        shifted_bw_col = f'shifted_{bw_col}'
        # Apply the shift in the new column
        pop[shifted_bw_col] = pop[bw_col] + shift
        # Get integer index of category based on ga and shifted bw
        idx = self._get_category_indexer(pop, shifted_bw_col, ga_col)
        pop['valid_shift'] = in_bounds = idx != -1
        # Reset out-of-bounds birthweights back to their original values
        pop.loc[~in_bounds, shifted_bw_col] = pop.loc[~in_bounds, bw_col].array
#         pop['new_lbwsg_category'] = self._get_category_for_ga_bw(self, pop, shifted_bw_col, ga_col, cat_col)
#         pop['new_lbwsg_category'] = np.select([in_bounds, ~in_bounds], [self._get_categories_for_indexer(idx), pop['lbwsg_category']])
#         pop['new_lbwsg_category'] = np.where(in_bounds, self._get_categories_for_indexer(idx), pop['lbwsg_category'])
        pop['new_lbwsg_category'] = self.get_category_for_bw_ga(pop, shifted_bw_col, ga_col, cat_colname='new_lbwsg_category')
        pop['lbwsg_category_changed'] = pop['new_lbwsg_category'] != pop['lbwsg_category']
        return pop

    def apply_birthweight_shift3(self, pop, shift, bw_col='birthweight', ga_col='gestational_age',
                                 cat_col='lbwsg_category', shifted_col_prefix='shifted', inplace=True):
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
#         # Get integer index of category based on ga and shifted bw, to check for out-of-bounds shifts
#         idx = self._get_category_indexer(pop, shifted_bw_col, ga_col)
#         pop['valid_shift'] = in_bounds = idx != -1
#         # Reset out-of-bounds birthweights back to their original values
#         pop.loc[~in_bounds, shifted_bw_col] = pop.loc[~in_bounds, bw_col].array
#         # Assign the new category
#         self.assign_category_for_bw_ga(pop, shifted_bw_col, ga_col, shifted_cat_col, inplace=True)
#         pop[f'{cat_col}_changed'] = pop[shifted_cat_col] != pop[cat_col]
        if not inplace:
            pop.drop(columns=[ga_col, bw_col, cat_col], inplace=True)
            return pop

    def _get_category_indexer(self, pop, bw_col, ga_col):
        return self.categories_by_interval.index.get_indexer(pd.MultiIndex.from_frame(pop[[ga_col, bw_col]]))

    def _get_categories_for_indexer(self, idx):
        return self.categoriess_by_interval.array[idx]

    def get_category_for_bw_ga(self, pop, bw_col, ga_col, cat_colname='lbwsg_category'):
#         idx = self._get_category_indexer(pop, bw_col, ga_col)
#         return pd.Series(self._get_categories_for_indexer(idx), index=pop.index, name=cat_colname)
        cats = self.categories_by_interval[pd.MultiIndex.from_frame(pop[[ga_col, bw_col]])]
        return pd.Series(cats.array, index=pop.index, name=cat_colname)

    def assign_category_for_bw_ga(self, pop, bw_col, ga_col, cat_col, fill_outside_bounds=None, inplace=True):
        # Need to convert the ga and bw columns to a pandas Index to work with .get_indexer below
        ga_bw_for_pop = pd.MultiIndex.from_frame(pop[[ga_col, bw_col]])
        # Default is to raise an indexing error if bw and gw are outside bounds
        if fill_outside_bounds is None:
            # Must convert cats to a pandas array to avoid trying to match differing indexes
            cats = self.categories_by_interval.loc[ga_bw_for_pop].array
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

class LBWSGRiskEffect:
    def __init__(self, rr_data, paf_data=None):
        self.rr_data = convert_draws_to_long_form(rr_data, name='relative_risk')
        # TODO: Maybe use 'lbwsg_category' instead of 'category' throughout this module?
        self.rr_data.rename(columns={'parameter': 'lbwsg_category'}, inplace=True)
        self.paf_data = paf_data
        
    def assign_relative_risk(self, pop, cat_colname):
        # TODO: Figure out better method of dealing with category column name...
        cols_to_match = ['sex', 'age_start', 'draw', cat_colname]
        df = pop.reset_index().merge(
            self.rr_data.rename(columns={'lbwsg_category': cat_colname}), on=cols_to_match
        ).set_index(pop.index.names)
#         return df
        pop['lbwsg_relative_risk'] = df['relative_risk']

