"""
This file will probably be a lightweight version of some of the things from the
LBWSG component in Vivarium Public Health:

https://github.com/ihmeuw/vivarium_public_health/blob/master/src/vivarium_public_health/risks/implementations/low_birth_weight_and_short_gestation.py
"""

import pandas as pd, numpy as np
import re
from typing import Tuple#, Dict, Iterable
import gbd_mapping as gbd

# The support of the LBWSG distribution is nonconvex, but adding this one category makes it convex,
# which makes life easier when shifting the birthweights or gestational ages.
# I think the category number was just arbitrarily chosen from those that weren't already taken.
MISSING_CATEGORY = {'cat212': 'Birth prevalence - [37, 38) wks, [1000, 1500) g'}

def read_lbwsg_data_by_draw(artifact_path, measure, draw, rename=None):
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

def read_lbwsg_data1(artifact_path, measure, *filter_terms, draws='all'):
    query_string = ' and '.join(filter_terms)
    if draws=='all':
        draws = range(1000)
    draw_cols = [f'draw_{draw}' for draw in draws]
    draw_data_dfs = []
    
    for draw in draws:
        draw_data = read_lbwsg_data_by_draw(artifact_path, measure, draw)
        if query_string != '':
            draw_data = draw_data.query(query_string)
        index_cols = draw_data.columns.difference(draw_cols)
        draw_data = draw_data.set_index(index_cols.to_list())
#         data = pd.concat([data, draw_data], axis=1)
        draw_data_dfs.append(draw_data)
        
#     return data
    return pd.concat(draw_data_dfs, axis=1, copy=False)

def read_lbwsg_data(artifact_path, measure, *filter_terms, draws='all'):
    """
    Reads multiple draws from the artifact.
    """
    key = f'risk_factor/low_birth_weight_and_short_gestation/{measure}'
    query_string = ' and '.join(filter_terms)
    if draws=='all':
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

def get_lbwsg_categories_by_interval(include_missing=True):
    """
    Return a pandas Series indexed by (gestational age interval, birth weight interval),
    mapping to the corresponding LBWSG category.
    """
    category_dict = gbd.risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
    if include_missing:
        category_dict.update(MISSING_CATEGORY)
    cats = (pd.DataFrame.from_dict(category_dict, orient='index')
            .reset_index()
            .rename(columns={'index': 'cat', 0: 'name'}))
    idx = pd.MultiIndex.from_tuples(cats.name.apply(get_intervals_from_name),
                                    names=['gestation_time', 'birth_weight'])
    cats = cats['cat']
    cats.index = idx
    return cats

def get_ga_bw_by_category(include_missing=True):
    """
    Returns a dataframe indexed by LBWSG category, with four columns describing the intervals for that category:
    ga_start, ga_end, bw_start, bw_end
    """
    cats = gbd.risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
    if include_missing:
        cats.update(MISSING_CATEGORY)
    cats = pd.Series(cats)
    # Example string to extract from: 'Birth prevalence - [0, 24) wks, [0, 500) g'
    # extraction_regex = r'.*\[(?P<ga_start>\d+), (?P<ga_end>\d+).*\[(?P<bw_start>\d+), (?P<bw_end>\d+).*' # this also works
    extraction_regex = r'Birth prevalence - \[(?P<ga_start>\d+), (?P<ga_end>\d+)\) wks, \[(?P<bw_start>\d+), (?P<bw_end>\d+)\) g'
    cats = cats.str.extract(extraction_regex).astype(int,copy=False)
    cats.index.rename('category', inplace=True)
    return cats

def get_category_data_by_interval(include_missing=False):
    """
    Returns a dataframe indexed by the bw and ga intervals, with columns for data about the
    corresponding LBWSG category.
    """
    cat_df = get_ga_bw_by_category(include_missing).reset_index()
    
    def make_interval(left,right):
        return pd.Interval(left=left, right=right, closed='left')
    
    cat_df['ga'] = np.vectorize(make_interval)(cat_df.ga_start, cat_df.ga_end)
    cat_df['bw'] = np.vectorize(make_interval)(cat_df.bw_start, cat_df.bw_end)
    
#     cat_df['ga_width'] = cat_df['ga_end'] - cat_df['ga_start']
    cat_df['ga_width'] = cat_df.ga.apply(lambda interval: interval.length)
    cat_df['bw_width'] = cat_df['bw_end'] - cat_df['bw_start']
    
    return cat_df.set_index(['ga','bw'])

class LBWSGDistribution:
    """
    Class to assign and adjust birthweights and gestational ages of a simulated population.
    """
    def __init__(self, exposure_data):
        self.exposure_dist = convert_draws_to_long_form(exposure_data, name='prevalence')
        self.exposure_dist.rename(columns={'parameter': 'category'}, inplace=True)
        self.cat_df = get_category_data_by_interval()
    
    def cat_to_ga_bw(self, cat, p1, p2):
        """Map from category `cat` to bivariate continuous values of birth weight and gestational age.

        Example usage:

        cat_df = get_ga_bw_by_category()
        cat = np.random.choice(cat_df.index, size=None)
        cat, cat_to_ga_bw(cat, np.random.rand(), np.random.rand())

        Parameters
        ----------
        cat : str or list-like of strings, one of the 58 values in index of cat_df
        p1 : float or list-like of floats, propensity between 0 and 1
        p2 : float or list-like of floats, propensity between 0 and 1

        Results
        -------
        Returns tuple of (gestational age, birth weight)
        """

        ga = self.cat_df.loc[cat, 'ga_start'] + p1 * self.cat_df.loc[cat, 'ga_width']
        bw = self.cat_df.loc[cat, 'bw_start'] + p2 * self.cat_df.loc[cat, 'bw_width']

        return ga, bw
    
    def assign_ga_bw_from_propensities_within_cat(self, pop, category_column):
        # Merge pop with cat_df to get function composition pop.index -> category -> category data
        # Unfortunately merging erases the index, so we have to manually reset it to pop.index
        df = pop.reset_index().merge(self.cat_df, left_on=category_column, right_on='category').set_index(pop.index.names)
        pop['gestational_age'] = df['ga_start'] + pop['ga_propensity'] * df['ga_width']
        pop['birthweight'] = df['bw_start'] + pop['bw_propensity'] * df['bw_width']
    
    def ga_bw_to_cat(self, ga, bw):
        """Map from (birth weight, gestational age) to category name.

        Example usage:

        ga_bw_to_cat(31., 3298.)

        Parameters
        ----------
        ga : float (gestational age)
        bw : float (birth weight)

        Results
        -------
        Returns cat for (bw,ga) pair
        """

        t = self.cat_df.query('@bw >= bw_start and @bw < bw_end and @ga >= ga_start and @ga < ga_end')
        assert len(t) >= 1
        return t.index[0]

    def assign_exposure(self, pop):
        """
        Assign birthweights and gestational ages to each simulant in the population based on
        this object's distribution.
        """
        # Based on simulant's age and sex, assign a random LBWSG category from GBD distribution
        # Index levels: location  sex  age_start  age_end   year_start  year_end  parameter
        for (draw, sex, age_start), group in pop.groupby(['draw', 'sex', 'age_start']):
            cat_dist = self.exposure_dist.query("draw==@draw and sex==@sex and age_start==@age_start")
            pop.loc[group.index, 'lbwsg_cat'] = \
                np.random.choice(cat_dist['category'], size=len(group), p=cat_dist['prevalence'])
        
        # Assign propensities for ga and bw, and use them to assign a ga and bw within each category
        ga_bw_propensities = np.random.uniform(size=(len(pop),2))
        pop['ga_propensity'] = ga_bw_propensities[:,0]
        pop['bw_propensity'] = ga_bw_propensities[:,1]
        self.assign_ga_bw_from_propensities_within_cat(pop, 'lbwsg_cat')
        
#         ga_bw_propensity = pd.DataFrame(np.random.uniform(size=(len(pop),2)),
#                                         index=pop.index, columns=['ga_propensity','bw_propensity'])
#         ga_bw_propensity['category'] = pop['lbwsg_cat']
#         self.assign_ga_bw_from_propensities_within_cat(ga_bw_propensity, 'category')
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
        # Categories should be compute using the proposed function above, rather than relying on what's already
        # stored in pop
        pop = pop.reset_index().set_index([ga_col, new_bw_col])
        in_bounds = self.cat_df.index.get_indexer(pop.index) != -1
        pop['valid_shift'] = in_bounds
        pop.loc[in_bounds, 'new_lbwsg_cat'] = self.cat_df.loc[pop.loc[in_bounds].index, 'category'].values
        pop = pop.reset_index(new_bw_col).set_index(bw_col, append=True)
        pop.loc[~in_bounds, 'new_lbwsg_cat'] = self.cat_df.loc[pop.loc[~in_bounds].index, 'category'].values
#         pop.loc[~in_bounds, 'new_lbwsg_cat'] = pop.loc[~in_bounds, 'lbwsg_cat']
        pop = pop.reset_index()
        pop.loc[~in_bounds, new_bw_col] = pop.loc[~in_bounds, bw_col].values
        return pop.set_index(index_cols)
        
        
        

class LBWSGRiskEffect:
    def __init__(self, rr_data, paf_data=None):
        self.rr_data = convert_draws_to_long_form(rr_data, name='relative_risk')
        # TODO: Maybe use 'lbwsg_cat' instead of 'category' throughout this module?
        self.rr_data.rename(columns={'parameter': 'lbwsg_cat'}, inplace=True)
        self.paf_data = paf_data
        
    def assign_relative_risk(self, pop, cat_colname):
        # TODO: Figure out better method of dealing with category column name...
        cols_to_match = ['sex', 'age_start', 'draw', cat_colname]
        df = pop.reset_index().merge(
            self.rr_data.rename(columns={'lbwsg_cat': cat_colname}), on=cols_to_match
        ).set_index(pop.index.names)
#         return df
        pop['lbwsg_relative_risk'] = df['relative_risk']