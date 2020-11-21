"""
This file will probably be a lightweight version of some of the things from the
LBWSG component in Vivarium Public Health:

https://github.com/ihmeuw/vivarium_public_health/blob/master/src/vivarium_public_health/risks/implementations/low_birth_weight_and_short_gestation.py
"""

import pandas as pd
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

    return pd.concat(draw_data_dfs, axis=1, copy=False).set_index(index_cols.columns.to_list())

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
    #Example string to extract from: 'Birth prevalence - [0, 24) wks, [0, 500) g'
    return cats.str.extract(r'Birth prevalence - \[(?P<ga_start>\d+), (?P<ga_end>\d+)\) wks, \[(?P<bw_start>\d+), (?P<bw_end>\d+)\) g')
#     return cats.str.extract(r'.*\[(?P<ga_start>\d+), (?P<ga_end>\d+).*\[(?P<bw_start>\d+), (?P<bw_end>\d+).*')

class LBWSGDistribution():
    """
    Class to assign and adjust birthweights and gestational ages of a simulated population.
    """
    def __init__(self):
        self.cat_df = get_ga_bw_by_category()
        self.cat_df['ga_width'] = self.cat_df['ga_end'] - self.cat_df['ga_start']
        self.cat_df['bw_width'] = self.cat_df['bw_end'] - self.cat_df['bw_start']
    
    def cat_to_ga_bw(cat, p1, p2):
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

