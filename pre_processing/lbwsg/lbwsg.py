"""
This file will probably be a lightweight version of some of the things from the
LBWSG component in Vivarium Public Health:

https://github.com/ihmeuw/vivarium_public_health/blob/master/src/vivarium_public_health/risks/implementations/low_birth_weight_and_short_gestation.py
"""

import pandas as pd
import re
from typing import Tuple#, Dict, Iterable
import gbd_mapping

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

def get_lbwsg_categories_by_interval(category_dict):
    """
    Return a pandas Series indexed by (gestational age interval, birth weight interval),
    mapping to the corresponding LBWSG category.
    """
    MISSING_CATEGORY = 'cat212'
    category_dict[MISSING_CATEGORY] = 'Birth prevalence - [37, 38) wks, [1000, 1500) g'
    cats = (pd.DataFrame.from_dict(category_dict, orient='index')
            .reset_index()
            .rename(columns={'index': 'cat', 0: 'name'}))
    idx = pd.MultiIndex.from_tuples(cats.name.apply(get_intervals_from_name),
                                    names=['gestation_time', 'birth_weight'])
    cats = cats['cat']
    cats.index = idx
    return cats

