import pandas as pd
import re
from typing import Tuple, Dict, Iterable
import gbd_mapping

def _get_stunting_exposed_unexposed_cats():
    """
    Returns the exposed and unexposed categories for Child Stunting.
    Exposed means < -2 sd. Unexposed means >= -2 sd.
    
    Note: A more flexible solution would allow passing the cutoff value,
    and then parse the category descriptions like with LBWSG.
    E.g. the description of 'cat4' is 'Unexposed', whereas 'cat3' has some exposure,
    even though for the present use case we are counting 'cat3' as unexposed.
    """
    return {'exposed': ['cat1', 'cat2'], 'unexposed': ['cat3', 'cat4']}

def _get_wasting_exposed_unexposed_cats():
    """
    Returns the exposed and unexposed categories for Child Wasting.
    Exposed means < -2 sd. Unexposed means >= -2 sd.
    
    Note: A more flexible solution would allow passing the cutoff value,
    and then parse the category descriptions like with LBWSG.
    E.g. the description of 'cat4' is 'Unexposed', whereas 'cat3' has some exposure,
    even though for the present use case we are counting 'cat3' as unexposed.
    """
    return {'exposed': ['cat1', 'cat2'], 'unexposed': ['cat3', 'cat4']}

def _get_intervals_from_name(name: str) -> Tuple[pd.Interval, pd.Interval]:
    """Converts a LBWSG category name to a pair of intervals.

    The first interval corresponds to gestational age in weeks, the
    second to birth weight in grams.
    """
    numbers_only = [int(n) for n in re.findall(r'\d+', name)] # The regex \d+ matches 1 or more digits
    return (pd.Interval(numbers_only[0], numbers_only[1], closed='left'),
            pd.Interval(numbers_only[2], numbers_only[3], closed='left'))

def _get_lbwsg_categories_by_interval(category_dict):
    """
    Return a pandas Series indexed by (gestational age interval, birth weight interval),
    mapping to the corresponding LBWSG category.
    """
    MISSING_CATEGORY = 'cat212'
    category_dict[MISSING_CATEGORY] = 'Birth prevalence - [37, 38) wks, [1000, 1500) g'
    cats = (pd.DataFrame.from_dict(category_dict, orient='index')
            .reset_index()
            .rename(columns={'index': 'cat', 0: 'name'}))
    idx = pd.MultiIndex.from_tuples(cats.name.apply(_get_intervals_from_name),
                                    names=['gestation_time', 'birth_weight'])
    cats = cats['cat']
    cats.index = idx
    return cats

def _get_lbwsg_exposed_unexposed_cats():
    """
    Get the exposed and unxeposed categories for Low Birth Weight and Short Gestation.
    Exposed categories are those with either low birth weight (<2500g) or short gestation (<37wk).
    Unexposed categories are the rest.
    
    Note: It would be better to enable passing specific cutoff values for different use cases.
    (E.g. maybe we're interested in extreme preterm births (<27wk) rather than just preterm (<37wk).)
    """
    LBW_CUTOFF = 2500 #grams - standard LBW cutoff
#     SG_CUTOFF = 37 #weeks - standard SG cutoff
    SG_CUTOFF = 0 # We don't care about short gestation right now - want LBW exposure only
    
    # Get pd.Series that maps interval pairs to categories, and call .reset_index()
    # to get a dataframe with a column for each interval, as well as a 'cat' column.
    lbwsg = gbd_mapping.risk_factors.low_birth_weight_and_short_gestation
    cats = _get_lbwsg_categories_by_interval(lbwsg.categories.to_dict())
    cats = cats.reset_index()
    
    # Get the minimum and supremum of the birth weights and gestational ages for each category.
    # (Note that it's a supremum not a maximum since the intervals are right-open.)
    cats['min_birth_weight'] = cats['birth_weight'].apply(lambda x: x.left)
    cats['supremum_birth_weight'] = cats['birth_weight'].apply(lambda x: x.right)
    cats['min_gestation_time'] = cats['gestation_time'].apply(lambda x: x.left)
    cats['supremum_gestation_time'] = cats['gestation_time'].apply(lambda x: x.right)
    
    exposed_unexposed_cats = {}
    
    # Note that since the intervals are left-closed and right-open, e.g. [2000,2500) and [2500,3000),
    # we need to use <= for the exposed categories to get all the values < CUTOFF.
    exposed_unexposed_cats['exposed'] = cats.loc[
        (cats['supremum_birth_weight'] <= LBW_CUTOFF) |
        (cats['supremum_gestation_time'] <= SG_CUTOFF),
        'cat']
    
    exposed_unexposed_cats['unexposed'] = cats.loc[
        (cats['min_birth_weight'] >= LBW_CUTOFF) &
        (cats['min_gestation_time'] >= SG_CUTOFF),
        'cat']
    
    # A category is 'partially exposed' if one of its intervals spans a cutoff.
    # There *shouldn't* be any of these for the standard cutoff values.
    exposed_unexposed_cats['partially_exposed'] = cats.loc[
        ((cats['supremum_birth_weight'] > LBW_CUTOFF) & (cats['min_birth_weight'] < LBW_CUTOFF)) |
        ((cats['supremum_gestation_time'] > SG_CUTOFF) & (cats['min_gestation_time'] < SG_CUTOFF)),
        'cat']
    
    return exposed_unexposed_cats

_risk_functions = {
    'child_stunting': _get_stunting_exposed_unexposed_cats,
    'child_wasting': _get_wasting_exposed_unexposed_cats,
    'low_birth_weight_and_short_gestation': _get_lbwsg_exposed_unexposed_cats,
}

def get_exposed_unexposed_cats_for_risk(risk: str) -> Dict[str, Iterable[str]]:
    """
    Get the exposed and unexposed categories for the specified risk
    """
    return _risk_functions[risk]()
    