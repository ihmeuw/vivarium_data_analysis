"""
Module to facilitate using GBD id's in the shared functions.
"""
from db_queries import get_ids
from pandas import DataFrame

# The following list of valid entities was retrieved on 2020-10-12 from the hosted documentation:
# https://scicomp-docs.ihme.washington.edu/db_queries/current/get_ids.html
entities = [
 'age_group',
 'age_group_set',
 'cause',
 'cause_set',
 'cause_set_version',
 'covariate',
 'decomp_step',
 'gbd_round',
 'healthstate',
 'indicator_component',
 'life_table_parameter',
 'location',
 'location_set',
 'location_set_version',
 'measure',
 'metric',
 'modelable_entity',
 'sdg_indicator',
 'sequela',
 'sequela_set',
 'sequela_set_version',
 'sex',
 'split',
 'study_covariate',
 'rei',
 'rei_set',
 'rei_set_version',
 'year'
]

def get_entities():
    """Returns the entities listed as valid arguments to `get_ids()` in the online documentation on 2020-10-12.
    https://scicomp-docs.ihme.washington.edu/db_queries/current/get_ids.html
    """
    return entities

def get_entities_from_docstring():
    """Returns the entities listed as valid arguments in the docstring of `get_ids()`.
    Currently there are only 22 entities listed in the docstring, whereas 28 entities are listed
    in the online documentation; those are accessible via get_entities().
    """
    docstring = get_ids.__doc__
    # This simplistic solution works with the current version, but it may need to be updated
    # to a more robust solution if the docstring changes...
    return docstring[docstring.find('[')+1:docstring.find(']')].split()

def find_anomalous_name_columns(entities):
    """Lists columns of entity tables that do not conatin a column called f'{entity}_name'."""
    entities_columns = {entity: get_ids(entity).columns for entity in entities}
    return {entity: columns for entity, columns in entities_columns.items() if f'{entity}_name' not in columns}

def get_name_column(entity):
    """Returns the name column for the entity in the entity id table."""
    if entity=='year':
        return 'year_id'
    elif entity=='life_table_parameter':
        return 'parameter_name'
    elif entity in [
        'cause_set_version', 'gbd_round', 'location_set_version',
        'sequela_set_version', 'sex', 'study_covariate', 'rei_set_version'
    ]: 
        return entity
    else:
        return f'{entity}_name'

def names_to_ids(entity, *entity_names):
    """Returns a pandas Series mapping entity names to entity id's for the specified GBD entity."""
    ids = get_ids(entity)
    entity_name_col = get_name_column(entity)
    if len(entity_names)>0:
        ids = ids.query(f'{entity_name_col} in {entity_names}')
    # Year table only has one column, so we copy it
    if entity=='year':
        entity_name_col = 'year'
        ids[entity_name_col] = ids['year_id']
    return ids.set_index(entity_name_col)[f'{entity}_id']

def ids_to_names(entity, *entity_ids):
    """Returns a pandas Series mapping entity id's to entity names for the specified GBD entity."""
    ids = get_ids(entity)
    if len(entity_ids)>0:
        ids = ids.query(f'{entity}_id in {entity_ids}')
    entity_name_col = get_name_column(entity)
    # Year table only has one column, so we copy it
    if entity=='year':
        entity_name_col = 'year'
        ids[entity_name_col] = ids['year_id']
    return ids.set_index(f'{entity}_id')[entity_name_col]

def process_singleton_ids(ids, entity):
    """Returns a single id if len(ids)==1. If len(ids)>1, returns ids (assumed to be a list), or raises
    a ValueError if the shared functions expect a single id rather than a list for the specified entity.
    """
    if len(ids)==1:
        ids = ids[0]
    elif entity=='gbd_round': # Also version id's?
        raise ValueError("Only single gbd_round_id's are allowed in shared functions.")
    return ids

def list_ids(entity, *entity_names):
    """Returns a list of ids (or a single id) for the specified entity names,
    suitable for passing to GBD shared functions.
    """
    # Converting from Series to list is necessary for all entities
    # Converting from numpy int64 to int is necessary at least for gbd_round
#     ids = [int(entity_id) for entity_id in names_to_ids(entity, *entity_names)]
    ids = names_to_ids(entity, *entity_names).to_list()
#     if len(ids)==1:
#         ids = ids[0]
#     elif entity=='gbd_round':
#         raise ValueError("Only single gbd_round_id's are allowed in shared functions.")
    ids = process_singleton_ids(ids, entity)
    return ids

def get_entity_and_id_colname(table):
    """Returns the entity and entity id column name from an id table,
    assuming the entity id column name is f'{entity}_id',
    and that this is the only column ending in '_id'.
    """
    id_colname = table.columns[table.columns.str.contains(r'\w+_id$')][0]
    entity = id_colname[:-3]
    return entity, id_colname

def get_entity(table):
    """Returns the entity represented by a given id table,
    assuming the id column name is f'{entity}_id'.
    """
    return get_entity_and_id_colname(table)[0]

def get_id_colname(table):
    """Returns the entity id column name in the given id table,
    assuming it is the only column name that ends with '_id'.
    """
    return get_entity_and_id_colname(table)[1]

def ids_in(table):
    """Returns the ids in the given dataframe, either as a list of ints or a single int."""
#     ids = [int(entity_id) for entity_id in table[get_id_colname(table)]]
    entity, id_colname = get_entity_and_id_colname(table)
    ids = table[id_colname].to_list()
#     if len(ids)==1:
#         ids = ids[0]
#     elif entity=='gbd_round':
#         raise ValueError("Only single gbd_round_id's are allowed in shared functions.")
    ids = process_singleton_ids(ids, entity)
    return ids

def search_id_table(table_or_entity, pattern, search_col=None, return_all_columns=False, **kwargs_for_contains):
    """Searches an entity id table for entity names matching the specified pattern, using pandas.Series.str.contains()."""
    if type(table_or_entity)==DataFrame:
        df = table_or_entity
        entity = get_entity(df)
    elif type(table_or_entity)==str:
        entity = table_or_entity
        df = get_ids(entity, return_all_columns)
    else:
        raise TypeError(f'Expecting type `pandas.DataFrame` or `str` for `table_or_entity`. Got type {type(table_or_entity)}.')

    if search_col is None:
        search_col = get_name_column(entity)

    return df[df[search_col].str.contains(pattern, **kwargs_for_contains)]

def find_ids(table_or_entity, pattern, search_col=None, return_all_columns=False, **kwargs_for_contains):
    """Searches an entity id table for entity names matching the specified pattern, using pandas.Series.str.contains(),
    and returns a list of ids (or a single id) for the specified entity names, suitable for passing to GBD shared functions.
    """
    df = search_id_table(table_or_entity, pattern, search_col=None, return_all_columns=False, **kwargs_for_contains)
    return ids_in(df)
#     ids = [int(entity_id) for entity_id in ids[f'{entity}_id']]
#     if len(ids)==1:
#         ids = ids[0]
#     elif entity=='gbd_round':
#         raise ValueError("Only single gbd_round_id's are allowed in shared functions.")
#     return ids
