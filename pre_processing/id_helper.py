"""
Module to facilitate using GBD id's in the shared functions.
"""
from db_queries import get_ids

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

def get_entities_from_docstring():
    """Returns the entities listed as valid arguments in the docstring of `get_ids()`.
    Currently there are only 22 entities listed in the docstring, whereas 28 entities are listed
    in the online documentation; those are accessible as id_helper.entities.
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

def list_ids(entity, *entity_names):
    """Returns a list of ids (or a single id) for the specified entity names, suitable for passing to GBD shared functions.""" 
    # Converting from Series to list is necessary for all entities
    # Converting from numpy int64 to int is necessary at least for gbd_round
    ids = [int(entity_id) for entity_id in names_to_ids(entity, *entity_names)]
    if len(ids)==1:
        ids = ids[0]
    elif entity=='gbd_round':
        raise ValueError("Only single gbd_round_id's are allowed in shared functions.")
    return ids

def search_id_table(entity, pattern, **kwargs_for_contains):
    """Searches an entity id table for entity names matching the specified pattern, using pandas.Series.str.contains()."""
    ids = get_ids(entity)
    return ids[ids[get_name_column(entity)].str.contains(pattern, **kwargs_for_contains)]

def find_ids(entity, pattern, **kwargs_for_contains):
    """Searches an entity id table for entity names matching the specified pattern, using pandas.Series.str.contains(),
    and returns a list of ids (or a single id) for the specified entity names, suitable for passing to GBD shared functions.
    """
    ids = search_id_table(entity, pattern, **kwargs_for_contains)
    ids = [int(entity_id) for entity_id in ids[f'{entity}_id']]
    if len(ids)==1:
        ids = ids[0]
    elif entity=='gbd_round':
        raise ValueError("Only single gbd_round_id's are allowed in shared functions.")
    return ids
