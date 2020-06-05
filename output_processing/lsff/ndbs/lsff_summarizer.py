import numpy as np, pandas as pd
# import os
from collections import Counter
#from neonatal.tabulation import risk_mapper

# Define functional categories of columns in output.hdf
# The strategy for processing the data in a column depends on its category.

# Data about the simulation input parameters
INPUT_DRAW_COLUMN_CATEGORY = 'input_draw'
RANDOM_SEED_COLUMN_CATEGORY = 'random_seed'
LOCATION_COLUMN_CATEGORY = 'location'
INTERVENTION_COLUMN_CATEGORY = 'intervention'

# Categories to group by when aggregating over random seeds
INDEX_COLUMN_CATEGORIES = (LOCATION_COLUMN_CATEGORY, INTERVENTION_COLUMN_CATEGORY, INPUT_DRAW_COLUMN_CATEGORY)

# Count-space columns
PERSON_TIME_COLUMN_CATEGORY = 'person_time'
STATE_PERSON_TIME_COLUMN_CATEGORY = 'state_person_time'
BIRTH_PREVALENCE_COLUMN_CATEGORY = 'birth_prevalence'
LIVE_BIRTHS_COLUMN_CATEGORY = 'live_births'

# Non-count-space columns
PROPORTION_COLUMN_CATEGORY = 'proportion'
DISTRIBUTION_MEAN_COLUMN_CATEGORY = 'distribution_mean'
DISTRIBUTION_VARIANCE_COLUMN_CATEGORY = 'distribution_variance'
DISTRIBUTION_STD_DEV_COLUMN_CATEGORY = 'distribution_std_dev'

# Column to store counts of random seeds in groupby
RANDOM_SEED_COUNT_COLUMN = 'random_seed_count'

# Categories with special aggregations (all other columns default to 'sum')
# SPECIAL_AGGREGATIONS_BY_CATEGORY = {
#     RANDOM_SEED_COLUMN_CATEGORY: 'count',
#     PROPORTION_COLUMN_CATEGORY: 'mean',
#     DISTRIBUTION_MEAN_COLUMN_CATEGORY: 'mean',
#     DISTRIBUTION_STD_DEV_COLUMN_CATEGORY: lambda x: np.sqrt(np.mean(x**2)), # std. dev. of independent samples
# }

def root_mean_square(x):
    """Returns the root mean square of x (array-like)"""
    return np.sqrt(np.mean(x**2))

# Special aggergation functions (other than 'sum') and the column categories they apply to
SPECIAL_AGGREGATIONS_BY_CATEGORY = pd.DataFrame({
    'function_name': ['count', 'mean', 'root_mean_square'],
    'function': ['count', 'mean', root_mean_square],
    'column_categories': [(RANDOM_SEED_COLUMN_CATEGORY,),
                          (PROPORTION_COLUMN_CATEGORY, DISTRIBUTION_MEAN_COLUMN_CATEGORY,),
                          (DISTRIBUTION_STD_DEV_COLUMN_CATEGORY,),
                         ]
})

# Used in .categorize_data_by_column() method to categorize all columns in the data.
# Dictionary to find output columns in specific catories by matching column names with regular expressions
# using pd.DataFrame.filter()
def default_column_categories_to_search_regexes():
    return {
        # Data about the simulation input parameters
        INPUT_DRAW_COLUMN_CATEGORY: r'input_draw',
        RANDOM_SEED_COLUMN_CATEGORY: r'random_seed',
        LOCATION_COLUMN_CATEGORY: r'location', # any column containing the string 'location'
        INTERVENTION_COLUMN_CATEGORY: r'\w*\D\.\w+', # columns look like 'intervention_name.paramater'. Parameters determine scenario.
        # Metadata about the simulation runs
        'run_time': r'run_time', # simulation run time
        # Data about the simulation results
        'diseases_at_end': r'_prevalent_cases_at_sim_end$', # cause prevalence at end of simulation
        'transition_count': r'_event_count', # disease state transition events throughout simulation'
        'population': r'population', # population statistics at end of simulation
        PERSON_TIME_COLUMN_CATEGORY: r'^person_time', # string starts with 'person_time'
        STATE_PERSON_TIME_COLUMN_CATEGORY: r'_person_time', # string contains '_person_time'
        'treated_days': r'treated_days', # total number of days of treatment
        'mortality': r'^death_due_to_', # string starts with 'death_due_to_'
        'total_daly': r'^years_lived_with_disability$|^years_of_life_lost$', # sum of these 2 columns = DALYs for whole sim
        'yld': r'^ylds_due_to_', # YLD columns start with 'ylds_due_to_'
        'yll': r'^ylls_due_to_', # YLL columns start with 'ylls_due_to_'
        'categorical_risk': r'_cat\d+_exposed', # columns for categorical risk exposures contain, e.g. '_cat16_exposed'
#         'graded_sequela': r'mild|moderate|severe|unexposed', # anemia, for example
        BIRTH_PREVALENCE_COLUMN_CATEGORY: r'born_with',
        LIVE_BIRTHS_COLUMN_CATEGORY: r'live_births',
        PROPORTION_COLUMN_CATEGORY: r'_proportion',
        DISTRIBUTION_MEAN_COLUMN_CATEGORY: r'_mean', # mean of a distribution
        DISTRIBUTION_VARIANCE_COLUMN_CATEGORY: r'_variance', # variance of a distribution
        DISTRIBUTION_STD_DEV_COLUMN_CATEGORY: r'_sd', # standard deviation of a distribution
        'prevalence': r'prevalent_count', # prevalent count of a cause, e.g. 'prevalent_count_at_birth'
    }


# Used in .reindex_sub_dataframes() method to create a MultiIndex from original one-level index.
# Dictionary to extract useful information from column names with regular expressions
# using pd.Series.str.extract()
# Comments show example column names to extract cause/metric and demographic details from:
def default_column_categories_to_extraction_regexes():
    return {
        'diseases_at_end':
            # 'lower_respiratory_infections_prevalent_cases_at_sim_end'
            # 'measles_prevalent_cases_at_sim_end'
            r'^(?P<cause>\w+)_(?P<measure>prevalent_cases_at_sim_end)$',
####
# 2019-07-26: These regexen work, but I'm not sure if they're useful. Some extracted strings will be empty
#             but should later be replaced with e.g. 'infected' or 'all', which complicates things.
#             It could perhaps make things easier later to extract NaN's instead of empty strings,
#             but I'm not sure.
#             Idea: Add a method to standardize column names before any other processing. You could
#             pass it 2 regexen (a search and a replacement) for each column category you want to alter.
#             OR: In this dictionary, use lists as values in order to pass 1 or 3 regexes
#             (optional search and replace, plus 1 for extraction).
#         'disease_event_count':
#             # 'susceptible_to_lower_respiratory_infections_event_count',
#             # 'lower_respiratory_infections_event_count',
#             # 'susceptible_to_measles_event_count',
#             # 'measles_event_count', # event category should be 'infected'
#             # 'recovered_from_measles_event_count'
#             r'^(?P<category>susceptible|recovered|)(?:_to_|_from_|)(?P<cause>\w+)_(?P<measure>event_count)$',
#         'population':
#             # 'total_population_untracked',
#             # 'total_population_tracked',
#             # 'total_population', # population category should be 'all'
#             # 'total_population_living',
#             # 'total_population_dead'
#             r'^total_(?P<measure>population)(?:_|)(?P<category>\w*)$',
        'person_time':
            # 'person_time_in_2020_among_female_in_age_group_late_neonatal'
            # 'person_time_in_2024_among_male_in_age_group_1_to_4'
#             '(?P<person_time_metric>^person_time)_in_(?P<year>\d{4})_among_(?P<sex>\w+)_in_age_group_(?P<age_group>\w+$)',
            '^(?P<measure>person_time)(?:_in_(?P<year>\d{4}))?(?:_among_(?P<sex>male|female))?(?:_in_age_group_(?P<age_group>\w+))?$',
        'mortality':
            # 'death_due_to_lower_respiratory_infections'
            # 'death_due_to_other_causes'
            '^(?P<measure>death)_due_to_(?P<cause>\w+?)(?:_in_(?P<year>\d{4}))?(?:_among_(?P<sex>male|female))?(?:_in_age_group_(?P<age_group>\w+))?$',
        'yld':
            # 'ylds_due_to_diarrheal_diseases_in_2020_among_male_in_age_group_early_neonatal'
            # 'ylds_due_to_hemolytic_disease_and_other_neonatal_jaundice_in_2025_among_female_in_age_group_1_to_4'
#             '(?P<yld_cause_name>^ylds_due_to_\w+)_in_(?P<year>\d{4})_among_(?P<sex>\w+)_in_age_group_(?P<age_group>\w+$)',
            r'^(?P<measure>ylds)_due_to_(?P<cause>\w+?)(?:_in_(?P<year>\d{4}))?(?:_among_(?P<sex>male|female))?(?:_in_age_group_(?P<age_group>\w+))?$',
        'yll':
            # 'ylls_due_to_protein_energy_malnutrition_in_2020_among_male_in_age_group_early_neonatal'
            # 'ylls_due_to_hemolytic_disease_and_other_neonatal_jaundice_in_2025_among_female_in_age_group_post_neonatal'
            # 'ylls_due_to_other_causes_in_2025_among_female_in_age_group_1_to_4'
            # 'ylls_due_to_protein_energy_malnutrition_among_male_in_age_group_early_neonatal',
            # 'ylls_due_to_hemolytic_disease_and_other_neonatal_jaundice_in_2025',
            # 'ylls_due_to_other_causes_in_age_group_1_to_4',
            # 'ylls_due_to_protein_energy_malnutrition',
#             '(?P<yll_cause_name>^ylls_due_to_\w+)_in_(?P<year>\d{4})_among_(?P<sex>\w+)_in_age_group_(?P<age_group>\w+$)',
            r'^(?P<measure>ylls)_due_to_(?P<cause>\w+?)(?:_in_(?P<year>\d{4}))?(?:_among_(?P<sex>male|female))?(?:_in_age_group_(?P<age_group>\w+))?$',
    #     'total_daly': '',
        'categorical_risk':
            # 'child_stunting_cat2_exposed_in_2020_among_0_to_5'
            r'^(?P<risk>\w+)_(?P<category>cat\d+)_exposed(?:_in_(?P<year>\d{4})|)(?:_among_(?P<age_group>\w+)|)$',
        'graded_sequela':
            # 'anemia_unexposed_in_2020_among_0_to_5'
            # 'anemia_mild_in_2020_among_0_to_5'
            r'^(?P<sequela>\w+)_(?P<category>mild|moderate|severe|unexposed)(?:_in_(?P<year>\d{4})|)(?:_among_(?P<age_group>\w+)|)$',
    }

class LSFFOutputSummarizer():
    """Class to provide functions to summarize output from neonatal model"""

    def __init__(self, model_output_df, column_categories_to_search_regexes=None):
        """Initialize this object with a pandas DataFrame of output"""
        self.data = model_output_df
        self.column_categories_to_search_regexes = column_categories_to_search_regexes
        # Initializes self.columns, self.found_columns, self.missing_columns, self.repeated_columns, self.empty_categories:
#         self.find_columns(column_categories_to_search_regexes)
#         # Sub-DataFrames corresponding to each column category
#         self._subdata = {column_category: self.data[column_names]
#                         for column_category, column_names in self.columns.items()}
        self.column_categories_to_extraction_regexes = None
        self.index_column_categories = None # or an empty tuple ()?
#         self.index_columns = None
        # Initializes self._subdata,
        # self.found_columns, self.missing_columns, self.repeated_columns, self.empty_categories:
        self.categorize_data_by_column(column_categories_to_search_regexes)

    def categorize_data_by_column(self, column_categories_to_search_regexes=None):
        """Categorize the columns in the data to make sure we don't miss anything or overcount"""

        # If we don't already have a search dictionary, initialize it to the default
        if self.column_categories_to_search_regexes is None:
            self.column_categories_to_search_regexes = default_column_categories_to_search_regexes()
        # If a search dictionary was passed, update our current dictionary with it
        if column_categories_to_search_regexes is not None:
            self.column_categories_to_search_regexes.update(column_categories_to_search_regexes)

#         # If dictionary was passed, replace any existing dictionary
#         if column_categories_to_search_regexes is not None:
#             self.column_categories_to_search_regexes = column_categories_to_search_regexes
#         # Otherwise, use the existing dictionary if we have one, or initialize to default
#         elif self.column_categories_to_search_regexes is None:
#             self.column_categories_to_search_regexes = default_column_categories_to_search_regexes()

        # If self.aggregate_over_random_seeds() has already been called, the index columns will have become
        # the dataframe's index; copy them back into columns so we can track them in the column report.
        if self.index_column_categories is not None:
            all_data = pd.concat([self.data.index.to_frame(), self.data], axis='columns')
        else:
            all_data = self.data

        # Create dictionary mapping each column category to a sub-dataframe of columns in that category
        self._subdata = {category: all_data.filter(regex=cat_regex)
                        for category, cat_regex in self.column_categories_to_search_regexes.items()}

        # Create dictionary mapping each column category to a pd.Index of column names in that category
        # 2019-07-18: Eliminating this attribute in favor of accessing column names via _subdata frames.
        # 2019-07-26: Reinstating this attribute to retain original column names vs. only MultiIndices after parsing.
#         self.columns = {category: self.data.filter(regex=cat_regex).columns
#                         for category, cat_regex in self.column_categories_to_search_regexes.items()}
        self._columns = pd.Series({category: df.columns for category, df in self._subdata.items()})

        # Get a list (or pd.Series) of the found columns to check for missing or duplicate columns
        # found_columns = pd.concat(pd.Series(col_names) for col_names in columns.values())
        self.found_columns = [column for cat_data in self._subdata.values() for column in cat_data.columns]

        # Find any missing or duplicate columns
        self.missing_columns = set(self.data.columns) - set(self.found_columns)
        self.repeated_columns = {column_name: count for column_name, count in Counter(self.found_columns).items() if count > 1}

        # Also find any categories that didn't return a match
        self.empty_categories = [category for category, cat_data in self._subdata.items() if len(cat_data.columns) == 0]

    def print_column_report(self):
        """
        Print the total number of columns and the number of columns found in each category. Also print
        the missing and repeated columns if there were any, and any categories that didn't return a match.
        """

#         col_cat_counts = self.column_category_counts()

        print(f"Number of data columns in output: {len(self.data.columns)}")
        print(f"Total number of columns captured in categories: {len(self.found_columns)}\n")

        print("Number of columns in each category:\n", self.column_category_counts(), "\n")

        print(f"Missing ({len(self.missing_columns)} data column(s) not captured in a category):\n",
              self.missing_columns)
        print(f"\nRepeated ({len(self.repeated_columns)} data column(s) appearing in more than one category):\n",
              self.repeated_columns)
        print(f"\nEmpty categories ({len(self.empty_categories)} categories with no matching data columns):\n",
              self.empty_categories)

    def column_category_counts(self):
        """Get a dictionary mapping column categories to the number of columns found in that category."""
        return {category: len(cat_data.columns) for category, cat_data in self._subdata.items()}

    def column_categories(self):
        """Get the list of column categories."""
#         return list(self.column_categories_to_search_regexes.keys()) # This should be equivalent
        return list(self._subdata.keys())

    def columns(self, *column_categories):
        """Get the column names in the specified category(ies), or in all categories if none specified."""
#         return self._subdata[column_category].columns
#         return self._columns[column_category]
        if len(column_categories) == 0:
            column_categories = self.column_categories()
        return [column for columns in self._columns[list(column_categories)] for column in columns]

    def index_columns(self):
        """Get the names of the index columns. Returns an empty list if index column categories have not been assigned."""
        if self.index_column_categories is None: return []
        return self.columns(*self.index_column_categories)

    def subdata(self, column_category=None):
        """Get the sub-dataframe containing columns of the specified category, or all the columns if no category is specified."""
        if column_category is None:
            return self.data
        else:
            return self._subdata[column_category]

    def rename_intervention_columns(self, column_name_mapper=None):
        """
        Shorten the intervention column names by removing 'intervention_name.' from the beginning,
        or use the specified mapper to rename the columns if one is passed.
        (These columns become part of the MultiIndex in the aggregate_over_random_seeds() method below.)
        """
        if column_name_mapper is None:
#         # Replace all characters from start up through '.' with the empty string - or use more descriptive version below
#         intervention_columns = self.columns['intervention'].str.replace(r'^.*\.', '')
            # Replace whole string with the short name that comes after the period:
            intervention_columns = self._columns['intervention'].str.replace(r'.+\.(?P<short_name>\w+)', r'\g<short_name>')
            column_name_mapper = {long_name: short_name for long_name, short_name
                                  in zip(self._columns['intervention'], intervention_columns)}
        else:
            intervention_columns = column_name_mapper.values()

        self.data = self.data.rename(columns=column_name_mapper)
        self._subdata['intervention'] = self.data[intervention_columns]
        self._columns['intervention'] = self._subdata['intervention'].columns
        self.column_categories_to_search_regexes['intervention'] = '|'.join(intervention_columns)

    def aggregate_over_random_seeds(self, index_column_categories=None, alternative_aggregation_functions=None):
        """
        Group the data by location, intervention parameters, and input draw, and sum values over random seeds.
        Maybe it would be good to be able to specify custom names for the new MultiIndex...
        """
#         self._rename_intervention_columns() # Replace the intervention column names with shorter versions
        # If a list of index columns was passed, use these to index the groupby.
        if index_column_categories is not None:
            self.index_column_categories = index_column_categories
        # Otherwise, initialize to the default INDEX_COLUMN_CATEGORIES if necessary
        elif self.index_column_categories is None:
#             self.index_columns = [*self.columns('location'), *self.columns('intervention'), *self.columns('input_draw')]
#             self.index_columns = [column for category in INDEX_COLUMN_CATEGORIES for column in self._columns[category]]
            self.index_column_categories = INDEX_COLUMN_CATEGORIES
        # Otherwise, we have already aggregated, so do nothing and return
        else:
            return

#         self.data = self.data.drop(columns=self.columns(RANDOM_SEED_COLUMN_CATEGORY)) # Don't sum random seeds. Instead...
#         self.data[RANDOM_SEED_COUNT_COLUMN] = 1 # This will count random seeds when we do .groupby().sum()
#         # Update the mapped random_seed sub-dataframe and column name with the new value
#         self._subdata[RANDOM_SEED_COLUMN_CATEGORY] = self.data[[RANDOM_SEED_COUNT_COLUMN]] # Use a list to create a DataFrame rather than a Series
#         self._columns[RANDOM_SEED_COLUMN_CATEGORY] = self._subdata[RANDOM_SEED_COLUMN_CATEGORY].columns

        # ADD CODE TO HANDLE AGGREGATIONS BESIDES SUM (Done)
        # Use SPECIAL_AGGREGATIONS_BY_CATEGORY dictionary and .agg()
        # CURRENT VERSION IS SLOW - CHANGE to use .sum() on most columns,
        # and only use .agg(agg_dict) for special columns, then concatenate.
#         self.data = self.data.groupby(self.index_columns).agg(agg_dict)

#         non_sum_columns = [column
#                            for categories in SPECIAL_AGGREGATIONS_BY_CATEGORY['column_categories']
#                            for column in self.columns(*categories)
#                           ]
#         sum_columns = [column for column in self.columns() if column not in non_sum_columns]

        agg_df = SPECIAL_AGGREGATIONS_BY_CATEGORY.set_index('function_name')
        if alternative_aggregation_functions is not None:
            for function_name, function in alternative_aggregation_functions.items():
                agg_df.loc[function_name, 'function'] = function

        non_sum_categories = [category
                                 for categories in agg_df['column_categories']
                                 for category in categories
                             ]
        sum_categories = set(self.column_categories()) - set(non_sum_categories) - set(self.index_column_categories)

#         print(sum_categories)

        sum_columns = self.columns(*sum_categories)
        sum_data = self.data.groupby(self.index_columns())[sum_columns].sum()
        aggregated_dfs = [sum_data]

        for function_name in agg_df.index:
            agg_columns = self.columns(*agg_df.loc[function_name, 'column_categories'])
            aggregated_data = self.data.groupby(self.index_columns())[agg_columns].agg(
                agg_df.loc[function_name, 'function']
            )
            aggregated_dfs.append(aggregated_data)

        self.data = pd.concat(aggregated_dfs, axis=1, copy=False)

#         self.data = self.data.groupby(self.index_columns).sum()

        # Perhaps add some code to count random seeds per location_intervention_draw combination,
        # and count how many scenarios each draw appears in. Ok, random_seed_count is now included.
        # It could be useful to add separate functions to return DataFrames with the following columns:
        # a) location, random_seed_count, number_of_scenario_draw_combinations
        # b) location, draw_number, number_of_scenarios
        # These should be simple to implement by passing a Counter object into a new dataframe
#         self._subdata = {category: self.data.filter(regex=cat_regex)
#                         for category, cat_regex in self.column_categories_to_search_regexes.items()}
        self.categorize_data_by_column()

    def parse_column_names_and_reindex(self, column_categories_to_extraction_regexes=None):
        """
        Create subdataframes for the specified categories with columns MultiIndexed by
        data extracted from original column names.
        """
        # If our extraction dictionary hasn't been initialized, initialize it with default
        if self.column_categories_to_extraction_regexes is None:
            self.column_categories_to_extraction_regexes = default_column_categories_to_extraction_regexes()
        # If an extraction dictionary was passed, update the current dictionary with it
        if column_categories_to_extraction_regexes is not None:
            self.column_categories_to_extraction_regexes.update(column_categories_to_extraction_regexes)

#         if column_categories_to_extraction_regexes is not None:
#             self.column_categories_to_extraction_regexes = column_categories_to_extraction_regexes
#         elif self.column_categories_to_extraction_regexes is None:
#             self.column_categories_to_extraction_regexes = default_column_categories_to_extraction_regexes()

        for category, extraction_regex in self.column_categories_to_extraction_regexes.items():
#             print(category)
#             column_decompositions = self._subdata[category].columns.str.extract(extraction_regex)
            column_decompositions = self._columns[category].columns.str.extract(extraction_regex)
#             print(column_decompositions)
            # Note: pd.MultiIndex.from_frame() requires pandas 0.24 or higher. If using version 0.23 or lower, instead use:
            # pd.MultiIndex.from_tuples(column_decomposition.itertuples(index=False), names=column_decomposition.columns)
            self._subdata[category].columns = pd.MultiIndex.from_frame(column_decompositions.dropna(axis=1, how='all'))
