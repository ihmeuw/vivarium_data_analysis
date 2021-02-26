import numpy as np, pandas as pd
from db_queries import get_ids, get_population

# seven_days = 0.01917808 # 7/365, rounded to 8 decimal places
# twenty_eight_days = 0.07671233 # 28/365, rounded to 8 decimal places

def get_age_group_data(birth_age_end=None, birth_age_end_description=None):
    if birth_age_end is None:
        if birth_age_end_description is not None:
            raise ValueError("Value of 'birth_age_end_description' cannot be specified if 'birth_age_end' is None.")
        birth_age_end = np.round(5/(365*24*60), 8)
        birth_age_end_description = '5 minutes = 5/(365*24*60) years, rounded to 8 decimals'
    elif birth_age_end_description is None:
        raise ValueError("Value of 'birth_age_end_description' must be specified if 'birth_age_end' is not None.")

    # Define boundaries between age groups, with descriptions
    age_breaks_and_descriptions = [
        (0, "0 days = 0 years"),
        (birth_age_end, birth_age_end_description),
        *((np.round(d/365, 8), f"{d} days = {d}/365 years, rounded to 8 decimals") for d in (7,28)),
        (1, "1 year"),
        *((n, f"{n} years") for n in range(5,85,5)),
    ]
    # Unzip the list of 2-tuples to get two lists
    age_breaks, age_descriptions = zip(*age_breaks_and_descriptions)
    
    # Get age group names for the age group id's corresponding to the intervals between the age breaks
    age_group_ids = [164, *range(2,21)]
    age_group_df = (get_ids('age_group')
                    .set_index('age_group_id')
                    .loc[age_group_ids]
                    .reset_index()
                   )
    age_group_df.index = pd.IntervalIndex.from_breaks(age_breaks, closed='left', name='age_group_interval')
    
    # Record the age group start and end for each interval
    age_group_df['age_group_start'] =  age_breaks[:-1]
    age_group_df['age_group_end'] = age_breaks[1:]
    age_group_df['age_start_description'] = age_descriptions[:-1]
    age_group_df['age_end_description'] = age_descriptions[1:]
    return age_group_df

#     age_intervals = pd.IntervalIndex.from_breaks(age_breaks, closed='left', name='age_group_interval')
#     age_group_df = pd.DataFrame(
#         {'age_group_id': age_group_ids,
#          'age_group_name': 
#          'age_group_start': age_breaks[:-1],
#          'age_group_end': age_breaks[1:],
#          'age_end_description': desc
    
#     age_breaks = [0, *np.round([birth_age_end, 7/365, 28/365], 8), 1, *range(5,85,5)]
#     age_break_descriptions = ['0 days = 0 years',
#                               birth_age_end_description,
#                               '7 days = 7/365 years, rounded to 8 decimals',
#                               '28 days = 28/365 years, rounded to 8 decimals',
#                               '1 year',
#                               *[f"{n} years" for n in range(5,85,5)],
#                              ]
#     age_intervals = pd.IntervalIndex.from_breaks(
        
#     birth_end = 5/(365*24*60) # five minutes
#     seven_days = 7/365
#     twenty_eight_days = 28/365
#     breaks_under_one = np.round([birth_end, seven_days, twenty_eight_days], 8)
#     age_intervals = pd.IntervalIndex.from_breaks(
#         [0, *breaks_under_one, 1, *range(5,85,5)], closed='left', name='age_group_interval'
#     )
    
#     age_group_ids = pd.Series([164, *range(2,21)], index=age_intervals, name='age_group_id')
#     return age_group_ids