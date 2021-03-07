import pandas as pd, numpy as np
from scipy import stats
from collections import namedtuple

import sys, os.path
sys.path.append(os.path.abspath("../.."))
from probability import prob_utils
from plots_and_other_misc import lsff_plots

def create_bw_dose_response_distribution():
    """Define normal distribution representing parameter uncertainty of dose-response on birthweight.
    mean = 15.1 g per 10 mg daily iron, 95% CI = (6.0,24.2).
    Effect size comes from Haider et al. (2013)
    """
    # mean and 0.975-quantile of normal distribution for mean difference (MD)
    mean = 16.7 # g per 10 mg daily iron
    q_975 = 26.11 # 97.5th percentile
    std = prob_utils.normal_stdev_from_mean_quantile(mean, q_975, 0.975)
    # Frozen normal distribution for MD, representing uncertainty in our effect size
    return stats.norm(mean, std)

# Define distributions of iron concentration in fortified flour for each country,
# representing parameter uncertainty.
# Units are mg iron as NaFeEDTA per kg flour
# Currently we use a uniform distribution for India and degenerate (point mass)
# distributions for Ethiopia and Nigeria.
# Eventually this should be stored in some external data source that will then be loaded.
# iron_conc_distributions = {
#     'India': stats.uniform(loc=14, scale=21.5-14), # Uniform(14,21.5) mg iron as NaFeEDTA per kg flour
#     'Ethiopia': stats.bernoulli(p=0,loc=30), # 30 mg iron as NaFeEDTA per kg flour,
#     'Nigeria': stats.bernoulli(p=0,loc=40), # 40 mg iron as NaFeEDTA per kg flour,
# }

def get_iron_concentration(location, draws):
    """
    Get the iron concentration in flour for specified location (mg iron as NaFeEDTA per kg flour),
    with parameter uncertainty if there is any.
    
    Returns
    -------
    scalar if there is no uncertainty, or pandas Series indexed by draw if there is uncertainty.
    """
    if location == 'India':
        iron_conc_dist = stats.uniform(loc=14, scale=21.5-14), # Uniform(14,21.5) mg iron as NaFeEDTA per kg flour
#         if take_mean: # we'd have to pass another argument
#         if isinstance(draws, pd.CategoricalIndex): # We used a categorical index if we took the mean over draws
        if len(draws) == 1: # Use our best guess if there's only one draw or we took the mean
            iron_concentration = iron_conc_dist.mean()
        else:
            iron_concentration = pd.Series(
                iron_conc_dist.rvs(size=len(draws)), index=draws, name='iron_concentration')
    elif location == 'Ethiopia':
        iron_concentration = 30 # 30 mg iron as NaFeEDTA per kg flour
    elif location == 'Nigeria':
        iron_concentration = 40 # 40 mg iron as NaFeEDTA per kg flour
    else:
        raise ValueError(f'Unsupported location: {location}')

    return iron_concentration

def sample_flour_consumption(location, sample_size):
    """Sample from distribution of daily flour consumption (in Ethiopia).
    The distribution is uuniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
#     # TODO: Edit this to simply call the propensity version below
#     # Define quartiles in g of flour per day
#     q = (0, 77.5, 100, 200, 350.5) # min=0, q1=77.5, q2=100, q3=200, max=350.5
#     u = np.random.uniform(0,1,size=sample_size)
#     # Scale the uniform random number to the correct interval based on its quartile
#     return np.select(
#         [u<0.25, u<0.5, u<0.75, u<1],
#         [q[1]*u/0.25, q[1]+(q[2]-q[1])*(u-0.25)/0.25, q[2]+(q[3]-q[2])*(u-0.5)/0.25, q[3]+(q[4]-q[3])*(u-0.75)/0.25]
#     )
    return get_flour_consumption_from_propensity(location, np.random.uniform(0,1,size=sample_size))

def get_flour_consumption_from_propensity(location, propensity):
    """Get distribution of daily flour consumption (in Ethiopia) based on the specified propensity array.
    The distribution is uuniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
    # Define quartiles in g of flour per day
    q = (0, 77.5, 100, 200, 350.5) # q0=min=0, q1=77.5, q2=100, q3=200, q4=max=350.5
    u = propensity # use shorter name for readibility - u ~ uniform(0,1) random number
    # Scale the uniform random number to the correct interval based on its quartile
    return np.select(
        [u<0.25, u<0.5, u<0.75, u<1],
        [q[1]*u/0.25, q[1]+(q[2]-q[1])*(u-0.25)/0.25, q[2]+(q[3]-q[2])*(u-0.5)/0.25, q[3]+(q[4]-q[3])*(u-0.75)/0.25]
    )

def calculate_birthweight_shift(dose_response, iron_concentration, daily_flour):
    """
    Computes the increase in birthweight (in grams) given the following:
    
    dose_response: g of birthweight increase per 10 mg daily iron
    iron_concentration: mg iron as NaFeEDTA per kg flour
    daily flour: g of iron eaten per day by pregnant mother
    """
    return (dose_response/10)*(iron_concentration)*(daily_flour/1_000)

def get_flour_coverage_df():
    """Return the dataframe of flour fortification coverage."""
    # Import dataframe storing wheat flour fortification coverage parameters from lsff_plots.
    # Eventually, this will need to be updated to incorporate data from more countries.
    return lsff_plots.get_coverage_dfs()['flour'].T

def get_global_data(draws, mean_draws_name=None):
    """
    Information shared between locations and scenarios. May vary by draw.
    
    Returns
    -------
    draw_numbers
    draws - a pandas Index object
    dose-response of birthweight for iron (g per additional 10mg iron per day)
    """
    draw_numbers = draws # Save original values in case we take the mean over draws
#     take_mean = mean_draws_name is not None
    if mean_draws_name is None:
        draws = pd.Index(draws, dtype='int64', name='draw')
    else:
        # Use the single mean value as the only "draw", labeled by `mean_draws_name`
        draws = pd.CategoricalIndex([mean_draws_name], name='draw')

    bw_dose_response_distribution = create_bw_dose_response_distribution()
#     if len(draws)==1:
#         birthweight_dose_response = bw_dose_response_distribution.mean()
#     else:
    # Use our best guess if there's only one draw or we took the mean
    birthweight_dose_response = pd.Series(
        bw_dose_response_distribution.rvs(size=len(draws)) if len(draws)>1 else bw_dose_response_distribution.mean(),
        index=draws,
        name='birthweight_dose_response'
    )
    GlobalIronFortificationData = namedtuple('GlobalIronFortificationData', "draw_numbers, draws, birthweight_dose_response")
    return GlobalIronFortificationData(draw_numbers, draws, birthweight_dose_response)

def get_local_data(location, global_data):
    """
    Information shared between scenarios for a specific location. May vary by draw and location.
    
    Returns
    -------
    location
    iron concentration in flour (mg iron as NaFeEDTA per kg flour)
    ?? mean flour consumption (g per day) -- unneeded if we have mean birthweight shift
    mean birthweight shift (g)
    baseline fortification coverage (proportion of population)
    target fortification coverage (proportion of population)
    """
    iron_concentration = get_iron_concentration(location, global_data.draws)
    # Same mean daily flour for all draws - no parameter uncertainty in flour consumption distribution
    mean_daily_flour = sample_flour_consumption(location, 10_000).mean()
    # Check data dimensions (scalar vs. Series) to make sure multiplication will work
    mean_birthweight_shift = calculate_birthweight_shift(
        global_data.birthweight_dose_response, # indexed by draw
        iron_concentration, # scalar or indexed by draw
        mean_daily_flour # scalar
    ) # returns a Series since global_data.birthweight_dose_response is a Series
    mean_birthweight_shift.rename('mean_birthweight_shift', inplace=True)
    # Load flour coverage data
    # TODO: For now these are scalars, but we can easily add samples from beta distributions indexed by draw
    flour_coverage_df = get_flour_coverage_df()
    eats_fortified = flour_coverage_df.loc[location, ('eats_fortified', 'mean')] / 100
    eats_fortifiable = flour_coverage_df.loc[location, ('eats_fortifiable', 'mean')] / 100
    LocalIronFortificationData = namedtuple('LocalIronFortificationData',
                                            ['location',
                                            'iron_concentration', # scalar or indexed by draw
                                            'mean_daily_flour', # scalar
                                            'mean_birthweight_shift', # indexed by draw
                                            'eats_fortified', # scalar
                                            'eats_fortifiable',] # scalar
                                           )
    return LocalIronFortificationData(location,
                                      iron_concentration,
                                      mean_daily_flour,
                                      mean_birthweight_shift,
                                      eats_fortified,
                                      eats_fortifiable,
                                     )

class IronFortificationIntervention:
    """
    Class for applying iron fortification intervention to simulants.
    """
#     propensity_name = 'iron_fortification_propensity'
    
    def __init__(self, global_data, local_data):
        """Initializes an IronFortificationIntervention with the specified global and local data."""
        self.global_data = global_data
        self.local_data = local_data
#         # OLD VERSION, pre-common random numbers:
#         # __init__(self, location, baseline_coverage, target_coverage)
#         # TODO: Eliminate the distributions in favor of storing a value for each draw (see below)
#         self.iron_conc_distribution = iron_conc_distributions[location]
#         self.bw_dose_response_distribution = create_bw_dose_response_distribution()
#         # TODO: Change constructor to accept the pre-retrieved data instead of looking it up here
#         # OR: Instead, pass baseline and target coverage into the functions below...
#         self.baseline_coverage = baseline_coverage
#         self.target_coverage = target_coverage
        
#         # Currently these distributions are sampling one value for all draws.
#         # TODO: Update to sample a different value for each draw (need to pass draws to constructor).
#         self.dose_response = self.bw_dose_response_distribution.rvs()
#         self.iron_concentration = self.iron_conc_distribution.rvs()

      # TODO: Perhaps create a Population class that can assign the propensities by getting called from here...
#     def assign_propensities(self, pop):
#         """
#         Assigns propensities to simualants for quantities relevant to this intervention.
#         """
#         propensities = np.random.uniform(size=(len(pop),2))
#         pop['iron_fortification_propensity'] = propensities[:,0]
#         pop['mother_flour_consumption_propensity'] = propensities[:,1]

    def get_propensity_names(self):
        """Get the names of the propensities used by this object."""
        return 'iron_fortification_propensity', 'mother_flour_consumption_propensity'

    def assign_treatment_deleted_birthweight(self, pop, lbwsg_distribution, baseline_coverage):
        """
        Assigns "treatment-deleted" birthweights to each simulant in the population,
        i.e. birthweights assuming the absence of iron fortification.
        """
#         # NOTE: We don't necessarily need to sample flour consumption every time if we could
#         # compute the mean ahead of time... I need to think more about which data varies by draw vs. population...
#         flour_consumption = sample_flour_consumption(10_000)
#         mean_bw_shift = calculate_birthweight_shift(self.global_data.dose_response, self.iron_concentration, flour_consumption).mean()
        # Shift everyone's birthweight down by the average shift
#         # Old version that couldn't modify in place:
#         # TODO: actually, maybe we don't need to store the treatment-deleted category, only the treated categories
#         # 2021-03-07: Actually, I now lean towards just recording everything, which is the default in the new lbwsg funcions.
#         shifted_pop = lbwsg_distribution.apply_birthweight_shift(
#             pop, -baseline_coverage * self.local_data.mean_birthweight_shift)
#         pop['treatment_deleted_birthweight'] = shifted_pop['new_birthweight']
        lbwsg_distribution.apply_birthweight_shift(
            pop, -baseline_coverage * self.local_data.mean_birthweight_shift, shifted_col_prefix='treatment_deleted')

    def assign_treated_birthweight(self, pop, lbwsg_distribution, target_coverage):
        """
        Assigns birthweights resulting after iron fortification is implemented.
        Assumes `assign_propensities` and `assign_treatment_deleted_birthweight` have already been called on pop.
        """
        pop['mother_is_iron_fortified'] = pop['iron_fortification_propensity'] < target_coverage
        # DONE: Can this line be rewritten to avoid sampling flour consumption for rows that will get set to 0?
        # Yes, initialize the column with pop['mother_is_iron_fortified'].astype(float),
        # then index to the relevant rows and reassign.
        # NOTE: If we want to compare intervention to baseline by simulant, we can't sample flour consumption
        # every time, because simulants who are fortified in baseline should have the same flour consumption
        # in the intrvention scenario. We'd have to use propensities instead... which is easy because my
        # current sampling function is based on quantiles - I just need to pass it the propensity instead.
        # However, if we don't care about comparability between scenarios, the current implementation should
        # be sufficient for large enough populations.
#         pop['mother_daily_flour'] = pop['mother_is_iron_fortified'] * sample_flour_consumption(len(pop))
        pop['mother_daily_flour'] = pop['mother_is_iron_fortified'].astype(float)
        pop.loc[pop.mother_is_iron_fortified, 'mother_daily_flour'] = get_flour_consumption_from_propensity(
            self.local_data.location,
            pop.loc[pop.mother_is_iron_fortified, 'mother_flour_consumption_propensity']
        )
        # This should broadcast draws of dose response and iron concentration over simulant id's indexing daily flour
        pop['birthweight_shift'] = calculate_birthweight_shift(
            self.global_data.birthweight_dose_response, self.local_data.iron_concentration, pop['mother_daily_flour'])
#         # Old version that couldn't modify in place:
#         shifted_pop = lbwsg_distribution.apply_birthweight_shift(
#             pop, pop['birthweight_shift'], bw_col='treatment_deleted_birthweight')
#         pop['treated_birthweight'] = shifted_pop['new_birthweight']
#         pop['treated_lbwsg_cat'] = shifted_pop['new_lbwsg_cat']
        # Note: The new columns will be called 'treated_treatment_deleted_birthweight' and 'treated_lbwsg_cat'...
        shifted_pop = lbwsg_distribution.apply_birthweight_shift(
            pop, pop['birthweight_shift'], bw_col='treatment_deleted_birthweight', shifted_col_prefix='treated')

