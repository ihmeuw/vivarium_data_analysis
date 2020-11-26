import pandas as pd, numpy as np
from scipy import stats

import sys, os.path
sys.path.append(os.path.abspath("../.."))
from probability import prob_utils
from plots_and_other_misc import lsff_plots

# Define distributions of iron concentration in fortified flour for each country,
# representing parameter uncertainty.
# Units are mg iron as NaFeEDTA per kg flour
# Currently we use a uniform distribution for India and degenerate (point mass)
# distributions for Ethiopia and Nigeria.
# Eventually this should be stored in some external data source that will then be loaded.
iron_conc_distributions = {
    'India': stats.uniform(loc=14, scale=21.5-14), # Uniform(14,21.5) mg iron as NaFeEDTA per kg flour
    'Ethiopia': stats.bernoulli(p=0,loc=30) # 30 mg iron as NaFeEDTA per kg flour,
    'Nigeria': stats.bernoulli(p=0,loc=40) # 40 mg iron as NaFeEDTA per kg flour,
}

def sample_flour_consumption(sample_size):
    """Sample from distribution of daily flour consumption (in Ethiopia).
    The distribution is uuniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
    # Define quartiles in g of flour per day
    q = (0, 77.5, 100, 200, 350.5) # min=0, q1=77.5, q2=100, q3=200, max=350.5
    u = np.random.uniform(0,1,size=sample_size)
    # Scale the uniform random number to the correct interval based on its quartile
    return np.select(
        [u<0.25, u<0.5, u<0.75, u<1],
        [q[1]*u/0.25, q[1]+(q[2]-q[1])*(u-0.25)/0.25, q[2]+(q[3]-q[2])*(u-0.5)/0.25, q[3]+(q[4]-q[3])*(u-0.75)/0.25]
    )

def create_bw_dose_response_distribution(self):
    """Define normal distribution representing parameter uncertainty of dose-response on birthweight.
    mean = 15.1 g per 10 mg daily iron, 95% CI = (6.0,24.2).
    Effect size comes from Haider et al. (2013)
    """
    # mean and 0.975-quantile of normal distribution for mean difference (MD)
    mean = 15.1 # g per 10 mg daily iron
    q_975 = 24.2 # 97.5th percentile
    std = prob_utils.normal_stdev_from_mean_quantile(mean, q_975, 0.975)
    # Frozen normal distribution for MD, representing uncertainty in our effect size
    return stats.norm(mean, std)

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

class IronFortificationIntervention:
    """
    Class for applying iron fortification intervention to simulants.
    """
    propensity_name = 'iron_fortification_propensity'
    
    def __init__(self, location, basline_coverage_key, target_coverage_key):
        # TODO: Eliminate the distributions in favor of storing a value for each draw (see below)
        self.iron_conc_distribution = iron_conc_distributions[location]
        self.bw_dose_response_distribution = create_bw_dose_response_distribution()
        # TODO: Change constructor to accept the pre-retrieved data instead of looking it up here
        self.baseline_coverage = coverage_df.loc[location, basline_coverage_key]
        self.target_coverage = coverage_df.loc[location, target_coverage_key]
        
        # Currently these distributions are sampling one value for all draws.
        # TODO: Update to sample a different value for each draw (need to pass draws to constructor).
        self.dose_response = self.bw_dose_response_distribution.rvs()
        self.iron_concentration = self.iron_conc_distribution.rvs()
    
    def assign_treatment_deleted_birthweight(self, pop, lbwsg_distribution):
        """
        Assigns "treatment-deleted" birthweights to each simulant in the population.
        """
        flour_consumption = sample_flour_consumption(10_000)
        mean_bw_shift = calculate_birthweight_shift(self.dose_response, self.iron_conc, flour_consumption).mean()
        # Shift everyone's birthweight down by the average shift
        # TODO: actually, maybe we don't need to store the treatment-deleted category, only the treated categories
        pop.loc[:,['treatment_deleted_birthweight', 'treatment_deleted_lbwsg_cat']] = \
            lbwsg_distribution.apply_birthweight_shift(pop, -self.baseline_coverage * mean_bw_shift).values
        
    def assign_treated_birthweight(self, pop, lbwsg_distribution):
        """
        Assigns birthweights resulting after iron fortification is implemented.
        """
        pop['mother_is_iron_fortified'] = pop[propensity_name] < self.target_coverage
        # TODO: Can this line be rewritten to avoid sampling flour consumption for rows that will get set to 0?
        # Yes, initialize the column with pop['mother_is_iron_fortified'].astype(float),
        # then index to the relevant rows and reassign.
        pop['mother_daily_flour'] = pop['mother_is_iron_fortified'] * sample_flour_consumption(len(pop))
        pop['birthweight_shift'] = calculate_birthweight_shift(self.dose_response, self.iron_conc, pop['mother_daily_flour'])
        pop['treated_birthweight'] = lbwsg_distribution.apply_birthweight_shift(pop, pop['birthweight_shift'])
        pop['treated_lbwsg_cat'] = 5 #FIXME
        
    
    
    
    
    

        
        