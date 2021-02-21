import pandas as pd, numpy as np
from collections import namedtuple

import lbwsg, lsff_interventions
from lbwsg import LBWSGDistribution, LBWSGRiskEffect
from lsff_interventions import IronFortificationIntervention

# Class to store and name the arguments passed to main()
ParsedArgs = namedtuple('ParsedArgs', "location, artifact_path, year, draws, take_mean, num_simulants")

def initialize_population_table(draws, num_simulants):
    """Creates populations for baseline scenario and iron fortification intervention scenario,
    assigns birthweights and gestational ages to each simulant, shifts birthweights appropriately,
    and assigns relative risks for mortality based on resulting LBWSG categories.
    """
    # Create baseline population and assign demographic data
    pop = pd.DataFrame(index=pd.MultiIndex.from_product(
        [draws, range(num_simulants)], names=['draw', 'simulant_id']))
    assign_sex(pop)
    assign_age(pop)
    return pop

def assign_simulant_property(pop, property_name, choice_function=None):
    # Default is to assign uniform propensities
    if choice_function is None:
        choice_function = lambda size: np.random.uniform(size=size)
    simulant_index = pop.index.unique(level='simulant_id')
    simulant_values = pd.Series(choice_function(len(simulant_index)), index=simulant_index, name=property_name)
    # Join simulant values with pop.index to broadcast the same values over all draws
    pop[property_name] = pop[[]].join(simulant_values)

def assign_sex(pop):
#     pop['sex'] = np.random.choice(['Male', 'Female'], size=len(pop))
#     simulant_index = pop.index.unique(level='simulant_id')
#     sexes = pd.Series(np.random.choice(['Male', 'Female'], size=len(simulant_ids)), index=simulant_index, name='sex')
#     pop['sex'] = pop[[]].join(sexes)
    def choose_random_sex(size): return np.random.choice(['Male', 'Female'], size=size)
    assign_simulant_property(pop, 'sex', choose_random_sex)

def assign_age(pop):
    pop['age'] = 0.0 # np.random.uniform(0,28/365, size=num_simulants)
    pop['age_start'] = 0.0
    pop['age_end'] = 7/365
    pop['age_group_id'] = 2 # early neonatal

def assign_propensity(pop, propensity_name):
    """Assigns an independent uniform random number to each simulant.
    Enables sharing randomness across draws and scenarios.
    """
#     pop[propensity_name] = np.random.uniform(size=len(pop))
    assign_simulant_property(pop, propensity_name, choice_function=None)

def assign_propensities(pop, propensity_names):
    """Assigns propensities for each name in a list of propensity names.
    """
    for propensity_name in propensity_names:
        assign_simulant_property(pop, propensity_name)

class IronBirthweightCalculator:
    """Class to run nanosimulations for the effect of iron on low birthweight."""
    
    treated_lbwsg_rr_colname = 'treated_lbwsg_rr'

    def __init__(self, location, artifact_path, year, draws, take_mean=False):
        """
        """
        # Save input parameters so we can look them up later if necessary/desired
        self.location = location
        self.artifact_path = artifact_path
        self.year = year
        self.draws = draws # These will also be stored in global_data, unless take_mean is True
        
        # TODO: Perhaps create and save a numpy random generator, and share it via global_data
        
        if take_mean:
            mean_colname = f'mean_of_{len(draws)}_draws'
            draws = [mean_colname]
            
        # Load iron intervention data
#         flour_coverage_df = lsff_interventions.get_flour_coverage_df()
#         baseline_coverage = flour_coverage_df.loc[location, ('eats_fortified', 'mean')] / 100
#         intervention_coverage = flour_coverage_df.loc[location, ('eats_fortifiable', 'mean')] / 100
        self.global_data = lsff_interventions.get_global_data(draws)
        self.local_data = lsff_interventions.get_local_data(location, self.global_data)
        
        # Load LBWSG data
        exposure_data = lbwsg.read_lbwsg_data(
            artifact_path, 'exposure', "age_end < 1", f"year_start == {year}", draws=self.draws)
        rr_data = lbwsg.read_lbwsg_data(
            artifact_path, 'relative_risk', "age_end < 1", f"year_start == {year}", draws=self.draws)
        
        if take_mean:
            exposure_data = exposure_data.mean(axis=1).rename(mean_colname).to_frame()
            rr_data = rr_data.mean(axis=1).rename(mean_colname).to_frame()

        # Create model components
        self.lbwsg_distribution = LBWSGDistribution(exposure_data)
        self.lbwsg_effect = LBWSGRiskEffect(rr_data, paf_data=None) # We don't need PAFs to initialize the pop tables with RR's

#         self.baseline_fortification = IronFortificationIntervention(location, baseline_coverage, baseline_coverage)
#         self.intervention_fortification = IronFortificationIntervention(location, baseline_coverage, intervention_coverage)
        self.iron_intervention = IronFortificationIntervention(self.global_data, self.local_data)
    
        # Declare variables for baseline and intervention populations,
        # which will be initialized in initialize_population_tables
        self.baseline_pop = None
        self.intervention_pop = None
        self.potential_impact_fraction = None

    def initialize_population_tables(self, num_simulants):
        """Creates populations for baseline scenario and iron fortification intervention scenario,
        assigns birthweights and gestational ages to each simulant, shifts birthweights appropriately,
        and assigns relative risks for mortality based on resulting LBWSG categories.
        """
        # Create baseline population and assign demographic data
        self.baseline_pop = initialize_population_table(self.global_data.draws, num_simulants)

        # Assign propensities to share between scenarios
#         assign_propensity(self.baseline_pop, IronFortificationIntervention.propensity_name)
        self.lbwsg_distribution.assign_propensities(self.baseline_pop)
        self.iron_intervention.assign_propensities(self.baseline_pop)

#         # Assign baseline exposure - ideally this would be done with a propensity to share between scenarios,
#         # but that's more complicated to implement, so I'll just copy the table after assigning lbwsg exposure.
#         self.lbwsg_distribution.assign_exposure(self.baseline_pop)
        
        # Create intervention population - all the above data will be the same in intervention scenario
        self.intervention_pop = self.baseline_pop.copy()
        
        # Reset PIF to None until we're ready to recompute it
        self.potential_impact_fraction = None

    def assign_lbwsg_exposure(self):
        # Assign baseline exposure - ideally this would be done with a propensity to share between scenarios,
        # but that's more complicated to implement, so I'll just copy the table after assigning lbwsg exposure.
        self.lbwsg_distribution.assign_exposure(self.baseline_pop)
        self.intervention_pop = self.baseline_pop.copy() # Hack to deal with slow assign_exposure function
        
    def assign_iron_treatment_deleted_birthweights(self):  
        # Apply the birthweight shifts in baseline and intervention scenarios:
        # First comput treatment-deleted birthweight, then birthweight with iron fortification.
#         self.baseline_fortification.assign_treatment_deleted_birthweight(baseline_pop, self.lbwsg_distribution)
#         self.intervention_fortification.assign_treatment_deleted_birthweight(intervention_pop, self.lbwsg_distribution)
        self.iron_intervention.assign_treatment_deleted_birthweight(
            self.baseline_pop, self.lbwsg_distribution, self.local_data.eats_fortified)
        self.iron_intervention.assign_treatment_deleted_birthweight(
            self.intervention_pop, self.lbwsg_distribution, self.local_data.eats_fortified)

    def assign_iron_treated_birthweights(self):
#         self.baseline_fortification.assign_treated_birthweight(baseline_pop, self.lbwsg_distribution)
#         self.intervention_fortification.assign_treated_birthweight(intervention_pop, self.lbwsg_distribution)
        self.iron_intervention.assign_treated_birthweight(
            self.baseline_pop, self.lbwsg_distribution, self.local_data.eats_fortified)
        self.iron_intervention.assign_treated_birthweight(
            self.intervention_pop, self.lbwsg_distribution, self.local_data.eats_fortifiable)

    def assign_lbwsg_relative_risks(self):
        # Compute the LBWSG relative risks in both scenarios - these will be used to compute the PIF
        # TODO: Maybe have lbwsg return the RR values instead, and assign them to the appropriate column here
        self.lbwsg_effect.assign_relative_risk(self.baseline_pop, cat_colname='treated_lbwsg_cat')
        self.lbwsg_effect.assign_relative_risk(self.intervention_pop, cat_colname='treated_lbwsg_cat')

#         return namedtuple('InitPopTables', 'baseline, iron_fortification')(baseline_pop, intervention_pop)
    
    def calculate_potential_impact_fraction(self):
        self.potential_impact_fraction = potential_impact_fraction(
            self.baseline_pop, self.intervention_pop, 'lbwsg_relative_risk')
        
    def do_back_of_envelope_calculation(self, num_simulants):
        """
        """
        self.initialize_population_tables(num_simulants)
        self.assign_lbwsg_exposure()
        self.assign_iron_treatment_deleted_birthweights()
        self.assign_iron_treated_birthweights()
        self.assign_lbwsg_relative_risks()
        self.calculate_potential_impact_fraction()

# def initialize_population_table(num_simulants, draws):
#     """
#     Initialize a population table with a gestational age and birthweight for each simulant.
#     """
#     simulant_ids = range(num_simulants)
#     pop = pd.DataFrame(index=pd.MultiIndex.from_product([simulant_ids, draws],names=['simulant_id', 'draw']))
#     assign_sex(pop)
#     assign_age(pop)
    
#     lbwsg_distribution.assign_exposure(pop)
#     return pop

    
def potential_impact_fraction(baseline_pop, counterfactual_pop, rr_colname):
    """Computes the population impact fraction for the specified baseline and counterfactual populations."""
    baseline_mean_rr = baseline_pop.groupby('draw')[rr_colname].mean()
    counterfactual_mean_rr = counterfactual_pop.groupby('draw')[rr_colname].mean()
    return (baseline_mean_rr - counterfactual_mean_rr) / baseline_mean_rr

def parse_args(args):
    """"""
    if len(args)>0:
        # Don't do any parsing for now, just make args into a named tuple
        args = ParsedArgs._make(args)
    else:
        # Hardcode some values for testing
        location = "Nigeria"
        artifact_path = f'/share/costeffectiveness/artifacts/vivarium_conic_lsff/{location.lower()}.hdf'
        year=2017
        draws = [0,50,100]
        take_mean = False
        num_simulants = 10
        args = ParsedArgs(location, artifact_path, year, draws, take_mean, num_simulants)
    return args

def main(args=None):
    """
    Does a back of the envelope calculation for the given arguments
    """
    if args is None:
        args = sys.argv[1:]
        
    args = parse_args(args)
    sim = IronBirthweightCalculator(args.location, args.artifact_path, args.year, args.draws, args.take_mean)
    baseline_pop, intervention_pop = sim.initialize_population_tables(args.num_simulants)
    pif = population_impact_fraction(baseline_pop, intervention_pop, IronBirthweightNanoSim.treated_lbwsg_rr_colname)
    # do something with pif...

