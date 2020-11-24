import pandas as pd, numpy as np
import lbwsg
from collections import namedtuple

# Class to store and name the arguments passed to main()
Args = namedtuple('Args', "location, artifact_path, year, draws, num_simulants")

def do_back_of_envelope_calculation(artifact_path, year, draws, num_simulants):
    """
    """
    exposure_data = lbwsg.read_lbwsg_data(
        artifact_path, 'exposure', "age_end < 1", "year_start == @year", draws=draws)
    lbwsg_distribution = LBWSGDistribution(exposure_data)
    lbwsg_effect = LBWSGRiskEffect(rr_data, paf_data)
    bw_shift_distribution = BWShiftDistribution(data_for_distribution) # depends on location
    # responsible for determining who is covered and what their corresponding birthweight shift will be
    iron_intervention = IronFortificationIntervention(coverage_data, bw_shift_distribution)

    # Create baseline population and assign demographic data
#     baseline_pop = initialize_population_table(num_simulants, draws)
#     simulant_ids = range(num_simulants)
    baseline_pop = pd.DataFrame(index=pd.MultiIndex.from_product(
        [range(num_simulants), draws], names=['simulant_id', 'draw']))
    assign_sex(baseline_pop)
    assign_age(baseline_pop)

    # Assign baseline exposure
    lbwsg_distribution.assign_exposure(baseline_pop)
    baseline_coverage = iron_intervention.baseline_coverage_proportion()
    mean_bw_shift = bw_shift_distribution.mean()
    lbwsg_distribution.assign_treatment_deleted_birthweight(baseline_pop, baseline_coverage, mean_bw_shift)
    iron_intervention.assign_fortification_propensity(baseline_pop)
    
    # Create intervention population - all the above data will be the same in intervention
    intervention_pop = baseline_pop.copy()
    
    # For each scenario
    for pop in (baseline_pop, intervention_pop):
        iron_intervention.assign_birthweight_shifts(pop)
        lbwsg_distribution.shift_birthweights(pop)
        lbwsg_effect.assign_relative_risks(pop)
        # Now calculate csmr's...
    
    # Create intervention population and do calculations
    intervention_pop = pop.copy()
    bw_shift_distribution.assign_shifts(intervention_pop)
    lbwsg_distribution.shift_birthweights(intervention_pop, bw_shift_distribution)
    lbwsg_effect.assign_relative_risks(intervention_pop)
    # Now calculate csmr's...
    
    
    # Finally, calculate reduction in mortality...


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

def assign_sex(pop):
    pop['sex'] = np.random.choice(['Male', 'Female'], size=len(pop))

def assign_age(pop):
    pop['age'] = 0 # np.random.uniform(0,28/365, size=num_simulants)
    pop['age_start'] = 0
    pop['age_end'] = 7/365
    
def parse_args(args):
    """"""
    if len(args)>0:
        # Don't do any parsing for now, just make args into a named tuple
        args = Args._make(args)
    else:
        # Hardcode some values for testing
        location = "nigeria"
        artifact_path = f'/share/costeffectiveness/artifacts/vivarium_conic_lsff/{location}.hdf'
        year=2017
        draws = [0,50,100]
        num_simulants = 10
        args = Args(location, artifact_path, year, draws, num_simulants)
    return args

def main(args=None):
    """
    Does a back of the envelope calculation for the given arguments
    """
    if args is None:
        args = sys.argv[1:]
        
    args = parse_args(args)
    do_back_of_envelope_calculation(args.artifact_path, args.year, args.draws, args.num_simulants)

