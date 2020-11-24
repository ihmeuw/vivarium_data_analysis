import pandas as pd, numpy as np
import lbwsg

lbwsg_distribution = None

def do_back_of_envelope_calculation(artifact_path, num_simulants, draws):
    global lbwsg_distribution
    
    exposure_data = lbwsg.read_lbwsg_data(
        artifact_path, 'exposure', "year_start == 2017", "age_end < 1", draws=draws
    )
    lbwsg_distribution = LBWSGDistribution(exposure_data)
    lbwsg_effect = LBWSGRiskEffect(rr_data, paf_data)
    bw_shift_distribution = BWShiftDistribution(data_for_distribution)
    
    # Create baseline population and do calculations
    baseline_pop = initialize_population_table(num_simulants, draws)
    
    lbwsg_distribution.stratify_baseline_pop(baseline_pop, bw_shift_distribution)
    lbwsg_effect.assign_relative_risks(baseline_pop)
    # Now calculate csmr's...
    
    # Create intervention population and do calculations
    intervention_pop = pop.copy()
    bw_shift_distribution.assign_shifts(intervention_pop)
    lbwsg_distribution.shift_birthweights(intervention_pop, bw_shift_distribution)
    lbwsg_effect.assign_relative_risks(intervention_pop)
    # Now calculate csmr's...
    
    
    # Finally, calculate reduction in mortality...


def initialize_population_table(num_simulants, draws):
    """
    Initialize a population table with a gestational age and birthweight for each simulant.
    """
    simulant_ids = range(num_simulants)
    pop = pd.DataFrame(index=pd.MultiIndex.from_product([simulant_ids, draws],names=['simulant_id', 'draw']))
    assign_sex(pop)
    assign_age(pop)
    
    lbwsg_distribution.assign_exposure(pop)
    return pop

def assign_sex(pop):
    pop['sex'] = np.random.choice(['Male', 'Female'], size=len(pop))

def assign_age(pop):
    pop['age'] = 0 # np.random.uniform(0,28/365, size=num_simulants)
    pop['age_start'] = 0
    pop['age_end'] = 7/365
    
def parse_args(args):
    """"""
    location = "nigeria"
    artifact_path = f'/share/costeffectiveness/artifacts/vivarium_conic_lsff/{location}.hdf'
    num_simulants = 10
    draws = [0,50,100]
    return artifact_path, num_simulants, draws

def main(args=None):
    """
    Does a back of the envelope calculation for the given arguments
    """
    if args is None:
        args = sys.argv[1:]
        
    artifact_path, num_simulants, draws = parse_args(args)
    do_back_of_envelope_calculation(artifact_path, num_simulants, draws)