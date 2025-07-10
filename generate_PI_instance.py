import os
import numpy as np
from itertools import product
from pathlib import Path

from instance_parameters import InstanceParameters
from problem.enums import ValueFunction, SpreadDistribution
from problem.portfolio_selection_instance import generate_instance, portfolio_selection_instance_from_file, \
    instance_from_pickle

output_dir = r"C:\Users\liuji\OneDrive - UNSW\Project portfolio\portfolio_optimization-master\instance"
random_seed = 1  # np.random.randint(1, 10000)  # 1

# define the parameters that will characterize the instances
num_projects = [1000]  #  4000, 5000, 6000, 7000, 8000, 9000, 10000
planning_window = [25]  # 20, 25, 30
start_maintain_constraints = [(0.25, 0.75)]  # (0.25, 0.75)
prerequisites = [[(2, 0.5), (3, 0.5)]]  # PI, 1PI, 3PI [(2, 0.50)], [(3, 0.50)]; 2PI [[(2, 0.25), (3, 0.25)]]
exclusions = [[(2, 0.1), (3, 0.1)]]  # 3PI [(2, 0.2), (3, 0.2)]; rest [(2, 0.1), (3, 0.1)]
discounts = [0.01]  # [0.00, 0.01, 0.03]; 0.01

# ensure the output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ensure a directory exists for each number of projects
for num_proj in num_projects:
    Path(os.path.join(output_dir, str(num_proj))).mkdir(parents=True, exist_ok=True)

# the index of the current problem instance, used for an incrementing random seed
index = 1

# generate all permutations of the parameters defined above and create the instances
for n_proj, p_window, sm_const, prereqs, excls, disc in product(num_projects, planning_window,
                                                                start_maintain_constraints,
                                                                prerequisites, exclusions,
                                                                discounts):
    # construct an object that contains the parameter values for this problem instance
    parameters = InstanceParameters(num_projects=n_proj,
                                    planning_window=p_window,
                                    base_budget=n_proj * 15,  # value taken from 2020 FSP
                                    budget_increase=n_proj,  # value taken from 2020 FSP
                                    capability_stream_proportions=np.array([0.346, 0.296, 0.247, 0.074, 0.037]),
                                    # from 2020 FSP
                                    initiation_range=(1, n_proj),
                                    initiation_max_proportion=sm_const[0],
                                    ongoing_max_proportion=sm_const[1],
                                    prerequisite_tuples=prereqs,
                                    exclusion_tuples=excls,
                                    discount_rate=disc,
                                    completion_constraint_chance=0,
                                    value_func=ValueFunction.COST_DUR,
                                    cost_distribution=SpreadDistribution.WEIBULL,
                                    value_distribution=SpreadDistribution.WEIBULL,
                                    )
    # generate a semi-unique name for the problem instance
    instance_name = f"PI_{index}_{n_proj}_{p_window}_{sm_const[0]}_{sm_const[1]}_{disc}"

    instance = generate_instance(parameters, random_seed=index, fuzz=False, identifier=instance_name)
    instance.write_to_pickle(os.path.join(output_dir, str(n_proj), f"{instance_name}.pkl"))

    import pickle
    open_instance = open(os.path.join(output_dir, str(n_proj), f"{instance_name}.pkl"), 'rb')
    instance1 = pickle.load(open_instance)

    instance.write_to_file(os.path.join(output_dir, str(n_proj), f"{instance_name}.dat"))

    index += 1

