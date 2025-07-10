import os
import numpy as np
from pathlib import Path

from instance_parameters import InstanceParameters
from problem.enums import ValueFunction, SpreadDistribution
from problem.portfolio_selection_instance import generate_instance


def generate_ppssp_instance(output_dir, index, n_proj, p_window, start_budget_prop, discounts):
    maintain_budget_prop = 1 - start_budget_prop
    prerequisite = [(2, 0.5), (3, 0.5)]
    exclusion = [(2, 0.1), (3, 0.1)]

    # ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # generate all permutations of the parameters defined above and create the instances
    # construct an object that contains the parameter values for this problem instance
    parameters = InstanceParameters(num_projects=n_proj,
                                    planning_window=p_window,
                                    base_budget=n_proj * 15,  # value taken from 2020 FSP
                                    budget_increase=n_proj,  # value taken from 2020 FSP
                                    capability_stream_proportions=np.array([0.346, 0.296, 0.247, 0.074, 0.037]),
                                    # from 2020 FSP
                                    initiation_range=(1, n_proj),
                                    initiation_max_proportion=start_budget_prop,
                                    ongoing_max_proportion=maintain_budget_prop,
                                    prerequisite_tuples=prerequisite,
                                    exclusion_tuples=exclusion,
                                    discount_rate=discounts,
                                    completion_constraint_chance=0,
                                    value_func=ValueFunction.COST_DUR,
                                    cost_distribution=SpreadDistribution.WEIBULL,
                                    value_distribution=SpreadDistribution.WEIBULL,
                                    )
    # generate a semi-unique name for the problem instance
    instance_name = f"PI_{index}_{n_proj}_{p_window}_{start_budget_prop}_{discounts}"

    instance = generate_instance(parameters, random_seed=index, fuzz=False, identifier=instance_name)
    instance.write_to_pickle(os.path.join(output_dir, f"{instance_name}.pkl"))
    instance.write_to_file(os.path.join(output_dir, f"{instance_name}.dat"))



