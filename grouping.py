# Jing Liu
import os
import pickle
import math
import numpy as np
from itertools import product
from pathlib import Path


class Groups:
    def __init__(self, seps, non_groups):
        self.seps = seps
        self.non_groups = non_groups


def grouping(size_groups=500):
    np.random.seed(int(size_groups/100))
    base_dir = r"C:\Users\liuji\OneDrive - UNSW\Project portfolio\portfolio_optimization-master"
    instances_dir = r"C:\Users\liuji\OneDrive - UNSW\Project portfolio\portfolio_optimization-master\instance"
    output_base = os.path.join(base_dir, "Output", "Test")
    # instances_dir = r"C:\Users\z5159104\OneDrive - UNSW\Project portfolio\portfolio_optimization-master\instance"
    # instances_dir = r"/home/549/jl7122/portfolio_optimization/instance"
    num_projects = [9000, 10000]   # , 5000, 6000, 8000, 10000
    planning_window = [30]
    start_maintain_constraints = [(0.25, 0.75)]
    prerequisites = [[(2, 0.50), (3, 0.50)]]  # PI, 1PI [(2, 0.50)], [(3, 0.50)]; 2PI [[(2, 0.25), (3, 0.25)]]
    exclusions = [[(2, 0.1), (3, 0.1)]]
    discounts = [0.01]

    for n_proj, p_window, sm_const, prereqs, excls, disc in product(num_projects, planning_window,
                                                                    start_maintain_constraints,
                                                                    prerequisites, exclusions,
                                                                    discounts):
        instance_name = f"1PI_{n_proj}_{p_window}_{sm_const[0]}_{sm_const[1]}_{disc}"
        open_instance = open(os.path.join(instances_dir, str(n_proj), f"{instance_name}.pkl"), 'rb')
        instance = pickle.load(open_instance)
        print(f"Grouping instance {instance_name}.")
        # initial problem-dependent decomposition
        grouping_results = problem_dependent_random_grouping(instance, size_groups)

        Path(os.path.join(instances_dir, str(n_proj))).mkdir(parents=True, exist_ok=True)

        file_name = os.path.join(instances_dir, str(n_proj), f"{instance_name}_grouping_{size_groups}.pkl")

        with open(file_name, "wb") as f:
            pickle.dump(grouping_results, f)

        file_name = os.path.join(output_base, instance_name, f"grouping_{size_groups}.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(grouping_results, f)

        # grouping_results1 = pickle.load(open(file_name, "rb"))

def problem_dependent_grouping(instance, size_groups):
    dim = len(instance.projects)
    num_groups = math.ceil(dim / size_groups)
    x_remain = list(range(0, dim))
    seps = []
    non_groups = []
    x_visited = []

    # grouping according to the exclusion, prerequisite and successor relationships
    while len(x_remain) > 0:
        sub1 = x_remain[0]
        x_to_visit = [sub1]

        while len(x_to_visit) > 0:
            x_to_visit, sub1, x_remain, x_visited = find_interacts(instance, sub1, x_to_visit, x_remain, x_visited)

        if len(sub1) == 1:
            seps.append(sub1[0])
        else:
            non_groups.append(sub1)

        if len(x_remain) == 1:
            seps.append(x_remain[0])
            break

    # grouping_results = Groups(seps, non_groups)
    cnum_groups = len(seps) + len(non_groups)
    if len(seps) != 0:
        for i in range(len(seps)):
            non_groups.append(np.array([seps[i]]))

    # grouping according to the capability streams
    if cnum_groups > num_groups:
        groups = []
        group_capability = []
        capability_index = [[] for _ in range(len(non_groups))]
        dominant_capability = np.empty(len(non_groups))
        for i in range(len(non_groups)):
            for j in range(len(non_groups[i])):
                capability_index[i].append(instance.projects[non_groups[i][j]].capability_stream)
            dominant_capability[i] = max(capability_index[i], key=capability_index[i].count)

        groups_remain = list(range(0, len(non_groups)))
        while len(groups_remain) > 0:
            group1, groups_remain, current_capability = find_interacts_cap(non_groups, capability_index,
                                                                           dominant_capability, groups,
                                                                           groups_remain, size_groups)

            groups.append(group1)
            group_capability.append(current_capability)

    # merge the small groups
    i = 0
    while True:
        j = i + 1
        while True:
            if len(groups[i]) + len(groups[j]) < size_groups:
                groups[i].extend(groups[j])

                del groups[j]
            else:
                j += 1

            if j > len(groups) - 1:
                break
        i += 1
        if i > len(groups) - 2:
            break

    # sort the index in groups
    grouping_results = groups
    for i in range(len(groups)):
        grouping_results[i] = np.array(groups[i])
        grouping_results[i].sort()

    return grouping_results


def problem_dependent_random_grouping(instance, size_groups):
    dim = len(instance.projects)
    num_groups = math.ceil(dim / size_groups)
    x_remain = list(range(0, dim))
    seps = []
    non_groups = []
    x_visited = []

    # grouping according to the exclusion, prerequisite and successor relationships
    while len(x_remain) > 0:
        sub1 = x_remain[0]
        x_to_visit = [sub1]

        while len(x_to_visit) > 0:
            x_to_visit, sub1, x_remain, x_visited = find_interacts(instance, sub1, x_to_visit, x_remain, x_visited)

        if len(sub1) == 1:
            seps.append(sub1[0])
        else:
            non_groups.append(sub1)

        if len(x_remain) == 1:
            seps.append(x_remain[0])
            break

    # grouping_results = Groups(seps, non_groups)
    cnum_groups = len(seps) + len(non_groups)
    if len(seps) != 0:
        for i in range(len(seps)):
            non_groups.append(np.array([seps[i]]))
    groups = non_groups
    for i in range(len(non_groups)):
        groups[i] = non_groups[i].tolist()

    # merge the small groups randomly
    I = np.random.permutation(range(0, len(groups)))
    _groups = [groups[i] for i in I]
    i = 0

    while True:
        # j = np.random.randint(i + 1, len(groups) + 1)
        # if j == i:
        #     raise ValueError("j = i")
        j = i + 1
        while True:
            if len(_groups[i]) + len(_groups[j]) < size_groups:
                _groups[i].extend(_groups[j])

                del _groups[j]
            else:
                j += 1

            if j > len(_groups) - 1:
                break
        i += 1
        if i > len(_groups) - 2:
            break

    # sort the index in groups
    grouping_results = _groups
    for i in range(len(_groups)):
        grouping_results[i] = np.array(_groups[i])
        grouping_results[i].sort()

    return grouping_results


def find_interacts_cap(non_groups, capability_index, dominant_capability, groups, groups_remain, size_groups):
    group1 = []
    group1.extend(non_groups[groups_remain[0]])
    current_capability = dominant_capability[groups_remain[0]]
    del groups_remain[0]
    i = 0
    while len(group1) < size_groups:
        if dominant_capability[groups_remain[i]] == current_capability:
            group1.extend(non_groups[groups_remain[i]])
            del groups_remain[i]
        else:
            i += 1

        if i > len(groups_remain) - 1:
            break

    # capability_stream_set = np.empty(len(instance.projects))
    # for i in range(len(instance.projects)):
    #     capability_stream_set[i] = instance.projects[i].capability_stream
    # num_cap = int(np.max(capability_stream_set))
    # num_projects_cap = np.empty(num_cap)
    # for i in range(num_cap):
    #     num_projects_cap[i] = len(np.where(capability_stream_set == i)[0])

    return group1, groups_remain, current_capability


def find_interacts(instance, sub1, x_to_visit, x_remain, x_visited):
    var1 = int(x_to_visit[0])
    project = instance.projects[var1]
    interact_vars = []
    interact_vars = np.union1d(interact_vars, project.exclusion_list)
    interact_vars = np.union1d(interact_vars, project.prerequisite_list)
    interact_vars = np.union1d(interact_vars, project.successor_list).astype(int)
    # print(var1)
    x_to_visit.remove(var1)
    x_visited.append(var1)
    x_remain.remove(var1)
    if len(interact_vars) > 0:
        for i in range(len(interact_vars)):
            if interact_vars[i] not in x_visited and interact_vars[i] not in x_to_visit:
                x_to_visit.append(interact_vars[i])
    sub1 = np.union1d(sub1, interact_vars)

    return x_to_visit, sub1, x_remain, x_visited


# for size_groups in [400, 500, 600, 700, 800, 900]:
#     grouping(size_groups)
