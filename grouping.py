# Jing Liu
import os
import pickle
import math
import numpy as np
from itertools import product
from pathlib import Path


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
