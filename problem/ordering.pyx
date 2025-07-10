import numpy as np

from problem.enums import PortfolioOrdering

def cv_ratio(projects):
    num_projects = len(projects)
    values = np.zeros(num_projects)
    for i in range(num_projects):
        values[i] = projects[i].total_cost / projects[i].total_value

    return values

def cost_ascending(projects):
    num_projects = len(projects)
    values = np.zeros(num_projects)
    for i in range(num_projects):
        values[i] = -projects[i].total_cost

    return values

def cost_descending(projects):
    num_projects = len(projects)
    values = np.zeros(num_projects)
    for i in range(num_projects):
        values[i] = -projects[i].total_cost

    return values

def get_ordering(projects, ordering):
    switcher = {
        PortfolioOrdering.CV_RATIO: cv_ratio,
        PortfolioOrdering.COST_ASC: cost_ascending,
        PortfolioOrdering.COST_DESC: cost_descending,
    }

    func = switcher.get(ordering)
    values = func(projects)
    return np.argsort(values)
