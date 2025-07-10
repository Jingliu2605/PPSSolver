import os
import pathlib
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy.matlib import empty
from scipy import stats

from gantt_chart import Gantt
import gurobi_logtools as glt
from problem.portfolio import portfolio_from_pickle
from problem.portfolio_selection_instance import instance_from_pickle

def get_portfolio_name(output_dir, instance_name, algorithm_name, t_limitation=600, mip_gap=0.01, size_groups=600, run=1):
    """
    Get the portfolio name for a specific algorithm and instance.
    """
    result_path = get_result_path(output_dir, instance_name, algorithm_name, t_limitation=t_limitation, mip_gap=mip_gap, size_groups=size_groups)
    if algorithm_name == 'Gurobi':
        return os.path.join(result_path, f"{algorithm_name}_portfolio.pkl")
    else:
        return os.path.join(result_path, f"Portfolio_{run+1}.pkl")

def get_convergence_file_name(output_dir, instance_name, algorithm_name, t_limitation=None, mip_gap=0.01, size_groups=None, run=None):
    """
    Get the convergence file name for a specific algorithm and instance.
    """
    result_path = get_result_path(output_dir, instance_name, algorithm_name, t_limitation=t_limitation, mip_gap=mip_gap, size_groups=size_groups)
    if algorithm_name == 'Gurobi':
        return os.path.join(result_path, f"{algorithm_name}_{instance_name}.csv")
    else:
        return os.path.join(result_path, f"convergence_data_run_{run}.csv")

def get_result_path(output_dir, instance_name, algorithm_name, t_limitation=None, mip_gap=0.01, size_groups=None):
    """
    Get the result path for a specific algorithm and instance.
    """
    if algorithm_name == 'Gurobi':
        return os.path.join(output_dir, instance_name, f"{algorithm_name}_{t_limitation}_{mip_gap}")
    elif algorithm_name == 'HEGCL':
        return os.path.join(output_dir, instance_name, f"{algorithm_name}_{size_groups}_{t_limitation}")
    else:
        return os.path.join(output_dir, instance_name, f"{algorithm_name}")

def get_experimental_data(algorithms, instance_name, output_dir, analysis_type, runs=1, pause_event=None, stop_event=None):
    """
    Compare different algorithms on the same project file.
    :param algorithms: list of tuples (algorithm_name, algorithm_parameters)
    :param instance_name: instance name
    :param output_dir: directory to save the results
    :param pause_event: event to pause the execution
    :param stop_event: event to stop the execution
    """

    fitness = np.zeros([len(algorithms), runs])
    consumed_time = np.zeros([len(algorithms), runs])
    for alg_index, alg in enumerate(algorithms):
        if alg == 'Gurobi':
            time_history, fitness_history, best = read_gurobi(instance_name, output_dir, t_limitation=600, mip_gap=0.01)
            fitness[alg_index, :] = fitness_history[-1]
            consumed_time[alg_index, :] = time_history[-1]
        elif alg == 'HEGCL':
            size_groups = 600
            t_limitation = 200
            for i in range(runs):
                alg_instance_file = os.path.join(output_dir, instance_name, f"{alg}_{size_groups}_{t_limitation}",
                                                        f"convergence_data_run_{i + 1}.csv")
                time_history, fitness_history = read_convergence_data(alg_instance_file)
                fitness[alg_index, i] = fitness_history[-1]
                consumed_time[alg_index, i] = time_history[-1]
        else:
            for i in range(runs):
                alg_instance_file = os.path.join(output_dir, instance_name, f"{alg}",
                                                        f"convergence_data_run_{i + 1}.csv")
                time_history, fitness_history = read_convergence_data(alg_instance_file)
                fitness[alg_index, i] = fitness_history[-1]
                consumed_time[alg_index, i] = time_history[-1]
    if analysis_type == "Compare Portfolio Values and Runtimes":
        return np.mean(fitness, 1), np.std(fitness, 1), np.mean(consumed_time, 1), np.std(consumed_time, 1)
    elif analysis_type == "Visualize Convergence Graphs":
        return time_history, fitness_history

def read_convergence_data(alg_convergence_file):
    alg_data = pd.read_csv(alg_convergence_file)
    fitness_history = alg_data.iloc[:, 2].to_numpy()
    time_history = alg_data.iloc[:, 3].to_numpy()
    return time_history, fitness_history

def read_gurobi(instance_name, output_dir, t_limitation=60, mip_gap=0.01):
    gurobi_logfile = os.path.join(output_dir, instance_name, f"Gurobi_{t_limitation}_{mip_gap}",
                                         f"Gurobi_{instance_name}.csv")
    data = read_gurobi_logfile(gurobi_logfile)
    time_history, fitness_history = extract_data_gurobi_logfile(data)
    best = fitness_history[-1]
    return time_history, fitness_history, best

def read_gurobi_logfile(filename):
    if not os.path.exists(filename):
        log_filename = filename.removesuffix(".csv")
        gurobi_logfile_2_csv(log_filename)

    data = pd.read_csv(filename)
    data = data.to_numpy()
    return data

def gurobi_logfile_2_csv(filename):
    """
    Converts the Gurobi log file to a CSV file.
    :param filename: path to the Gurobi log file
    """
    # parse the Gurobi log file and write to CSV
    results = glt.parse([filename + ".log"])
    nodelog_progress = results.progress("nodelog")
    nodelog_progress.to_csv(filename + ".csv", index=False)

def extract_data_gurobi_logfile(data, consumed_time=0):
    # gap_history = data[:, 7]
    gurobi_time = data[:, 9] + consumed_time
    gurobi_fitness = data[:, 5]

    return gurobi_time, gurobi_fitness


def analyze_from_pickles(portfolio_pickle, instance_pickle, identifier, display=True, plot=True, plot_dir=None,
                         yearly_project_analysis=True):
    portfolio = portfolio_from_pickle(portfolio_pickle)
    instance = instance_from_pickle(instance_pickle)

    analyze_portfolio(portfolio, instance, identifier, display, plot, plot_dir, yearly_project_analysis)


def summarize_from_results(result_dir, summary_file, instance_identifier):
    alg_file = os.path.join(result_dir, "results.csv")
    with open(summary_file, "w") as result_file:
        result_file.write("Instance, Mean Fit, Std., Min, Mean Error, Std., Min, Mean Bound Error, Std., Min\n")
    # pathlib.Path(summary_file).mkdir(parents=True, exist_ok=True)

    alg_data = pd.read_csv(alg_file)
    fitnesses = alg_data[" Fitness"]
    errors = alg_data[" Error"]
    error_bounds = alg_data[" Error Bound"]

    with open(summary_file, "a") as result_file:
        result_file.write(
            f"{instance_identifier}, {np.mean(fitnesses)}, {np.std(fitnesses)}, {np.min(fitnesses)}, {np.mean(errors)}, "
            f"{np.std(errors)}, {np.min(errors)}, {np.mean(error_bounds)}, {np.std(error_bounds)}, {np.min(error_bounds)}\n")


def analyze_portfolio(portfolio, instance, identifier="", display=True, plot=True, plot_dir=None,
                      yearly_project_analysis=True):
    start_costs, continuing_costs = portfolio.costs_by_category(instance.projects)
    total_start_cost = start_costs.sum()
    total_continuing_cost = continuing_costs.sum()
    total_cost = total_start_cost + total_continuing_cost

    result = portfolio.result

    if display:
        selected_indices = result.nonzero()[0]
        print(f"Total number of started projects: {selected_indices.shape[0]}")

        print(f"Total cost for starting projects: {total_start_cost} "
              f"({total_start_cost / total_cost * 100 :.1f}%)")

        print(f"Total cost for ongoing projects: {continuing_costs.sum()} "
              f"({total_continuing_cost / total_cost * 100 :.1f}%)")

        stream_costs = portfolio.capability_stream_costs
        for i in range(portfolio.capability_streams):
            print(
                f"Capability Stream {i}: Allocated ${stream_costs[i]:.1f} of ${instance.capability_stream_budgets[i]:.1f}")

        # selected_indices = result.nonzero()
        # unselected_indices = np.nonzero(result == 0)
        # selected_projects = instance.projects[selected_indices]
        # unselected_projects = instance.projects[unselected_indices]
        #
        # print("\nDescriptive Statistics for Selected Projects")
        # statistics(selected_projects)
        #
        # print("\nDescriptive Statistics for Non-Selected Projects")
        # statistics(unselected_projects)

    if yearly_project_analysis:
        for t in range(1, instance.planning_window + 1):
            selected_indices = np.nonzero(result == t)[0]
            if selected_indices.size == 0:
                print(f"No projects selected for t={t}")
                continue
            print(selected_indices)
            selected_projects = np.array(instance.projects)[selected_indices]
            # selected_projects = instance.projects[selected_indices]
            print(f"\nDescriptive Statistics for t={t}")
            statistics(selected_projects)

    print()

    if plot:
        pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
        if plot_dir is None:
            cost_file = None
            count_file = None
            value_file = None
            stream_cost_file = None
            stream_value_file = None
            gantt_chart_file = None
        else:
            cost_file = os.path.join(plot_dir, f"{identifier}-costs.png")
            count_file = os.path.join(plot_dir, f"{identifier}-counts.png")
            value_file = os.path.join(plot_dir, f"{identifier}-value.png")
            stream_cost_file = os.path.join(plot_dir, f"{identifier}-stream_costs.png")
            stream_value_file = os.path.join(plot_dir, f"{identifier}-stream_values.png")
            gantt_chart_file = os.path.join(plot_dir, f"{identifier}-gantt_chart.png")

        plot_cost(instance.budget, start_costs, continuing_costs, portfolio.cost, cost_file)

        start_counts = portfolio.start_count(instance.budget_window)
        del start_counts[0]  # remove unscheduled projects
        continuing_counts = portfolio.continuing_count(instance.budget_window, instance.projects)
        del continuing_counts[0]  # remove time period 0 to match length of start times
        plot_start_times(start_counts, continuing_counts, count_file)

        value_over_time = portfolio.value_over_time
        plot_value_over_time(value_over_time, value_file)

        stream_costs, stream_values = portfolio.cost_value_by_stream(instance)

        plot_stream_costs(stream_costs, stream_cost_file)
        plot_stream_values(stream_values, stream_value_file)

        # create a Gantt chart of selected projects
        g = Gantt(portfolio, instance.projects, gantt_chart_file)
        g.render()
        g.show()



def statistics(projects):
    costs = np.fromiter((p.total_cost for p in projects), dtype=np.double)
    values = np.fromiter((p.total_value for p in projects), dtype=np.double)
    durations = np.fromiter((p.duration for p in projects), dtype=int)

    print(f"\tCost: {stats.describe(costs)}")
    print(f"\tValue: {stats.describe(values)}")
    print(f"\tDuration: {stats.describe(durations)}")


def plot_cost(budget, start_costs, continuing_costs, total_cost, path=None):
    budget = np.asarray(budget)
    total_cost = np.asarray(total_cost)
    fig, ax = plt.subplots()

    years = budget.shape[0]
    inds = np.arange(1, years + 1)

    budget_plot = ax.bar(inds, budget, color="lightblue", label="Budget")
    start_plot = ax.plot(inds, start_costs, label="Starting Cost")
    continuing_plot = ax.plot(inds, continuing_costs, label="Ongoing Cost")
    cost_plot = ax.plot(inds, total_cost, label="Total Cost")

    ax.legend()
    ax.set_ylabel('Cost ($m)', fontsize=14)
    ax.set_xlabel("Time Period", fontsize=14)
    plt.title("Cost Over Time")

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, 'major', 'y')

    plt.show()

    if path is not None:
        # TODO: create path if not exists
        fig.savefig(path)


def plot_start_times(start_times, continue_counts=None, path=None):
    years = len(start_times)
    fig, ax = plt.subplots()

    values = [*zip(*start_times.items())]

    chart = ax.bar(values[0], values[1], label="Starting")

    # add count labels on the bars
    # for rect in chart:
    #    height = rect.get_height()
    #    ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')

    if continue_counts is not None:
        cont_values = [*zip(*continue_counts.items())]
        stacked = ax.bar(cont_values[0], cont_values[1], bottom=values[1], label="Ongoing")

        # for rect in stacked:
        #    height = rect.get_height()
        #    ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')

    ax.set_ylabel('Number of projects', fontsize=14)
    ax.set_xlabel("Time Period", fontsize=14)
    plt.title("Number of Starting and Ongoing Projects Over Time")

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()
    ax.grid(True, 'major', 'y')

    plt.show()

    if path is not None:
        fig.savefig(path)


def plot_value_over_time(value_over_time, path=None):
    budget = np.asarray(value_over_time)
    fig, ax = plt.subplots()

    years = value_over_time.shape[0]
    inds = np.arange(1, years + 1)

    chart = ax.bar(inds, value_over_time)

    ax.set_ylabel('Value', fontsize=14)
    ax.set_xlabel("Time Period", fontsize=14)
    plt.title("Value Over Time")

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.legend()
    ax.grid(True, 'major', 'y')

    plt.show()

    if path is not None:
        fig.savefig(path)


def plot_stream_costs(stream_costs, path=None):
    stream_costs = np.asarray(stream_costs)
    fig, ax = plt.subplots()

    years = stream_costs.shape[1]
    inds = np.arange(1, years + 1)

    for i in range(stream_costs.shape[0]):
        cost = stream_costs[i]
        ax.plot(inds, cost, label=f"Stream {i + 1}")

    ax.legend()
    ax.set_ylabel('Cost ($m)', fontsize=14)
    ax.set_xlabel("Time Period", fontsize=14)
    plt.title("Stream Costs Over Time")

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, 'major', 'y')

    plt.show()

    if path is not None:
        # TODO: create path if not exists
        fig.savefig(path)


def plot_stream_values(stream_values, path=None):
    stream_costs = np.asarray(stream_values)
    fig, ax = plt.subplots()

    years = stream_costs.shape[1]
    inds = np.arange(1, years + 1)

    for i in range(stream_costs.shape[0]):
        value = stream_values[i]
        ax.plot(inds, value, label=f"Stream {i + 1}")

    ax.legend()
    ax.set_ylabel('Value', fontsize=14)
    ax.set_xlabel("Time Period", fontsize=14)
    plt.title("Stream Values Over Time")

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, 'major', 'y')

    plt.show()

    if path is not None:
        # TODO: create path if not exists
        fig.savefig(path)

# plot_cost(np.array([10, 11, 12, 13, 14]), np.array([4.6, 8.9, 11.1, 10.2, 6]))
# plot_start_times({1: 40, 2: 28, 3: 35, 4: 23, 5: 14})
