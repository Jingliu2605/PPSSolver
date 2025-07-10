import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import ticker

from instance_parameters import InstanceParameters
from problem.datagen import _weibull_estimate, mvlnorm_generate_costdur
from problem.enums import ValueFunction, SpreadDistribution
from problem.portfolio_selection_instance import generate_instance


def generate_cost_dur_plot():
    num_points = 148

    durations_iip = np.zeros(num_points)
    costs_iip = np.zeros(num_points)
    durations_mvn = np.zeros(num_points)
    costs_mvn = np.zeros(num_points)

    with open(r"D:\OneDrive - UNSW\IIP2016Data.csv", "r", newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the header
        index = 0
        for row in reader:
            durations_iip[index] = int(row[6])
            costs_iip[index] = int(row[7])
            index += 1

    cost_dur = mvlnorm_generate_costdur(num_points)

    for i in range(num_points):
        durations_mvn[i] = cost_dur[i, 0]
        costs_mvn[i] = cost_dur[i, 1]

    fig, ax = plt.subplots()
    ax.scatter(durations_iip, costs_iip, c='tab:orange', label='2016 IIP Data',
               alpha=1, edgecolors='none')

    from matplotlib import rcParams
    ax.scatter(durations_mvn, costs_mvn, c='tab:blue', label='MV Lognormal',
               alpha=1, s=rcParams['lines.markersize'] ** 1.6, edgecolors='none')

    ax.legend()
    ax.grid(True)
    ax.set_ylabel('Total Cost ($m)', fontsize=14)
    ax.set_xlabel("Project Duration", fontsize=14)

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, 'major', 'y')

    plt.show()
    fig.savefig(r'D:\OneDrive - UNSW\Plots\cost_duration_comparison.pdf')


def generate_cost_value_plot():
    num_points = 150

    costs = np.zeros(num_points)
    values = np.zeros(num_points)

    parameters = InstanceParameters(num_projects=num_points,
                                    planning_window=20,
                                    base_budget=14439,  # value taken from 2020 FSP
                                    budget_increase=1637,  # value taken from 2020 FSP
                                    value_func=ValueFunction.COST_DUR,
                                    cost_distribution=SpreadDistribution.WEIBULL,
                                    )
    instance = generate_instance(parameters, None, False)

    for i in range(num_points):
        costs[i] = instance.projects[i].total_cost
        values[i] = instance.projects[i].total_value

    fig, ax = plt.subplots()
    ax.scatter(costs, values, c='tab:blue', alpha=0.8, edgecolors='none')

    ax.grid(True)
    ax.set_xlabel('Total Cost ($m)', fontsize=14)
    ax.set_ylabel("Total Value", fontsize=14)

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, 'major', 'y')

    plt.show()
    fig.savefig(r'D:\OneDrive - UNSW\Plots\cost_value.pdf')


def generate_weibull_plot(duration=100, shape=1.589, scale=0.71, std_shape=2, std_scale=0.3):
    num_points = 10

    fig, ax = plt.subplots()
    for i in range(num_points):
        fuzzy_shape = np.random.normal(shape, std_shape)
        while fuzzy_shape < 0.01:
            fuzzy_shape = np.random.normal(shape, std_shape)
        fuzzy_scale = max(np.random.normal(scale, std_scale), 0.1)

        y = np.fromiter((_weibull_estimate((t + 1) / duration, fuzzy_shape, fuzzy_scale) for t in range(duration)),
                        dtype=np.double)

        plt.scatter(x=np.arange(duration) / 100, y=y)

    ax.grid(True)
    ax.set_xlabel('Project Completion', fontsize=14)
    ax.set_ylabel('Cumulative Proportion', fontsize=14)

    # ensure x axis only identifies integers with step sizes of 1, 2, 5, or 10
    ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, 'major', 'y')
    plt.show()

    fig.savefig(r'D:\OneDrive - UNSW\Plots\weibull_ecdf_samples.pdf')


def stream_piechart():
    fig, ax = plt.subplots()
    cap_budget = [75, 65, 55, 15, 7]
    cap_stream = ['Maritime', 'Air', 'Land', 'Info. & Cyber', 'Space']

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n(${:d}B)".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(cap_budget, autopct=lambda pct: func(pct, cap_budget),
                                      textprops=dict(color="w"))

    ax.legend(wedges, cap_stream,
              title="Capability",
              loc="center left",
              bbox_to_anchor=(.9, 0, 0, 1))

    plt.setp(autotexts, size=14)
    plt.show()

    #fig.savefig(r'D:\OneDrive - UNSW\Plots\capability_stream_budgets.pdf')

def stream_donut():
    fig, ax = plt.subplots(figsize=(5.3, 3), subplot_kw=dict(aspect="equal"))

    recipe = ["Maritime: $75B",
              "Air: $65B",
              "Land: $55B",
              "Info. and Cyber: $15B",
              "Space: $7B"]

    data = [75, 65, 55, 15, 7]

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    plt.show()
    fig.savefig(r'D:\OneDrive - UNSW\Plots\capability_stream_budgets.pdf')
