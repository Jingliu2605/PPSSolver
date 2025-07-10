"""
Gantt.py is a simple class to render Gantt charts.

Implementation from: https://github.com/stefanSchinkel/gantt

Used under MIT license.
"""

import os
import json
import platform
from operator import sub

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# TeX support: on Linux assume TeX in /usr/bin, on OSX check for texlive
from matplotlib.ticker import MultipleLocator

if (platform.system() == 'Darwin') and 'tex' in os.getenv("PATH"):
    LATEX = True
elif (platform.system() == 'Linux') and os.path.isfile('/usr/bin/latex'):
    LATEX = True
else:
    LATEX = False

# setup pyplot w/ tex support
if LATEX:
    rc('text', usetex=True)


class Package():
    """Encapsulation of a work package

    A work package is instantiated from a dictionary. It **has to have**
    a label, astart and an end. Optionally it may contain milestones
    and a color

    :arg str pkg: dictionary w/ package data name
    """
    def __init__(self, label, start, end):

        DEFCOLOR = "#32AEE0"

        self.label = label
        self.start = start
        self.end = end

        if self.start < 0 or self.end < 0:
            raise ValueError("Package cannot begin at t < 0")
        if self.start > self.end:
            raise ValueError("Cannot end before started")

        # try:
        #    self.milestones = pkg['milestones']
        # except KeyError:
        #    pass

        # try:
        #    self.color = pkg['color']
        # except KeyError:
        self.color = DEFCOLOR

        # try:
        #    self.legend = pkg['legend']
        # except KeyError:
        self.legend = None


class Gantt():
    """Gantt
    Class to render a simple Gantt chart, with optional milestones
    """
    def __init__(self, portfolio, projects, gantt_chart_file=None):
        """ Instantiation

        Create a new Gantt using the data in the file provided
        or the sample data that came along with the script

        :arg str dataFile: file holding Gantt data
        """
        # self.dataFile = dataFile

        # some lists needed
        self.packages = []
        self.labels = []
        self.gantt_chart_file = gantt_chart_file

        self._convert_data(portfolio, projects)
        self._process_data()

    def _convert_data(self, portfolio, projects):
        scheduled = np.nonzero(portfolio.result)[0]
        num_scheduled = scheduled.shape[0]

        self.packages = [None] * num_scheduled
        self.labels = [None] * num_scheduled

        # TODO: order projects by start time
        #sorted_schedule = np.copy(scheduled)
        #sorted(sorted_schedule, key=lambda x: portfolio.result[x])
        for i in range(num_scheduled):
            index = scheduled[i]
            name = projects[index].project_name
            start = portfolio.result[index] + 1  # TODO: this has 1-based time - is this appropriate?
            end = start + projects[index].duration
            self.packages[i] = Package(name, start, end)
            self.labels[i] = name

        # sort packages by start time
        self.packages = sorted(self.packages, key=lambda x: (x.start, x.end))

    def _process_data(self):
        """ Process data to have all values needed for plotting
        """
        # parameters for bars
        self.nPackages = len(self.labels)
        self.start = [None] * self.nPackages
        self.end = [None] * self.nPackages

        idx = 0
        for pkg in self.packages:
             #self.labels.index(pkg.label)
            self.start[idx] = pkg.start
            self.end[idx] = pkg.end
            idx += 1

        self.durations = map(sub, self.end, self.start)
        self.yPos = np.arange(self.nPackages, 0, -1)

    def format(self):
        """ Format various aspect of the plot, such as labels,ticks, BBox
        :todo: Refactor to use a settings object
        """
        # format axis
        plt.tick_params(
            axis='both',    # format x and y
            which='both',   # major and minor ticks affected
            bottom='on',    # bottom edge ticks are on
            top='on',      # top, left and right edge ticks are off
            left='off',
            right='off')

        # tighten axis but give a little room from bar height
        max_end = max(self.end)
        plt.xlim(1, max_end)
        plt.ylim(0.5, self.nPackages + .5)

        # add title and package names
        plt.yticks(self.yPos, self.labels, fontsize=1)  # label text size
        # plt.title(self.title)

        # if self.xlabel:
        #    plt.xlabel(self.xlabel)
        plt.xlabel("Time")
        plt.title("Gantt Chart of Optimized Portfolio")


        # create label for every 5th year
        xlabels = [None] * max_end
        for i in range(1, len(xlabels), 5):
            xlabels[i] = str(i)
        # add minor tick grid at each year and label every 5 years
        plt.xticks(np.arange(1, max_end + 1), xlabels)


    def add_milestones(self):
        """Add milestones to GANTT chart.
        The milestones are simple yellow diamonds
        """

        if not self.milestones:
            return

        x = []
        y = []
        for key in self.milestones.keys():
            for value in self.milestones[key]:
                y += [self.yPos[self.labels.index(key)]]
                x += [value]

        plt.scatter(x, y, s=120, marker="D",
                    color="yellow", edgecolor="black", zorder=3)

    def add_legend(self):
        """Add a legend to the plot iff there are legend entries in
        the package definitions
        """

        cnt = 0
        for pkg in self.packages:
            if pkg.legend:
                cnt += 1
                idx = self.labels.index(pkg.label)
                self.barlist[idx].set_label(pkg.legend)

        if cnt > 0:
            self.legend = self.ax.legend(
                shadow=False, ncol=3, fontsize="medium")

    def render(self):
        """ Prepare data for plotting
        """

        # init figure
        self.fig, self.ax = plt.subplots()
        self.ax.yaxis.grid(False)
        self.ax.xaxis.grid(True)

        # assemble colors
        colors = []
        for pkg in self.packages:
            colors.append(pkg.color)

        self.barlist = plt.barh(self.yPos, list(self.durations),
                                left=self.start,
                                align='center',
                                height=0.5,  # height of bars
                                alpha=1,
                                color=colors)

        # format plot
        self.format()
        if self.gantt_chart_file is not None:
            self.fig.savefig(self.gantt_chart_file)
        # self.add_milestones()
        # self.add_legend()

    @staticmethod
    def show():
        """ Show the plot
        """
        plt.show()

    @staticmethod
    def save(saveFile='img/GANTT.png'):
        """ Save the plot to a file. It defaults to `img/GANTT.png`.

        :arg str saveFile: file to save to
        """
        plt.savefig(saveFile, bbox_inches='tight')


# if __name__ == '__main__':
#     g = Gantt('sample.json')
#     g.render()
#     g.show()
#     # g.save('img/GANTT.png')
