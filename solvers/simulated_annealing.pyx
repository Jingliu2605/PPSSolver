# Modified variant of simulated annealing attained from https://github.com/perrygeo/simanneal

# Copyright (c) 2009, Richard J. Wagner <wagnerr@umich.edu>
# Copyright (c) 2014, Matthew T. Perry <perrygeo@gmail.com>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#from abc import abstractmethod
import copy
import datetime
import math
import pickle
import signal
import sys
import time

import numpy as np

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    return str(datetime.timedelta(seconds=round(seconds)))

    # s = int(round(seconds))  # round to nearest second
    # h, s = divmod(s, 3600)   # get hours and remainder
    # m, s = divmod(s, 60)     # split remainder into minutes and seconds
    # return f'{h:4i}:{m:02i}:{s:02i}' #'%4i:%02i:%02i' % (h, m, s)


class SimulatedAnnealing:
    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """

    # defaults
    t_max = 25000.0
    t_min = 2.5
    steps = 50000
    updates = 100
    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    # placeholders
    best_state = None
    best_energy = None
    start = None

    def __init__(self, initial_state=None, load_state=None, random_seed=None):

        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.load_state(load_state)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')
        np.random.seed(random_seed)

        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        """Saves state to pickle"""
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self.state, fh)

    def load_state(self, fname=None):
        """Loads state from pickle"""
        with open(fname, 'rb') as fh:
            self.state = pickle.load(fh)

    #    @abstractmethod
    def move(self):
        """Create a state change"""
        pass

    #    @abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass

    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def set_schedule(self, schedule):
        """Takes the output from `auto` and sets the attributes
        """
        self.t_max = schedule['tmax']
        self.t_min = schedule['tmin']
        #self.steps = int(schedule['steps'])
        #self.updates = int(schedule['updates'])

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of
        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def update(self, *args, **kwargs):
        """Wrapper for internal update.
        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E):
        """Default update, outputs to stderr.
        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.
        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.
        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        elapsed = time.perf_counter() - self.start
        if step == 0:
            print('\n Temperature          Best     Elapsed   Remaining')
            print(
            f'\r{T:12.5f}  {self.best_energy:12.2f}                      {time_string(elapsed):s}            ', end="")
            sys.stdout.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print(f'\r{T:12.5f}  {self.best_energy:12.2f}  {time_string(elapsed):s}  '
                  f'{self.steps - step:10}', end="")
            sys.stdout.flush()

    def anneal(self):

        """Minimizes the energy of a system by simulated annealing.
        Parameters
        state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
        """
        cdef int step = 0
        #cdef double tfactor, T, E, update_wavelength, dE
        #cdef int trials, accepts, improces

        self.start = time.perf_counter()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.t_min <= 0.0:
            raise Exception('Exponential cooling requires a minimum temperature greater than zero.')
        tfactor = -math.log(self.t_max / self.t_min)

        # Note initial state
        T = self.t_max
        E = self.energy()
        prev_state = self.copy_state(self.state)
        prev_energy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        #trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            update_wavelength = self.steps / self.updates
            self.update(step, T, E)

        # Attempt moves to new states
        while step < self.steps:
            step += 1
            T = self.t_max * math.exp(tfactor * step / self.steps)
            self.move()
            E = self.energy()
            dE = E - prev_energy
            if dE > 0.0 and math.exp(-dE / T) < np.random.random():
                # Restore previous state
                self.state = self.copy_state(prev_state)
                E = prev_energy
            else:
                prev_state = self.copy_state(self.state)
                prev_energy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // update_wavelength) > ((step - 1) // update_wavelength):
                    self.update(step, T, E)

        self.update(step, T, self.best_energy)
        print()  # force newline after completion
        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy

    def auto(self, steps=2000, t_max_percentage=0.98):
        """Explores the annealing landscape and
        estimates optimal temperature settings.
        Returns a dictionary suitable for the `set_schedule` method.
        """

        initial_state = self.copy_state(self.state)  # save the initial state

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.energy()
            prev_state = self.copy_state(self.state)
            prev_energy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                self.move()
                E = self.energy()
                dE = E - prev_energy
                if dE > 0.0 and math.exp(-dE / T) < np.random.random():
                    self.state = self.copy_state(prev_state)
                    E = prev_energy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prev_state = self.copy_state(self.state)
                    prev_energy = E
            return E, float(accepts) / steps, float(improves) / steps

        print(f"Running automatic parameter search using {steps} iterations.")

        step = 0
        self.start = time.perf_counter()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.energy()
        while T == 0.0:
            step += 1
            self.move()
            T = abs(self.energy() - E)

        # Search for Tmax
        E, acceptance, improvement = run(T, steps)

        while acceptance > t_max_percentage:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
        while acceptance < t_max_percentage:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
        t_max = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
        t_min = T

        elapsed = time.perf_counter() - self.start

        print(f"\tDone in {elapsed:0.1E}s. tmax: {t_max}, tmin:{t_min}")

        sys.stdout.flush()

        self.state = initial_state  #restore the initial state
        # return params
        return {'tmax': t_max, 'tmin': t_min}
