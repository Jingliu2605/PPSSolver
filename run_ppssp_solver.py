import pickle
import os
import pathlib
import numpy as np
from IPython.core.display_functions import display

from solvers.gurobi_solver import GurobiSolver
from problem.enums import SchedulingOrder, Optimizer
from executor import run_optimizer, report_heuristic
from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
from pymoo.operators.mutation.swap_mutation import SwapMutation
from analysis import get_result_path
from problem.portfolio import portfolio_from_pickle
import tkinter as tk
from tkinter import messagebox
from operators.gurobi_operators import get_gurobi_start_from_ea_solutions, generate_new_instance_with_changes
from problem.portfolio_problem_with_repair import build_from_array_and_repair
from pymoo.util.display import SingleObjectiveDisplay

def run_dynamic_ppssp_solver(project_file, optimized_portfolio, solver, param_values, output_dir, current_year, new_budget, removed_projects, pause_event=None, stop_event=None, gui_output=None):
    try:
        open_instance = open(project_file, 'rb')
        instance = pickle.load(open_instance)
        optimized_portfolio = portfolio_from_pickle(optimized_portfolio)

        # apply the changes to the instance
        omit_projects = [int(x) for x in removed_projects.split(",")]
        new_instance, baseline_solution_for_new_instance = generate_new_instance_with_changes(instance, optimized_portfolio.result,
                                                                                 current_year=current_year,
                                                                                 omit_projects=omit_projects,
                                                                                 reduce_budget=new_budget)
        
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        instance_output_dir = os.path.join(output_dir, new_instance.identifier)
        pathlib.Path(instance_output_dir).mkdir(parents=True, exist_ok=True)
        new_instance_file = os.path.join(instance_output_dir, f"{new_instance.identifier}.pkl")
        new_instance.write_to_pickle(new_instance_file)

        results = run_ppssp_solver(new_instance_file, solver, param_values, output_dir, baseline_solution_for_new_instance, pause_event=pause_event, stop_event=stop_event, gui_output=gui_output)
        return results
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    return

def run_ppssp_solver(project_file, solver, param_values, output_dir, baseline_solution_for_new_instance=None, pause_event=None, stop_event=None, gui_output=None):
    try:
        open_instance = open(project_file, 'rb')
        instance = pickle.load(open_instance)
        instance_output_dir = os.path.join(output_dir, instance.identifier)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(instance_output_dir).mkdir(parents=True, exist_ok=True)
        # Create a summary file for the solver
        if solver == "Gurobi":
            gurobi_summary = os.path.join(instance_output_dir, "Gurobi.csv")
            if not os.path.exists(gurobi_summary):
                with open(gurobi_summary, "w") as result_file:
                    result_file.write("Instance, Fitness, Bound, Gap, Time\n")
        else:
            ea_summary_file = os.path.join(instance_output_dir, f"{solver}.csv")
            if not os.path.exists(ea_summary_file):
                with open(ea_summary_file, "w") as result_file:
                    result_file.write(
                        'Instance, Mean Fit, Std., Min\n')

        if solver == "Gurobi":
            t_limitation = int(param_values[0])
            mip_gap = float(param_values[1])
            exact_dir = get_result_path(output_dir, instance.identifier, solver, t_limitation=t_limitation, mip_gap=mip_gap)
            # os.path.join(output_dir, f"Gurobi_{t_limitation}_{mip_gap}")
            pathlib.Path(exact_dir).mkdir(parents=True, exist_ok=True)
            print(f"Run Gurobi with time limitation {t_limitation} and mip gap {mip_gap}")
            log_file = os.path.join(exact_dir, f"Gurobi_{instance.identifier}.log")
            gurobi_pool_solutions_file = os.path.join(exact_dir, f"Gurobi_pool_solutions.npy")

            if baseline_solution_for_new_instance is not None:
                portfolio, violation, x, phenotype = build_from_array_and_repair(baseline_solution_for_new_instance,
                                                                                 baseline_solution_for_new_instance,
                                                                                 instance, 0.5)
                n_start, start_solutions = get_gurobi_start_from_ea_solutions(x, instance)
            else:
                n_start = None
                start_solutions = None
                
            try:
                exact_solver = GurobiSolver(instance, time_limit=t_limitation, error_threshold=mip_gap, num_solutions=10,
                                            pool_search_mode=1,  num_start=n_start,
                                            start_solutions=start_solutions)
                exact_portfolio, status, model, exec_time, pool_solutions, suboptimal = exact_solver.solve(verbose=True,
                                                                                                        log_to_console=True,
                                                                                                        log_file=log_file)
            except Exception as e:
                gui_output.insert(tk.END, f"Error in Gurobi solver: {e}\n")
                gui_output.update_idletasks()
                messagebox.showerror("Error", f"Error in Gurobi solver: {e}")
                return None
            exact_portfolio.write_to_pickle(os.path.join(exact_dir, f"Gurobi_portfolio.pkl"))
            gap = model.getAttr('MIPGap')
            bound = model.getAttr('ObjBound')
            np.save(gurobi_pool_solutions_file, pool_solutions)
            with open(gurobi_summary, "a") as result_file:
                result_file.write(f"{instance.identifier}, {exact_portfolio.value}, {bound}, {gap}, {exec_time}\n")
            print("Results saved")
            return exact_portfolio.value, exec_time, gap
        elif solver == "GA":
            eval_termination = ('n_eval', int(param_values[2]))
            ga_cross_rate = float(param_values[3])
            ga_mut_rate = float(param_values[4])
            display_frequency = int(param_values[5])
            pop_size = int(param_values[1])
            start_time_rep = True
            pheno = False
            output_solver_dir = get_result_path(output_dir, instance.identifier, solver)
            # os.path.join(output_dir, f"{solver}")
            runs = int(param_values[0])
            fitnesses = np.zeros(runs)
            times = np.zeros(runs)
            generations = np.zeros(runs)
            evals = np.zeros(runs)
            for run_index in range(runs):
                results = run_optimizer(Optimizer.GA, pop_size, instance, eval_termination, run_index, 1, output_solver_dir,
                                        SchedulingOrder.EARLIEST, display_each_run=False,
                                        analyze_output=False, crossover=HalfUniformCrossover(prob=ga_cross_rate),
                                        crossover_rate=ga_cross_rate, pymoo_verbose=True, 
                                        seeded_time=0, old_solution=baseline_solution_for_new_instance,
                                        mutation=SwapMutation(prob=ga_mut_rate), mutation_rate=ga_mut_rate, pheno=False, 
                                        pause_event=pause_event, stop_event=stop_event, gui_output=gui_output,
                                        display=SingleObjectiveDisplay(resolution=display_frequency))

                fitnesses[run_index], times[run_index], generations[run_index], evals[run_index] = report_heuristic(instance, solver, run_index, output_solver_dir, 1, results, start_time_rep, pheno)
            return np.mean(fitnesses), np.mean(times), np.mean(generations), np.mean(evals)
        elif solver == "DE":
            eval_termination = ('n_eval', int(param_values[2]))
            pop_size = int(param_values[1])
            start_time_rep = False
            pheno = True
            de_cross_rate = float(param_values[3])
            de_weight = float(param_values[4])
            display_frequency = int(param_values[5])
            output_solver_dir = get_result_path(output_dir, instance.identifier, solver)
            runs = int(param_values[0])
            fitnesses = np.zeros(runs)
            times = np.zeros(runs)
            generations = np.zeros(runs)
            evals = np.zeros(runs)
            for run_index in range(runs):
                results = run_optimizer(Optimizer.DE, pop_size, instance, eval_termination, run_index, 1, output_solver_dir,
                                        SchedulingOrder.EARLIEST, display_each_run=False,
                                        analyze_output=False, F=de_weight, CR=de_cross_rate, crossover='bin',
                                        selection='rand', dither='no', jitter=True, pymoo_verbose=True,
                                        pheno=True, pause_event=pause_event, stop_event=stop_event,
                                        display=SingleObjectiveDisplay(resolution=display_frequency), gui_output=gui_output)
                fitnesses[run_index], times[run_index], generations[run_index], evals[run_index] = report_heuristic(instance, solver, run_index, output_solver_dir, 1, results, start_time_rep, pheno)
            return np.mean(fitnesses), np.mean(times), np.mean(generations), np.mean(evals)
        elif solver == "BRKGA":
            eval_termination = ('n_eval', int(param_values[2]))
            pop_size = int(param_values[1])
            output_solver_dir = get_result_path(output_dir, instance.identifier, solver)
            runs = int(param_values[0])
            fitnesses = np.zeros(runs)
            times = np.zeros(runs)
            generations = np.zeros(runs)
            evals = np.zeros(runs)
            start_time_rep = False
            pheno = True
            display_frequency = int(param_values[5])
            bias = float(param_values[3])  # 0.3663063352420889
            prop_elites = float(param_values[4]) #0.21573475847727405


            for run_index in range(runs):
                results = run_optimizer(Optimizer.BRKGA, pop_size, instance, eval_termination, run_index, 1, output_solver_dir,
                                        SchedulingOrder.EARLIEST, display_each_run=False, analyze_output=False,
                                        bias=bias,
                                        prop_elites=prop_elites, prop_mutants=0.1,
                                        display=SingleObjectiveDisplay(resolution=display_frequency),
                                        pymoo_verbose=True, pheno=True, pause_event=pause_event, stop_event=stop_event, gui_output=gui_output)
                fitnesses[run_index], times[run_index], generations[run_index], evals[run_index] = report_heuristic(
                    instance, solver, run_index, output_solver_dir, 1, results, start_time_rep, pheno)
            return np.mean(fitnesses), np.mean(times), np.mean(generations), np.mean(evals)
        elif solver == "HEGCL":
            size_groups = int(param_values[1])
            t_limitation = int(param_values[2])
            pop_size = int(param_values[3])
            ga_cross_rate = float(param_values[4])
            ga_mut_rate = float(param_values[5])
            display = int(param_values[6])
            start_time_rep = True
            pheno = False
            output_solver_dir = get_result_path(output_dir, instance.identifier, solver, t_limitation=t_limitation, size_groups=size_groups)
            # os.path.join(output_dir, f"{solver}_{size_groups}_{t_limitation}")
            runs = int(param_values[0])
            fitnesses = np.zeros(runs)
            times = np.zeros(runs)
            generations = np.zeros(runs)
            evals = np.zeros(runs)
            for run_index in range(runs):
                results = run_optimizer(Optimizer.AGA, pop_size, instance, None, run_index, 1, output_solver_dir,
                                        SchedulingOrder.EARLIEST, display_each_run=False,
                                        analyze_output=False, crossover=HalfUniformCrossover(prob=ga_cross_rate),
                                        crossover_rate=ga_cross_rate, pymoo_verbose=True, seeded_time=t_limitation,
                                        group_size=size_groups, old_solution=baseline_solution_for_new_instance,
                                        mutation=SwapMutation(prob=ga_mut_rate), mutation_rate=ga_mut_rate, pheno=False,
                                        display=SingleObjectiveDisplay(resolution=display),
                                        pause_event=pause_event, stop_event=stop_event, gui_output=gui_output)
                fitnesses[run_index], times[run_index], generations[run_index], evals[run_index] = report_heuristic(instance, solver, run_index, output_solver_dir, 1, results, start_time_rep, pheno)
            return np.mean(fitnesses), np.mean(times), np.mean(generations), np.mean(evals)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
        