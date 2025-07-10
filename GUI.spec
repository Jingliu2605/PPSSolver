# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pymoo.operators.repair.bounds_repair', 'autograd', 'problem.datagen', 'problem.enums', 'problem.ordering', 'problem.portfolio', 
                  'problem.portfolio_permutation_problem', 'problem.portfolio_problem_with_repair', 'problem.portfolio_real_ordered_problem', 'problem.portfolio_real_ordered_problem_multi_evaluation', 'problem.portfolio_real_problem', 'problem.portfolio_selection_instance', 'problem.portfolio_selection_problem', 'problem.project', 'problem.value_functions', 'operators.feasibility_budget_repair', 'operators.feasibility_budget_repair_real', 'operators.insertion_mutation', 'operators.portfolio_shuffled_crossover', 'solvers.auto_meta_raps', 'solvers.meta_raps', 'solvers.permutation_annealer', 'solvers.real_permutation_annealer', 'solvers.simulated_annealing', 'solvers.start_time_annealer', 'pymoo.model.algorithm', 'pymoo.model.callback', 'pymoo.model.crossover', 'pymoo.model.duplicate', 'pymoo.model.evaluator', 'pymoo.model.indicator', 'pymoo.model.individual', 'pymoo.model.infill', 'pymoo.model.initialization', 'pymoo.model.mating', 'pymoo.model.mutation', 'pymoo.model.population', 'pymoo.model.problem', 'pymoo.model.repair', 'pymoo.model.replacement', 'pymoo.model.result', 'pymoo.model.sampling', 'pymoo.model.selection', 'pymoo.model.survival', 'pymoo.model.termination', 'pymoo.nds.cpp_non_dominated_sorting', 'pymoo.nds.dominator', 'pymoo.nds.fast_non_dominated_sort', 'pymoo.nds.naive_non_dominated_sort', 'pymoo.nds.non_dominated_sorting', 'pymoo.termination.collection', 'pymoo.termination.cv_tol', 'pymoo.termination.default_termination', 'pymoo.termination.max_eval', 'pymoo.termination.max_gen', 'pymoo.termination.max_time', 'pymoo.termination.sliding_window_termination', 'pymoo.termination.so_obj_tol', 'pymoo.termination.x_tol', 'pymoo.util.display', 'pymoo.util.factory', 'pymoo.util.misc', 'pymoo.util.normalization', 'pymoo.util.randomized_argsort', 'pymoo.util.roulette', 'pymoo.util.sliding_window', 'pymoo.algorithms.genetic_algorithm', 'pymoo.algorithms.multipop_algorithm', 'pymoo.algorithms.so_brkga', 'pymoo.algorithms.so_de', 'pymoo.algorithms.so_genetic_algorithm', 'pymoo.algorithms.so_local_search', 'pymoo.algorithms.so_nelder_mead', 'pymoo.algorithms.so_niching_ga', 'pymoo.algorithms.so_pattern_search', 'pymoo.operators.fitness_survival', 'pymoo.operators.integer_from_float_operator', 'pymoo.operators.no_duplicate_elimination', 'pymoo.operators.crossover.biased_crossover', 'pymoo.operators.crossover.crossover_mask', 'pymoo.operators.crossover.differential_evolution_crossover', 'pymoo.operators.crossover.edge_recombination_crossover', 'pymoo.operators.crossover.exponential_crossover', 'pymoo.operators.crossover.half_uniform_crossover', 'pymoo.operators.crossover.ordered_crossover', 'pymoo.operators.crossover.partially_mapped_crossover', 'pymoo.operators.crossover.point_crossover', 'pymoo.operators.crossover.similar_block_order_crossover', 'pymoo.operators.crossover.similar_job_order_crossover', 'pymoo.operators.crossover.simulated_binary_crossover', 'pymoo.operators.crossover.uniform_crossover', 'pymoo.operators.mutation.bit_flip_mutation', 'pymoo.operators.mutation.no_mutation', 'pymoo.operators.mutation.polynomial_mutation', 'pymoo.operators.mutation.scramble_mutation', 'pymoo.operators.mutation.swap_mutation', 'pymoo.operators.sampling.latin_hypercube_sampling', 'pymoo.operators.sampling.permutation_sampling', 'pymoo.operators.sampling.random_sampling', 'pymoo.operators.repair.biased_crossover', 'pymoo.operators.repair.bounce_back_repair', 'pymoo.operators.repair.no_repair', 'pymoo.operators.repair.out_of_bounds_repair', 'pymoo.operators.selection.random_selection', 'pymoo.operators.selection.tournament_selection', 'pymoo.performance_indicator.distance_indicator', 'pymoo.performance_indicator.igd', 'pymoo.performance_indicator.kktpm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
