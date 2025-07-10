import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
from run_ppssp_solver import run_ppssp_solver, run_dynamic_ppssp_solver
from generate_ppssp_instance import generate_ppssp_instance
import os
import threading
import time
from analysis import get_experimental_data, analyze_from_pickles, get_portfolio_name, get_result_path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class PPSSPGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PPSSP Solver")
        self.geometry("820x840")

        # --- Color and Style Definitions ---
        self.bg_color = "#f4f6fa"   # light blue-gray background
        self.accent_color = "#1c5887"
        self.button_color = "#4984b3"
        self.button_fg = "#ffffff"
        self.frame_bg = "#e3eaf2"
        self.label_fg = "#222222"
        self.entry_bg = "#ffffff"
        self.highlight_color = "#ff9800"
        self.font_family = "Segoe UI"
        self.font_size = 10

        self.configure(bg=self.bg_color)
        self.style = ttk.Style()
        self._setup_styles()

        self.solver_thread = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.pause_event.set()  # Initially not paused

        self.dynamic_solver_thread = None
        self.pause_event_dynamic = threading.Event()
        self.stop_event_dynamic = threading.Event()
        self.pause_event_dynamic.set()  # Initially not paused

        self.create_widgets()

    def _setup_styles(self):
        style = self.style
        style.theme_use("clam")
        style.configure(".",
                        font=(self.font_family, self.font_size),
                        background=self.bg_color,
                        foreground=self.label_fg
                        )
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.label_fg)
        style.configure("TButton", background=self.button_color, foreground=self.button_fg,
                        font=(self.font_family, self.font_size, "bold"))
        style.map("TButton",
                  background=[("active", self.accent_color)],
                  foreground=[("active", self.button_fg)]
                  )
        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background=self.frame_bg, foreground=self.label_fg,
                        font=(self.font_family, self.font_size, "bold"), padding=[10, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", self.accent_color)],
                  foreground=[("selected", self.button_fg)]
                  )
        style.configure("TLabelframe", background=self.bg_color, foreground=self.accent_color,
                        font=(self.font_family, self.font_size, "bold"))
        style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.accent_color)
        style.configure("Treeview", background=self.entry_bg, fieldbackground=self.entry_bg, foreground=self.label_fg,
                        font=(self.font_family, self.font_size))
        style.configure("Treeview.Heading", background=self.accent_color, foreground=self.button_fg,
                        font=(self.font_family, self.font_size, "bold"))
        style.configure("TProgressbar", background=self.accent_color)
        style.configure("TCheckbutton", background=self.bg_color, foreground=self.label_fg)
        style.configure("TEntry", fieldbackground=self.entry_bg, background=self.entry_bg)
        # For text widgets, set manually below

        # Reduce OptionMenu (TMenubutton) padding
        style.configure("TMenubutton", padding=[4, 2])  # [horizontal, vertical] padding

    def create_widgets(self):
        # Create a custom style for the notebook tabs
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[10, 3])  # [horizontal padding, vertical padding]

        # Notebook for organizing pages
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.notebook = notebook

        # Make tabs expand to fill the width
        self.update_idletasks()
        total_tabs = 5  # Number of tabs/pages
        tab_width = int(self.winfo_width() / total_tabs)
        style = ttk.Style()
        style.configure("TNotebook.Tab", width=tab_width, padding=[10, 3])

        # Add a Help button at the top
        help_button = ttk.Button(self, text="Help", command=self.open_help)
        help_button.pack(side="top", anchor="ne", padx=10, pady=5)

        # Pages
        self.configuration_page = ttk.Frame(notebook)
        self.instance_generation_page = ttk.Frame(notebook)
        self.instance_optimization_page = ttk.Frame(notebook)
        self.dynamic_instance_optimization_page = ttk.Frame(notebook)
        self.results_analysis_page = ttk.Frame(notebook)

        notebook.add(self.configuration_page, text="Paths Configuration")
        notebook.add(self.instance_generation_page, text="Instance Generation")
        notebook.add(self.instance_optimization_page, text="Instance Optimization")
        notebook.add(self.dynamic_instance_optimization_page, text="Dynamic Handling")
        notebook.add(self.results_analysis_page, text="Results Analysis")

        # Create widgets for each page
        self.create_configuration_widgets()
        self.create_instance_generation_widgets()
        self.create_instance_optimization_widgets()
        self.create_results_analysis_widgets()
        self.create_dynamic_instance_optimization_widgets()

        # Set the default page to the instance generation page
        notebook.select(self.configuration_page)

    def create_configuration_widgets(self):
        # Frame for specifying paths
        paths_frame = ttk.LabelFrame(self.configuration_page, text="")
        paths_frame.pack(fill="x", padx=10, pady=1)
        paths_frame.configure(style="TLabelframe")
        # Remove border
        paths_frame['borderwidth'] = 0

        ttk.Label(paths_frame, text="Instance Path:").grid(row=0, column=0, sticky="w")
        self.instance_save_path = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.instance_save_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(paths_frame, text="Load Default Path", command=self.load_default_instance_path).grid(row=0, column=2, padx=5)
        ttk.Button(paths_frame, text="Browse", command=self.select_instance_save_path).grid(row=0, column=3, padx=5)

        ttk.Label(paths_frame, text="Result Path:").grid(row=1, column=0, sticky="w")
        self.result_save_path = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.result_save_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(paths_frame, text="Load Default Path", command=self.load_default_result_path).grid(row=1, column=2, padx=5)
        ttk.Button(paths_frame, text="Browse", command=self.select_result_save_path).grid(row=1, column=3, padx=5)

        # Add note about paths configuration
        ttk.Label(
            paths_frame,
            text="Please set the paths above before generating instances or running optimization.",
            foreground=self.highlight_color,
            font=(self.font_family, self.font_size, "bold")
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(0, 5))

    def load_default_instance_path(self):
        # Ensure the directory exists
        path = os.path.join(os.getcwd(), "instances")
        if not os.path.exists(path):
            os.makedirs(path)

        # Set the path
        self.instance_save_path.set(path)

    def load_default_result_path(self):
        # Ensure the directory exists
        path = os.path.join(os.getcwd(), "output")
        if not os.path.exists(path):
            os.makedirs(path)

        # Set the path
        self.result_save_path.set(path)

    def create_instance_generation_widgets(self):
        # Frame for instance generation
        instance_frame = ttk.LabelFrame(self.instance_generation_page, text="")
        instance_frame.pack(fill="x", padx=10, pady=1)
        instance_frame.configure(style="TLabelframe")
        instance_frame['borderwidth'] = 0

        ttk.Label(instance_frame, text="Instance Index:").grid(row=0, column=0, sticky="w")
        self.instance_index = tk.IntVar(value=1)
        ttk.Entry(instance_frame, textvariable=self.instance_index).grid(row=0, column=1)

        ttk.Label(instance_frame, text="Number of Projects:").grid(row=1, column=0, sticky="w")
        self.num_projects = tk.IntVar(value=1000)
        ttk.Entry(instance_frame, textvariable=self.num_projects).grid(row=1, column=1)

        ttk.Label(instance_frame, text="Planning Years:").grid(row=2, column=0, sticky="w")
        self.planning_years = tk.IntVar(value=25)
        ttk.Entry(instance_frame, textvariable=self.planning_years).grid(row=2, column=1)

        ttk.Label(instance_frame, text="Initial Budget Proportion (0-1):").grid(row=3, column=0, sticky="w")
        self.budget_prop = tk.DoubleVar(value=0.25)
        ttk.Entry(instance_frame, textvariable=self.budget_prop).grid(row=3, column=1)

        ttk.Label(instance_frame, text="Discount Rate (0-1):").grid(row=4, column=0, sticky="w")
        self.discount_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(instance_frame, textvariable=self.discount_rate).grid(row=4, column=1)

        ttk.Button(instance_frame, text="Generate Instance", command=self.generate_instance).grid(row=5, columnspan=2, pady=10)

        # Add log output area
        ttk.Label(instance_frame, text="Instance Generation Output:").grid(row=6, column=0, columnspan=2, sticky="w")
        self.instance_generation_output = tk.Text(instance_frame, height=5, width=60, wrap="word", bg=self.entry_bg, fg=self.label_fg, font=(self.font_family, self.font_size))
        self.instance_generation_output.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

        self.progress_generate = ttk.Progressbar(instance_frame, mode="indeterminate")
        self.progress_generate.grid(row=8, column=0, columnspan=2, pady=(10, 0))
        self.progress_generate.stop()

    def create_instance_optimization_widgets(self):
        # Frame for instance optimization
        optimization_frame = ttk.LabelFrame(self.instance_optimization_page, text="")
        optimization_frame.pack(fill="x", padx=10, pady=1)
        optimization_frame.configure(style="TLabelframe")
        optimization_frame['borderwidth'] = 0

        ttk.Label(optimization_frame, text="Select Instance File:").grid(row=0, column=0, sticky="w")
        self.project_file_path = tk.StringVar()
        ttk.Entry(optimization_frame, textvariable=self.project_file_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(optimization_frame, text="Browse", command=self.load_project_file).grid(row=0, column=2, padx=5)

        ttk.Label(optimization_frame, text="Select Solver:").grid(row=1, column=0, sticky="w")
        self.solver_choice = tk.StringVar(value="GA")
        solver_menu = ttk.OptionMenu(
            optimization_frame,
            self.solver_choice,
            "GA",
            "GA", "DE", "BRKGA", "Gurobi", "HEGCL", "Others",
            command=self.update_solver_params
        )
        solver_menu.grid(row=1, column=1)
        ttk.Button(optimization_frame, text="Load Default Parameters", command=self.load_default_params).grid(row=1, column=2, padx=5)

        self.param_labels = []
        self.param_entries = []
        self.solver_params_frame = ttk.Frame(optimization_frame)
        self.solver_params_frame.grid(row=2, column=0, columnspan=3)

        self.update_solver_params("GA")

        button_frame = ttk.Frame(optimization_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        ttk.Button(button_frame, text="Run Solver", command=self.run_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Pause", command=self.pause_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Continue", command=self.continue_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_solver).pack(side="left", padx=5)

        ttk.Label(optimization_frame, text="Solver Output:").grid(row=4, column=0, columnspan=2, sticky="w")
        self.solver_output = tk.Text(optimization_frame, height=10, width=60, wrap="word", bg=self.entry_bg, fg=self.label_fg, font=(self.font_family, self.font_size))
        self.solver_output.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

        # Move progress bar to this page
        self.progress_optimize = ttk.Progressbar(optimization_frame, mode="indeterminate")
        self.progress_optimize.grid(row=6, column=0, columnspan=3, pady=(10, 0))
        self.progress_optimize.stop()

    def create_results_analysis_widgets(self):
        # Frame for results analysis
        analysis_frame = ttk.LabelFrame(self.results_analysis_page, text="")
        analysis_frame.pack(fill="x", padx=10, pady=1)
        analysis_frame.configure(style="TLabelframe")
        analysis_frame['borderwidth'] = 0

        ttk.Label(analysis_frame, text="Select Analysis Type:").grid(row=0, column=0, sticky="w", pady=(5, 0))
        self.analysis_type = tk.StringVar(value="Compare Portfolio Values and Runtimes")
        analysis_type_menu = ttk.OptionMenu(
            analysis_frame,
            self.analysis_type,
            "Compare Portfolio Values and Runtimes",
            "Compare Portfolio Values and Runtimes",
            "Visualize Convergence Graphs",
            "Analyze Optimized Portfolio"
        )
        analysis_type_menu.grid(row=0, column=1, sticky="w", pady=(5, 0))
        analysis_type_menu.configure(takefocus=False)

        ttk.Label(analysis_frame, text="Select Comparison Algorithms:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.algorithm_vars = {
            "Gurobi": tk.BooleanVar(value=False),
            "GA": tk.BooleanVar(value=False),
            "DE": tk.BooleanVar(value=False),
            "BRKGA": tk.BooleanVar(value=False),
            "HEGCL": tk.BooleanVar(value=False),
            "Others": tk.BooleanVar(value=False)
        }
        checkbox_frame = ttk.Frame(analysis_frame)
        checkbox_frame.grid(row=1, column=1, sticky="w", pady=(5, 0))

        # Display algorithms in two columns
        column_count = 2
        for i, (algorithm, var) in enumerate(self.algorithm_vars.items()):
            ttk.Checkbutton(
                checkbox_frame, text=algorithm, variable=var, takefocus=False
            ).grid(row=i // column_count, column=i % column_count, sticky="w", padx=5, pady=2)

        # Place custom_algorithm_name in the next column of the last row
        self.custom_algorithm_name = tk.StringVar(value="")
        ttk.Entry(checkbox_frame, textvariable=self.custom_algorithm_name, width=30).grid(
            row=(len(self.algorithm_vars) - 1) // column_count, column=column_count, sticky="w", padx=5, pady=2
        )

        ttk.Label(analysis_frame, text="Select Instances:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.selected_instances = tk.Listbox(analysis_frame, selectmode=tk.SINGLE, height=1, width=50, exportselection=False)
        self.selected_instances.grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Button(analysis_frame, text="Browse", command=self.select_analysis_instances).grid(row=2, column=2, padx=5)

        ttk.Button(analysis_frame, text="Analyze Results", command=self.analyze_results).grid(row=3, column=0, columnspan=3, pady=5)

        ttk.Label(analysis_frame, text="Analysis Summary:").grid(row=5, column=0, columnspan=3, sticky="w")
        self.analysis_summary = tk.Text(analysis_frame, height=5, width=60, wrap="word", bg=self.entry_bg, fg=self.label_fg, font=(self.font_family, self.font_size))
        self.analysis_summary.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

        # Move progress bar here
        self.progress_analysis = ttk.Progressbar(analysis_frame, mode="indeterminate")
        self.progress_analysis.grid(row=7, column=0, columnspan=3, pady=(10, 0))
        self.progress_analysis.stop()

        # Create a frame to hold the results table and convergence graph side by side
        results_and_graph_frame = ttk.Frame(analysis_frame)
        results_and_graph_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        # Add the results table to the left side
        ttk.Label(results_and_graph_frame, text="Comparison Results:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.results_table = ttk.Treeview(
            results_and_graph_frame,
            columns=("Algorithm", "Mean Values", "Mean Time"),
            show="headings",
            height=10,
            style="Treeview"
        )
        self.results_table.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=5)
        self.results_table.heading("Algorithm", text="Algorithm")
        self.results_table.heading("Mean Values", text="Mean Values")
        self.results_table.heading("Mean Time", text="Mean Time")
        self.results_table.column("Algorithm", width=120)
        self.results_table.column("Mean Values", width=120)
        self.results_table.column("Mean Time", width=120)

        # Add the convergence graph placeholder to the right side
        ttk.Label(results_and_graph_frame, text="Graphs:").grid(row=0, column=1, sticky="w", pady=(0, 5))
        self.convergence_graph_canvas = None
        self.convergence_graph_placeholder = tk.Frame(results_and_graph_frame, bg=self.entry_bg, width=400, height=300)
        self.convergence_graph_placeholder.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=5)

    def create_dynamic_instance_optimization_widgets(self):
        # Frame for dynamic instance optimization
        dynamic_frame = ttk.LabelFrame(self.dynamic_instance_optimization_page, text="")
        dynamic_frame.pack(fill="x", padx=10, pady=1)
        dynamic_frame.configure(style="TLabelframe")
        dynamic_frame['borderwidth'] = 0

        ttk.Label(dynamic_frame, text="Select Optimized Instance File:").grid(row=0, column=0, sticky="w")
        self.dynamic_instance_file_path = tk.StringVar()
        ttk.Entry(dynamic_frame, textvariable=self.dynamic_instance_file_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(dynamic_frame, text="Browse", command=self.load_dynamic_instance_file).grid(row=0, column=2, padx=5)

        # Add new label and file selection for current portfolio file
        ttk.Label(dynamic_frame, text="Select Current (Optimized) Portfolio File:").grid(row=1, column=0, sticky="w")
        self.current_portfolio_file_path = tk.StringVar()
        ttk.Entry(dynamic_frame, textvariable=self.current_portfolio_file_path, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(dynamic_frame, text="Browse", command=self.load_current_portfolio_file).grid(row=1, column=2, padx=5)

        ttk.Label(dynamic_frame, text="Add Dynamics:").grid(row=2, column=0, sticky="w", pady=(10, 0))

        self.dynamic_factors_frame = ttk.Frame(dynamic_frame)
        self.dynamic_factors_frame.grid(row=3, column=0, columnspan=3)

        # Add dynamic factors
        ttk.Label(self.dynamic_factors_frame, text="Current Year (2-20):").grid(row=0, column=0, sticky="w")
        self.current_year = tk.IntVar(value="")
        ttk.Entry(self.dynamic_factors_frame, textvariable=self.current_year).grid(row=0, column=1, padx=5)

        ttk.Label(self.dynamic_factors_frame, text="New Budget (the ratio to the previous budget):").grid(row=1, column=0, sticky="w")
        self.new_budget = tk.DoubleVar(value="")
        ttk.Entry(self.dynamic_factors_frame, textvariable=self.new_budget).grid(row=1, column=1, padx=5)

        ttk.Label(self.dynamic_factors_frame, text="Removed Projects Indexes (comma-separated):").grid(row=2, column=0, sticky="w")
        self.removed_projects = tk.StringVar(value="")
        ttk.Entry(self.dynamic_factors_frame, textvariable=self.removed_projects).grid(row=2, column=1, padx=5)

        ttk.Label(dynamic_frame, text="Select Solver:").grid(row=4, column=0, sticky="w")
        self.dynamic_solver_choice = tk.StringVar(value="GA")
        dynamic_solver_menu = ttk.OptionMenu(dynamic_frame, self.dynamic_solver_choice, "GA", "GA", "DE", "BRKGA", "Gurobi", "HEGCL", "Others", command=self.update_dynamic_solver_params)
        dynamic_solver_menu.grid(row=4, column=1)
        ttk.Button(dynamic_frame, text="Load Default Parameters", command=self.load_default_params_for_dynamic).grid(row=4, column=2, padx=5)

        self.dynamic_param_labels = []
        self.dynamic_param_entries = []
        self.dynamic_solver_params_frame = ttk.Frame(dynamic_frame)
        self.dynamic_solver_params_frame.grid(row=5, column=0, columnspan=3)

        self.update_dynamic_solver_params("GA")

        button_frame = ttk.Frame(dynamic_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)
        ttk.Button(button_frame, text="Run Solver", command=self.run_dynamic_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Pause", command=self.pause_dynamic_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Continue", command=self.continue_dynamic_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_dynamic_solver).pack(side="left", padx=5)

        ttk.Label(dynamic_frame, text="Solver Output:").grid(row=7, column=0, columnspan=2, sticky="w")
        self.dynamic_solver_output = tk.Text(dynamic_frame, height=10, width=60, wrap="word", bg=self.entry_bg, fg=self.label_fg, font=(self.font_family, self.font_size))
        self.dynamic_solver_output.grid(row=8, column=0, columnspan=3, padx=5, pady=5)

        self.progress_dynamic_optimize = ttk.Progressbar(dynamic_frame, mode="indeterminate")
        self.progress_dynamic_optimize.grid(row=9, column=0, columnspan=3, pady=(10, 0))
        self.progress_dynamic_optimize.stop()

    def load_current_portfolio_file(self):
        filepath = filedialog.askopenfilename(title="Select Current Portfolio File",
                                              filetypes=[("pkl Files", "*.pkl"), ("All Files", "*.*")])
        if filepath:
            if os.path.exists(filepath):
                self.current_portfolio_file_path.set(filepath)
                self.dynamic_solver_output.insert(tk.END, f"Current portfolio file loaded: {filepath}\n")
                self.solver_output.update_idletasks()
            else:
                messagebox.showerror("File Error", "Selected file does not exist or file is not selected.")

    def select_instance_save_path(self):
        directory = filedialog.askdirectory(title="Select Folder to Save Instance")
        if directory:
            self.instance_save_path.set(directory)

    def select_result_save_path(self):
        directory = filedialog.askdirectory(title="Select Folder to Save Result")
        if directory:
            self.result_save_path.set(directory)

    def update_solver_params(self, solver_name):
        for label in self.param_labels:
            label.destroy()
        for entry in self.param_entries:
            entry.destroy()

        self.param_labels.clear()
        self.param_entries.clear()
        self.solver_params = []

        if solver_name == "Gurobi":
            param_names = ["Time Limit", "MIP Gap"]
        elif solver_name == "HEGCL":
            param_names = ["Runs", "Group size", "Time Limit", "Population Size", "Crossover Rate", "Mutation Rate"]
        elif solver_name == "GA":
            param_names = ["Runs", "Population Size", "Max Evaluations", "Crossover Rate", "Mutation Rate"]
        elif solver_name == "DE":
            param_names = ["Runs", "Population Size", "Max Evaluations", "Crossover Rate", "Differential weight"]
        elif solver_name == "BRKGA":
            param_names = ["Runs", "Population Size", "Max Evaluations"]
        elif solver_name == "Others":
            param_names = ["Custom Name"]
        else:
            param_names = []

        for i, name in enumerate(param_names):
            label = ttk.Label(self.solver_params_frame, text=f"{name}:")
            label.grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(self.solver_params_frame)
            entry.grid(row=i, column=1)
            self.param_labels.append(label)
            self.param_entries.append(entry)
            self.solver_params.append(entry)

    def load_default_params(self):
        solver_name = self.solver_choice.get()
        defaults = {
            "Gurobi": ["600", "0.01"],
            "HEGCL": ["1", "600", "200", "100", "0.8", "0.1"],
            "GA": ["1", "100", "250000", "0.8", "0.1"],
            "DE": ["1", "100", "250000", "0.5", "0.1"],
            "BRKGA": ["1", "100", "250000"]
        }
        if solver_name in defaults:
            for entry, value in zip(self.param_entries, defaults[solver_name]):
                entry.delete(0, tk.END)
                entry.insert(0, value)

    def update_dynamic_solver_params(self, solver_name):
        for label in self.dynamic_param_labels:
            label.destroy()
        for entry in self.dynamic_param_entries:
            entry.destroy()

        self.dynamic_param_labels.clear()
        self.dynamic_param_entries.clear()
        self.dynamic_solver_params = []

        if solver_name == "Gurobi":
            param_names = ["Time Limit", "MIP Gap"]
        elif solver_name == "HEGCL":
            param_names = ["Runs", "Group size", "Time Limit", "Population Size", "Crossover Rate", "Mutation Rate"]
        elif solver_name == "GA":
            param_names = ["Runs", "Population Size", "Max Evaluations", "Crossover Rate", "Mutation Rate"]
        elif solver_name == "DE":
            param_names = ["Runs", "Population Size", "Max Evaluations", "Crossover Rate", "Differential weight"]
        elif solver_name == "BRKGA":
            param_names = ["Runs", "Population Size", "Max Evaluations"]
        elif solver_name == "Others":
            param_names = ["Custom Name"]
        else:
            param_names = []

        for i, name in enumerate(param_names):
            label = ttk.Label(self.dynamic_solver_params_frame, text=f"{name}:")
            label.grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(self.dynamic_solver_params_frame)
            entry.grid(row=i, column=1)
            self.dynamic_param_labels.append(label)
            self.dynamic_param_entries.append(entry)
            self.dynamic_solver_params.append(entry)

    def load_default_params_for_dynamic(self):
        solver_name = self.dynamic_solver_choice.get()
        defaults = {
            "Gurobi": ["600", "0.01"],
            "HEGCL": ["1", "600", "200", "100", "0.8", "0.1"],
            "GA": ["1", "100", "250000", "0.8", "0.1"],
            "DE": ["1", "100", "250000", "0.5", "0.1"],
            "BRKGA": ["1", "100", "250000"]
        }
        if solver_name in defaults:
            for entry, value in zip(self.dynamic_param_entries, defaults[solver_name]):
                entry.delete(0, tk.END)
                entry.insert(0, value)

    def generate_instance(self):
        self.progress_generate.start()
        threading.Thread(target=self._generate_instance).start()

    def show_path_error_dialog(self):
        messagebox.showerror("Path Error", "Instance save path is not selected.\n Please set the path first.")
        self.notebook.select(self.configuration_page)

    def _generate_instance(self):
        save_path = self.instance_save_path.get()
        if not save_path:
            self.show_path_error_dialog()
            return
        msg = (f"Generating instance {self.instance_index.get()} with {self.num_projects.get()} projects, {self.planning_years.get()} years, "
               f"budget proportion {self.budget_prop.get()}, and discount rate {self.discount_rate.get()}")
        self.instance_generation_output.insert(tk.END, msg + "\n")
        self.instance_generation_output.update_idletasks()

        try:
            generate_ppssp_instance(save_path, self.instance_index.get(), self.num_projects.get(), self.planning_years.get(), self.budget_prop.get(), self.discount_rate.get())
            self.instance_generation_output.insert(tk.END, f"Instance {self.instance_index.get()} generated successfully.\n")
            self.instance_generation_output.insert(tk.END,f"Instance {self.instance_index.get()} saved to the {save_path}.\n")
        except Exception as e:
            self.instance_generation_output.insert(tk.END, f"Error: {str(e)}\n")
        finally:
            self.after(0, self.progress_generate.stop)

    def load_project_file(self):
        filepath = filedialog.askopenfilename(title="Select Project File",
                                              filetypes=[("pkl Files", "*.pkl"), ("All Files", "*.*")])
        if filepath:
            if os.path.exists(filepath):
                self.project_file_path.set(filepath)
                self.solver_output.insert(tk.END, f"Project file loaded: {filepath}\n")
                self.solver_output.update_idletasks()
            else:
                messagebox.showerror("File Error", "Selected file does not exist or file is not selected.")

    def run_solver(self):
        self.solver_output.delete("1.0", tk.END)  # Clear previous output
        self.pause_event.set()  # Ensure not paused
        self.stop_event.clear()  # Ensure not stopped
        self.progress_optimize.start()  # Start progress bar
        if self.solver_thread is None or not self.solver_thread.is_alive():
            self.solver_thread = threading.Thread(target=self._run_solver_task)
            self.solver_thread.start()

    def pause_solver(self):
        self.pause_solver_generic(self.pause_event, self.solver_output)

    def continue_solver(self):
        self.continue_solver_generic(self.pause_event, self.solver_output)

    def stop_solver(self):
        self.stop_solver_generic(self.stop_event, self.solver_output)

    def pause_dynamic_solver(self):
        self.pause_solver_generic(self.pause_event_dynamic, self.dynamic_solver_output)

    def continue_dynamic_solver(self):
        self.continue_solver_generic(self.pause_event_dynamic, self.dynamic_solver_output)

    def stop_dynamic_solver(self):
        self.stop_solver_generic(self.stop_event_dynamic, self.dynamic_solver_output)

    def pause_solver_generic(self, pause_event, output_frame):
        pause_event.clear()  # Pause the solver
        output_frame.insert(tk.END, "Solver paused.\n")
        output_frame.update_idletasks()

    def continue_solver_generic(self, pause_event, output_frame):
        pause_event.set()  # Resume the solver
        output_frame.insert(tk.END, "Solver continued.\n")
        output_frame.update_idletasks()

    def stop_solver_generic(self, stop_event, output_frame):
        stop_event.set()  # Stop the solver
        output_frame.insert(tk.END, "Solver stopped.\n")
        output_frame.update_idletasks()

    def _run_solver_task(self):
        solver = self.solver_choice.get()
        # Use custom name if "Others" is selected

        if not self.solver_params:
            messagebox.showerror("Solver Error", "Solver parameters are not set. Please configure the solver parameters.")
            return

        if solver == "Others":
            solver = self.solver_params[0].get().strip()
            param_values = []
        else:
            param_values = [entry.get() for entry in self.solver_params]
        project_file = self.project_file_path.get()
        save_dir = self.result_save_path.get()

        if not os.path.exists(project_file):
            messagebox.showerror("File Error", f"The project file does not exist:\n{project_file}. Please select the project file." )
            return
        if not os.path.exists(save_dir):
            messagebox.showerror("Save Path Error", "Result save path is not set. Please set the result save path.")
            self.notebook.select(self.configuration_page)
            return

        msg = f"Running {solver} with parameters: {param_values} on: {os.path.basename(project_file)}"
        self.solver_output.insert(tk.END, msg + "\n")
        self.solver_output.update_idletasks()

        try:
            results = run_ppssp_solver(
                project_file,
                solver,
                param_values,
                save_dir,
                pause_event=self.pause_event,
                stop_event=self.stop_event,
                gui_output=self.solver_output
            )

            print_results(results, solver, self.solver_output, param_values[0] if param_values else "", save_dir)

        except Exception as e:
            if self.stop_event and self.stop_event.is_set():
                self.solver_output.insert(tk.END, "Solver stopped by user.\n")
            else:
                self.solver_output.insert(tk.END, f"Error: {str(e)}\n")
                self.solver_output.update_idletasks()
        finally:
            self.after(0, self.progress_optimize.stop)  # Stop progress bar on the main thread

    def select_analysis_instances(self):
        filepaths = filedialog.askopenfilenames(title="Select Instance Files",
                                                filetypes=[("pkl Files", "*.pkl"), ("All Files", "*.*")])
        if filepaths:
            # Clear existing selections
            self.selected_instances.delete(0, tk.END)
            # Add new file paths to the Listbox
            for filepath in filepaths:
                self.selected_instances.insert(tk.END, filepath)

            # Automatically select all items in the Listbox
            self.selected_instances.select_set(0, tk.END)

    def analyze_results(self):
        self.progress_analysis.start()  # Start progress bar
        threading.Thread(target=self._analyze_results_task).start()

    def _analyze_results_task(self):
        selected_algorithms = [alg for alg, var in self.algorithm_vars.items() if var.get()]
        if "Others" in selected_algorithms:
            custom_name = self.custom_algorithm_name.get().strip()
            if custom_name:
                selected_algorithms.remove("Others")
                selected_algorithms.append(custom_name)
        if not selected_algorithms:
            messagebox.showerror("Selection Error", "No comparison algorithms selected.")
            self.after(0, self.progress_analysis.stop)  # Stop progress bar on the main thread
            return

        selected_instances = [self.selected_instances.get(i) for i in self.selected_instances.curselection()]
        if not selected_instances:
            messagebox.showerror("Selection Error", "No instances selected.")
            self.after(0, self.progress_analysis.stop)  # Stop progress bar on the main thread
            return

        analysis_type = self.analysis_type.get()
        self.analysis_summary.insert(tk.END, f"Performing analysis: {analysis_type}\n")
        self.analysis_summary.insert(tk.END, f"Selected algorithms: {', '.join(selected_algorithms)}\n")
        self.analysis_summary.insert(tk.END, f"Selected instances:\n")
        for instance in selected_instances:
            instance_name = os.path.basename(instance).split("/")[-1]
            instance_name = instance_name.rsplit('.', 1)[0]
            self.analysis_summary.insert(tk.END, f"  - {instance}\n")
            # Simulate analysis
            if analysis_type == "Compare Portfolio Values and Runtimes":
                # Call the comparison function
                self.results_table.delete(*self.results_table.get_children())
                mean_fitness, std_fitness, mean_time, std_time = get_experimental_data(selected_algorithms, instance_name, self.result_save_path.get(), analysis_type, runs=1, pause_event=None, stop_event=None)

                # Populate the results table
                for idx, alg in enumerate(selected_algorithms):
                    self.results_table.insert(
                        "", "end",
                        values=(
                            alg,
                            f"{mean_fitness[idx]:.2f}",
                            f"{mean_time[idx]:.2f}"
                        )
                    )
            elif analysis_type == "Visualize Convergence Graphs":
                self.plot_convergence_graph(selected_algorithms, instance_name)
            elif analysis_type == "Analyze Optimized Portfolio":
                self.analyze_optimized_portfolio(selected_algorithms, instance, instance_name)

        self.analysis_summary.insert(tk.END, f"Completed analysis: {analysis_type}\n\n")
        self.after(0, self.progress_analysis.stop)  # Stop progress bar on the main thread

    def plot_convergence_graph(self, algorithms, instance_name):
        # Remove the existing canvas if it exists
        if self.convergence_graph_canvas:
            self.convergence_graph_canvas.get_tk_widget().destroy()

        # Create a matplotlib figure
        fig = Figure(figsize=(4, 3.5), dpi=100)  # Limit the size of the graph
        ax = fig.add_subplot(111)

        for alg in algorithms:
            time_history, fitness_history = get_experimental_data(
                [alg],
                instance_name,
                self.result_save_path.get(),
                "Visualize Convergence Graphs",
                runs=1,
                pause_event=None,
                stop_event=None
            )
            fitness_history = [value / 1000 for value in fitness_history]
            ax.plot(time_history, fitness_history, label=alg)

        ax.set_title("Convergence Graph")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Portfolio Value (k)")
        ax.legend()

        # Adjust layout to ensure labels are visible
        fig.subplots_adjust(bottom=0.2)  # Increase bottom margin to make space for the x-axis label
        fig.subplots_adjust(left=0.2)

        # Save the graph as an image file
        convergence_graph_save_path = os.path.join(self.result_save_path.get(), instance_name, f"{instance_name}_convergence_graph.png")
        fig.savefig(convergence_graph_save_path)
        self.analysis_summary.insert(tk.END, f"Convergence graph saved to: {convergence_graph_save_path}\n")

        # Embed the plot into the placeholder area with margins
        self.convergence_graph_canvas = FigureCanvasTkAgg(fig, master=self.convergence_graph_placeholder)
        canvas_widget = self.convergence_graph_canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)  # Add margins around the canvas
        self.convergence_graph_canvas.draw()

    def analyze_optimized_portfolio(self, algorithms, instance_pickle, instance_name):
        # Remove the existing canvas if it exists
        if self.convergence_graph_canvas:
            self.convergence_graph_canvas.get_tk_widget().destroy()

        for alg in algorithms:
            if alg == "Gurobi":
                t_limitation = 600
            elif alg == "HEGCL":
                t_limitation = 200
            portfolio_pickle = get_portfolio_name(self.result_save_path.get(), instance_name, alg, t_limitation=t_limitation,
                                          mip_gap=0.01, size_groups=600, run=0)
            if not os.path.exists(portfolio_pickle):
                messagebox.showerror("File Error", f"Portfolio file does not exist:\n{portfolio_pickle}")
                return
            result_path = get_result_path(self.result_save_path.get(), instance_name, alg, t_limitation=t_limitation, mip_gap=0.01, size_groups=600)

            analyze_from_pickles(portfolio_pickle, instance_pickle, instance_name, display=True, plot=True, plot_dir=result_path,
                             yearly_project_analysis=False)

        # Load the generated plot (assuming it's saved as a PNG file)
        plot_file = os.path.join(result_path, f"{instance_name}-gantt_chart.png")
        if os.path.exists(plot_file):
            # Create a matplotlib figure to display the plot
            fig = Figure(figsize=(4, 3.5), dpi=100)
            ax = fig.add_subplot(111)
            img = plt.imread(plot_file)
            ax.imshow(img)
            ax.axis('off')  # Hide axes for better visualization

            # Embed the plot into the placeholder area
            self.convergence_graph_canvas = FigureCanvasTkAgg(fig, master=self.convergence_graph_placeholder)
            canvas_widget = self.convergence_graph_canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)
            self.convergence_graph_canvas.draw()
        else:
            self.analysis_summary.insert(tk.END, f"Plot not found: {plot_file}\n")

    def load_dynamic_instance_file(self):
        filepath = filedialog.askopenfilename(title="Select Dynamic Instance File",
                                              filetypes=[("pkl Files", "*.pkl"), ("All Files", "*.*")])
        if filepath:
            if os.path.exists(filepath):
                self.dynamic_instance_file_path.set(filepath)
                self.dynamic_solver_output.insert(tk.END, f"Project file loaded: {filepath}\n")
                self.dynamic_solver_output.update_idletasks()
            else:
                messagebox.showerror("File Error", "Selected file does not exist or file is not selected.")

    def run_dynamic_solver(self):
        self.dynamic_solver_output.delete("1.0", tk.END)  # Clear previous output
        self.pause_event_dynamic.set()  # Ensure not paused
        self.stop_event_dynamic.clear()  # Ensure not stopped
        self.progress_dynamic_optimize.start()  # Start progress bar
        if self.dynamic_solver_thread is None or not self.dynamic_solver_thread.is_alive():
            self.dynamic_solver_thread = threading.Thread(target=self._run_dynamic_solver_task)
            self.dynamic_solver_thread.start()

    def _run_dynamic_solver_task(self):
        solver = self.dynamic_solver_choice.get()

        if not self.dynamic_solver_params:
            messagebox.showerror("Solver Error", "Dynamic solver parameters are not set. Please configure the dynamic solver parameters.")
            return

        dynamic_param_values = [entry.get() for entry in self.dynamic_solver_params]
        dynamic_instance_file = self.dynamic_instance_file_path.get()
        optimized_portfolio_file = self.current_portfolio_file_path.get()
        save_dir = self.result_save_path.get()

        if not os.path.exists(dynamic_instance_file):
            messagebox.showerror("File Error", f"The dynamic instance file does not exist:\n{dynamic_instance_file}")
            return
        if not os.path.exists(save_dir):
            messagebox.showerror("Save Path Error", "Result save path is not set. Please set the result save path.")
            self.notebook.select(self.configuration_page)
            return
        if not os.path.exists(optimized_portfolio_file):
            messagebox.showerror("File Error", f"The optimized portfolio file does not exist:\n{optimized_portfolio_file}")
            return

        msg = f"Running solver {solver} on: {os.path.basename(dynamic_instance_file)} with dynamics applied"
        self.dynamic_solver_output.insert(tk.END, msg + "\n")
        self.dynamic_solver_output.update_idletasks()

        try:
            results = run_dynamic_ppssp_solver(
                dynamic_instance_file,
                optimized_portfolio_file,
                solver,
                dynamic_param_values,
                save_dir,
                current_year=self.current_year.get(),
                new_budget=self.new_budget.get(),
                removed_projects=self.removed_projects.get(),
                pause_event=self.pause_event_dynamic,
                stop_event=self.stop_event_dynamic,
                gui_output=self.dynamic_solver_output
            )

            print_results(results, solver, self.dynamic_solver_output, dynamic_param_values[0], save_dir)

            # self.dynamic_solver_output.insert(tk.END, f"Dynamic solver {solver} completed successfully.\n")
        except Exception as e:
            if self.stop_event and self.stop_event.is_set():
                self.dynamic_solver_output.insert(tk.END, "Solver stopped by user.\n")
            else:
                self.dynamic_solver_output.insert(tk.END, f"Error: {str(e)}\n")
                self.dynamic_solver_output.update_idletasks()
        finally:
            self.after(0, self.progress_dynamic_optimize.stop)  # Stop progress bar on the main thread

    def open_help(self):
        # TODO: Open a help file or website
        help_url = "https://github.com/Jingliu2605/PPSSP-Solver/blob/main/README.md"  # Replace with the actual help URL or file path
        try:
            import webbrowser
            webbrowser.open(help_url)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open help: {str(e)}")


def print_results(results, solver, solver_output, runs, save_dir):
    if solver == "Gurobi":
        fitness, elapsed_time, gap = results
        solver_output.insert(tk.END, f"Fitness value: {fitness}\n")
        solver_output.insert(tk.END, f"Time: {elapsed_time}\n")
        solver_output.insert(tk.END, f"Gap: {gap}\n")
    else:
        fitness, elapsed_time, generation, eval = results
        solver_output.insert(tk.END, f"Runs: {int(runs)}\n")
        solver_output.insert(tk.END, f"Fitness value: {fitness}\n")
        solver_output.insert(tk.END, f"Time: {elapsed_time}\n")
        solver_output.insert(tk.END, f"Generation: {generation}\n")
        solver_output.insert(tk.END, f"Eval: {eval}\n")

    solver_output.insert(tk.END, f"Results saved in {save_dir}\n")
    solver_output.insert(tk.END, f"{solver} completed\n")
    solver_output.update_idletasks()


def main():
    app = PPSSPGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
