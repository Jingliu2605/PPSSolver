# PPSSP Solver GUI (PyPPS)

PPSSolver is an open-source Python software for solving Project Portfolio Selection and Scheduling Problems (PPSSP) using optimization techniques like Gurobi, GA, and DE. 

It is distributed as an executable desktop application (PPSSolver.exe) for ease of use and as a source code package for customization.

## Key functionalities

* Instance generation
* Multi-solver integration: Gurobi, GA, DE, BRKGA, HEGCL  
* Dynamic re-optimization for budget and project changes
* Result analysis and visualization through GUI

## System Requirements
PPSSolver requires Python 3.9 or higher and the packages listed in requirements.txt.
For exact optimization, Gurobi must be installed separately and activated with a valid academic license.

## Installation
```bash
git clone https://github.com/Jingliu2605/PPSSolver.git
cd PPSSolver
pip install -r requirements.txt
python Gui.py
```
Or, download and run PPSSolver.exe for the desktop version

[Download PPSSolver.exe](https://drive.google.com/file/d/184-1gYs-JMi-M8GPfJYOWnihBVzQirMf/view?usp=drive_link)

## Usage Guide:
PPSSolver can be used through running Gui.py via Python scripts or directly using executable desktop application.

### Using the GUI
1. **Set file paths**  
   Open the *Paths Configuration* page and specify the directories for storing instance files ("Instance Path")  and result files ("Result Path").
   
   This is a mandatory step before generating or optimizing instances.


2. **Generate instance**  
   Go to the *Instance Generation* page to create a new PPSSP instance with customizable parameters. You can define the following key parameters through the GUI input fields:.  
   - **Instance Index**: A numeric ID used to label and differentiate multiple instances.  
   - **Number of Projects**: Specifies how many projects are to be included in the instance.  
   - **Planning Years**: Sets the duration of the project planning horizon.  
   - **Initial Budget Proportion (0–1)**: Defines the proportion of the total budget allocated initially.  
   - **Discount Rate (0–1)**: Represents the time-discounting factor used in the objective function.  

   Then click "Generate Instance" button to create the data. Generated instances are saved in a standardized format, named as ``PI\_\{Instance Index\}\_\{Number of Projects\}\_\{Planning Years\}\_\{Initial Budget Proportion\}\_\{Discount Rate\}.pkl'', in the specified instance path.


3. **Run optimization**  
   Go to the *Instance Optimization* page to configure and run algorithms on previously generated or externally defined PPSSP instance files that follows the required data format. 
   - Click **Browse** to select the target instance file.  
   - Use the **Select Solver** dropdown to choose one of the available algorithms, such as GA, DE, BRKGA, Gurobi, or HEGCL.  
   - Adjust solver-specific parameters (e.g., time limit, population size, maximum fitness evaluations, crossover and mutation rates) directly through the GUI.  
     - for Gurobi, you can set parameters **Time Limit (seconds)** and **MIP Gap**.
     - for metaheuristics, you can set common parameters like **Runs** **Population Size**, and **Result Display Frequency**, as well as algorithm-specific parameters:
       - for GA, you can configure parameters **Max Evaluationss**, **Crossover Rate**, **Mutation Rate**.
       - for DE, you can configure parameters **Max Evaluationss**, **Crossover Rate**, **Differential Weight**.
       - for BRKGA, you can configure parameters **Max Evaluationss**, **Bias**, **Elite Proportion**.
       - for HEGCL, you can configure parameters **Group size**, **Time Limit (seconds)**, **Crossover Rate**, **Mutation Rate**.
   - Control buttons allow interactive management during execution:  
     - **Run Solver**: Launches the optimization process.  
     - **Pause**, **Continue**, and **Stop**: Allow user control over long-running computations.  
   
   When optimization finishes, results are automatically saved in the results directory under a subfolder named after the instance. Outputs include:  
   - A results summary file (`.csv`)  
   - The optimized portfolio file (`.pkl`)  
   - Convergence data (`.csv`) for metaheuristics or solver logs (`.log`) for Gurobi  

   During execution, real-time progress for metaheuristic algorithms (e.g., GA, DE, HEGCL) is shown in the **Solver Output** area, displaying values such as `n_gen` (number of generations), `n_eval` (number of evaluations), `f_best` (best fitness), and `f_avg` (average fitness).


4. **Handle dynamic changes**   
   The *Dynamic Instance Optimization* page is for simulating real-world changes such as budget reductions or project cancellations.

   - Click **Browse**  to select the optimized instance file and its corresponding optimized portfolio file.  

   - **Add Dynamics** allows you to define the changing conditions:
     - **Current Year**: The point in time when changes occur.
     - **New Budget (the ratio to the previous budget)**: Updates the remaining funding for future years.
     - **Removed Projects Indexes**: A list of removed projects indexes representing projects that are canceled or no longer executable. 
   
   - Then choose an optimization algorithm (e.g., Gurobi, GA, HEGCL) and re-optimize the modified instance. 
   
   Real-time progress is again displayed in the **Solver Output** panel.  


5. **Analyze results**  
   The *Results Analysis* page provides tools to evaluate and compare optimization outcomes.  
   - Choose an **analysis type**, select algorithms to compare, and specify the target PPSSP instance. 
   Available analysis types include:  
     - **Compare Portfolio Values and Runtimes**: Benchmarks algorithm performance by displaying total portfolio values and computation times in tabular form.  
     - **Visualize Convergence Graphs**: Plots how the best fitness improves over iterations, showing convergence speed and stability.  
     - **Analyze Optimized Portfolio**: Provides a detailed view of a single optimized solution, including value and cost trends, and Gantt-style charts.  

   - Click **Analyze Results** to run the analysis and display results in the **Comparison Results** or **Graphs** section.  

   All generated figures and comparison files are saved automatically within the instance’s results subfolder for reproducibility.