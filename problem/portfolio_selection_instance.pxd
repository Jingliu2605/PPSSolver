import numpy as np
cimport numpy as np
from problem.project import Project
from problem.project cimport Project


cdef class PortfolioSelectionInstance:

    cdef:
        public Project[:] projects
        public np.double_t[:] budget
        public np.double_t[:] capability_stream_budgets
        public np.double_t[:] initiation_budget
        public np.double_t[:] ongoing_budget

        public int planning_window
        public str identifier
        public int num_projects
        public int budget_window
        public double discount_rate

        public object parameters
        public tuple initiation_range