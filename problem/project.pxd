import array
from cpython cimport array

import numpy as np
cimport numpy as np
import array

cimport numpy as np
import numpy as np
from cpython cimport array

cdef class Project:

    cdef:
        public str project_name
        public array.array cost_raw
        public np.double_t[:] value
        public int duration
        public np.ndarray prerequisite_list
        public np.ndarray successor_list
        public np.ndarray exclusion_list
        public double total_cost
        public tuple completion_window
        public double total_value
        public int capability_stream

        inline bint has_prerequisites(self):
           return self.prerequisite_list.shape[0] > 0

        inline bint has_successors(self):
            return self.successor_list.shape[0] > 0

        inline double cost_value_ratio(self):
            return self.total_cost / np.sum(self.value)