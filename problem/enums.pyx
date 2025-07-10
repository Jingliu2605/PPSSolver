from enum import Enum


class SchedulingOrder(Enum):
    EARLIEST = "Earliest"
    LATEST = "Latest"
    CYCLIC = "Cyclic"
    STOCHASTIC_EARLIEST = "Stochastic Earliest"


class PortfolioOrdering(Enum):
    CV_RATIO = "Cost-Value"
    COST_ASC = "Cost (ascending)"
    COST_DESC = "Cost (descending)"
    VALUE_ASC = "Value (ascending)"
    VALUE_DESC = "Value (descending)"


class ValueFunction(Enum):
    COST = "Cost"
    RANDOM_COST = "Random cost"
    COST_DUR = "Cost-duration"
    RANDOM = "Random"


class SpreadDistribution(Enum):
    RAMPED = "Ramped"
    EVEN = "Even"
    WEIBULL = "Weibull"


class Optimizer(Enum):
    PermGA = "PermGA"
    BRKGA = "BRKGA"
    DE = "DE"
    GA = "GA"
    GUROBI = "Gurobi"
    SA = "SA"
    DE_BRKGA = "DE_BRKGA"
    Gurobi_DE = "Gurobi_DE"
    Gurobi_DE_Mutual = "Gurobi_DE_Mutual"
    Gurobi_DE_S = "Gurobi_DE_S"
    DE_Gurobi = "DE_Gurobi"
    DE_S_Gurobi = "DE_S_Gurobi"
    Gurobi_GA = "Gurobi_GA"
    GA_Gurobi = "GA_Gurobi"
    Gurobi_GA_Mutual = "Gurobi_GA_Mutual"
    Gurobi_BRKGA = "Gurobi_BRKGA"
    MMES = "MMES"
    CMAES = "CMAES"
    GTDE = "GTDE"
    GTDE_ST = "GTDE_ST"
    CCDE = "CCDE"
    CCDEPerm = "CCDEPerm"
    CCGAST = "CCGAST"
    CCGAPerm = "CCGAPerm"
    CBCC_DE = "CBCC_DE"
    DEST = "DEST"
    CCGAGurobi = "CCGAGurobi"
    JADEST = "JADEST"
    GA_DGrb = "GA_DGrb"
    AGA = "HEGCL"
    HGA = "HGA"
    JADE = "JADE"
    SAGA = "SAGA"  # original aga
    SGA = "SGA" # GA with the origianl repair method



class SchedulingStatus(Enum):
    PREREQUISITE_VIOL = "Prerequisite violation"
    EXCLUSION_VIOL = "Mutual exclusion violation"
    COMPLETION_WINDOW_VIOL = "Completion time window violation"
    BUDGET_VIOL = "Budget violation"
    SUCCESS = "Success"


class Weighting(Enum):
    VC_RATIO = "Value-to-cost ratio"
    VALUE = "Value"
    COST = "Cost"