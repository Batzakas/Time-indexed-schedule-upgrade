"""
Default parameters for the batch-scheduling MILP (makespan minimization).
Flat module of constants, overridable per-run via CLI flags in
core/model_tester.py or src/data/instance_generator.py.
"""

TIME_LIMIT = 2 * 60 * 60          
MIP_GAP = None                  
DEFAULT_MAX_MEMORY = None         
DEFAULT_GUROBI_LICENSE = "gurobi.lic"

M_DEFAULT = 1.0                  
# synthetic instance generator defaults (src/data/instance_generator.py)
DEFAULT_GRAPH_TYPE = "erdos_renyi"   # "erdos_renyi" | "ring" | "grid"
DEFAULT_N_NODES = 8
DEFAULT_EDGE_PROB = 0.5               
DEFAULT_U_FRACTION = 0.3              # fraction of edges that are upgradeable
DEFAULT_N_COMMODITIES = 3
DEFAULT_DEMAND_RANGE = (1.0, 2.0)
DEFAULT_L_RANGE = (1, 3)            
DEFAULT_U0_RANGE = (3.0, 5.0)      
DEFAULT_UPGRADE_FACTOR_RANGE = (1.5, 3.0)  # u1_e = u0_e * factor
DEFAULT_FIXED_CAP_RANGE = (6.0, 10.0)
DEFAULT_SEED = 0

HMAX_SLACK = 0
