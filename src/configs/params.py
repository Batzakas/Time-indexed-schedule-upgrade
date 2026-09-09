"""
Default parameters for the batch-scheduling MILP (makespan minimization).
Flat module of constants, overridable per-run via CLI flags in
core/model_tester.py or src/data/instance_generator.py.
"""

TIME_LIMIT = 2 * 60 * 60
MIP_GAP = None
DEFAULT_GUROBI_LICENSE = "gurobi.lic"

M_DEFAULT = 1.0                  
DEFAULT_GRAPH_TYPE = "erdos_renyi"   # "erdos_renyi" | "ring" | "grid"
DEFAULT_N_NODES = 8
DEFAULT_EDGE_PROB = 0.5               
DEFAULT_U_FRACTION = 0.3              
DEFAULT_N_COMMODITIES = 3
DEFAULT_DEMAND_RANGE = (1.0, 2.0)
DEFAULT_L_RANGE = (1, 3)            
DEFAULT_U0_RANGE = (3.0, 5.0)      
DEFAULT_UPGRADE_FACTOR_RANGE = (1.5, 3.0)  # u1_e = u0_e * factor
DEFAULT_FIXED_CAP_RANGE = (6.0, 10.0)
DEFAULT_SEED = 0

HMAX_SLACK = 0
REAL_TOPOLOGIES = ("test5", "dt12")

UPGRADE_TYPES = ("new_fiber_C", "new_fiber_CL", "band_upgrade", "core_upgrade")
UPGRADE_DURATION_DAYS = {
    "new_fiber_C": 0,     # t_C1
    "new_fiber_CL": 15,   # t_CL1
    "band_upgrade": 23,   # t_b
    "core_upgrade": 70,   # t_3C
}
UPGRADE_CAPACITY_MULTIPLIER = {
    "new_fiber_C": 2.0,
    "new_fiber_CL": 4.0,
    "band_upgrade": 3.0,
    "core_upgrade": 3.0,
}

TIME_UNIT_DAYS = 7  # 1 MILP time step (h) = 1 week
BASE_CHANNEL_CAPACITY = 80.0  # ~= C_BAND_SLOTS(320) / 4 slots-per-100G-channel
DEFAULT_DEMAND_CHANNELS_RANGE = (1.0, 4.0)  # 100-400G per commodity (DATARATE=[100])
DEFAULT_REAL_U_FRACTION = 0.3
DEFAULT_REAL_N_COMMODITIES = 5
