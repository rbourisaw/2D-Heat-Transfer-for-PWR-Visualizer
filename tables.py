import numpy as np
from iapws import IAPWS97

# Choose the T-range that your coolant will ever reach
T_min = 289.33+273.15
T_max = 647.0        # cannot go above critical
N_T = 1000                  # resolution (increase if needed)

T_grid = np.linspace(T_min, T_max, N_T)

# Saturation pressure table
P_sat_table = np.array([IAPWS97(T=T, x=0).P for T in T_grid])
print(P_sat_table)

np.savez("water_satP_table.npz", T_grid=T_grid, P_sat=P_sat_table)