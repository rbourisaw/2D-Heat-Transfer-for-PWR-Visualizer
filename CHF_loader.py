import numpy as np
import pandas as pd

def CHF_load():
    df = pd.read_excel('LUT2006.xls', skiprows=1, sheet_name='main')

    # Remove blank rows and drop label row
    df_cleaned = df.dropna(subset=["P", "G"], how="all").iloc[1:].reset_index(drop=True)

    # Axes
    P_vals = (df_cleaned["P"].values * 1000)    # convert kPa → Pa
    G_vals = df_cleaned["G"].values
    X_vals = df.iloc[0, 2:].values

    # Extract CHF table directly (no flattening)
    CHF_vals = df_cleaned.iloc[:, 2:].values * 1000   # kW/m² → W/m²

    # CHF_vals currently has shape (#rows, #qualities)
    # but rows represent combined (P,G) points.
    # We must reshape by grouping rows with same P,G.

    # Find unique values
    unique_P = np.unique(P_vals)
    unique_G = np.unique(G_vals)

    # Allocate final 3D array
    CHF_grid = np.full((len(unique_P), len(unique_G), len(X_vals)), np.nan)

    # Build index lookup tables
    P_to_idx = {p: i for i, p in enumerate(unique_P)}
    G_to_idx = {g: i for i, g in enumerate(unique_G)}

    # Populate CHF_grid using indexing
    for i in range(len(df_cleaned)):
        P_loc = df_cleaned.iloc[i]["P"] * 1000   # convert kPa → Pa
        G_loc = df_cleaned.iloc[i]["G"]
        pi = P_to_idx[P_loc]
        gi = G_to_idx[G_loc]
        CHF_grid[pi, gi, :] = df_cleaned.iloc[i, 2:].values * 1000  # convert kW/m² → W/m²
    return CHF_grid, unique_P, unique_G, X_vals