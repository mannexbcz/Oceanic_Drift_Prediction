import numpy as np 
import pandas as pd
import torch
from utils.read_data import wind_interpolated, water_interpolated
from utils.param_alpha import general_alpha
from utils.RK4 import compute_position

def physical_model(init_pos, time_init, dict_path):
    init_lat = init_pos[0,0].item()
    init_lon = init_pos[1,0].item()
    init_pos = np.array([init_lon, init_lat])

    # Get interpolated initial data
    u10_interpolation, v10_interpolation = wind_interpolated(dict_path['PATH_WIND'])
    water_u_interpolation, water_v_interpolation = water_interpolated(dict_path['PATH_WATER'])

    # Alpha
    alpha = general_alpha()

    # Drift velocity
    u_drift = u_drift_linear_matrix(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation)

    # Update position --> position at t+1h
    longitudes, latitudes, _ = compute_position(u_drift, init_pos, time_init,1,1) # make only one time step of 1h 
    final_lat, final_lon = latitudes[-1], longitudes[-1]
    final_position = torch.tensor([[final_lat], [final_lon]], dtype=torch.double)

    return final_position