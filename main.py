"""
This code allows the computation of trajectories using physical and hybrid models. 

It takes as argument the path to a config file with paths to environmental data as well as minimum and maximum 
latitudes and longitudes. A second argument is required, namely the data-driven model required 
(str, letter between K and P)

It computes and plots the predicted trajectories.
"""

import yaml
import netCDF4 as nc
import numpy as np
import torch
import pandas as pd

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from utils.read_data import *
from utils.param_alpha import get_alpha, get_complex_alpha, compute_alpha_whole_trajectory,compute_alpha_matrix, get_alpha_averaged, update_alpha_GD, update_alpha_GD_stockes, compute_alpha_succesive_position
from models.linear_model import u_drift_linear, u_drift_linear_complex, u_drift_linear_matrix
from utils.RK4 import RK4_step, compute_position
from data_driven.compute_trajectory_hybrid import compute_trajectory_hybrid, compute_trajectory_hybrid_w_history
from models.stockes_model import u_drift_ocean_wind_stockes_matrix, u_drift_ocean_stockes_matrix,u_drift_stockes_only
from argparse import ArgumentParser

checkpoint_path_K = './checkpoints/checkpoint_model_K.ckpt'
checkpoint_path_L = './checkpoints/checkpoint_model_L.ckpt'
checkpoint_path_M = './checkpoints/checkpoint_model_M.ckpt'
checkpoint_path_N = './checkpoints/checkpoint_model_N.ckpt'
checkpoint_path_O = './checkpoints/checkpoint_model_O.ckpt'
checkpoint_path_P = './checkpoints/checkpoint_model_P.ckpt'

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument("--config", type = str)
    parser.add_argument("--model", type = str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = args.model
    
    # Get initial position and true (observed) trajectory
    pos_1, time1 = get_initial_position(config['PATH_DRIFT'], NOAA=True) 
    true_lon, true_lat, true_time = get_true_drift_positions(config['PATH_DRIFT'],NOAA=True)
    
    # Get duration of the trajectory
    nhours = get_number_hours(true_time)

    # Get interpolated environmental data
    u10_interpolation, v10_interpolation = wind_interpolated(config['PATH_WIND'])
    water_u_interpolation, water_v_interpolation = water_interpolated(config['PATH_WATER'])
    ust_interpolation, vst_interpolation = wave_interpolated(config['PATH_WAVES'])

    # Compute alpha for the physical model
    alpha = update_alpha_GD(config['PATH_DRIFT'], water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,alpha = 0.03, theta = 0.349066,  step=1, npoints=3, NOAA=True)
    alpha_stockes = update_alpha_GD_stockes(config['PATH_DRIFT'], water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, ust_interpolation, vst_interpolation, alpha = 0.03, theta = 0.349066,  step=1, npoints=3, NOAA = True)

    # Compute the drift velocity for the physical model
    u_drift = u_drift_linear_matrix(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation)
    u_stokes = u_drift_ocean_wind_stockes_matrix(alpha_stockes, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation, ust_interpolation, vst_interpolation)

    # Compute latitude and longitude predicted by the physical module
    longitudes, latitudes, time_final = compute_position(u_drift, pos_1, time1,1,10)
    longitudes_stokes, latitudes_stokes, time_final = compute_position(u_stokes, pos_1, time1,1,72)

    if model == 'K':
        config['checkpoint_test'] = checkpoint_path_K
        config['d_context'] = 50
        config['npoints'] = 32
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid(config,nhours, NOAA=True)
    elif model == 'L':
        config['checkpoint_test'] = checkpoint_path_L
        config['d_context'] = 100
        config['npoints'] = 32
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid(config,nhours, NOAA=True)
    elif model == 'M':
        # Note that for this model it is necessary to change in data_driven/models/hybrid_model the size of the first hidden layer
        # in the Hybrid_Model module from 16 to 64.
        config['checkpoint_test'] = checkpoint_path_M
        config['d_context'] = 50
        config['npoints'] = 64
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid(config,nhours, NOAA=True)
    elif model == 'N':
        config['checkpoint_test'] = checkpoint_path_N
        config['d_context'] = 50
        config['npoints'] = 32
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid(config,nhours, NOAA=True)
    elif model == 'O':
        config['checkpoint_test'] = checkpoint_path_O
        config['d_context'] = 50
        config['npoints'] = 32
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid_w_history(config,nhours, NOAA=True)
    elif model == 'P':
        config['checkpoint_test'] = checkpoint_path_P
        config['d_context'] = 50
        config['npoints'] = 32
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid(config,nhours, NOAA=True)

    # Plot obtained trajectories
    fig = plt.figure(figsize=(6,4.3))

    # creating the map - setting latitude and longitude
    m = Basemap(projection = 'mill', llcrnrlat = config['min_lat'], urcrnrlat = config['max_lat'], llcrnrlon = config['min_lon'], urcrnrlon = config['max_lon'], resolution = 'i') 
    m.drawcoastlines()
    m.drawcountries(color='gray')
    m.drawstates(color='gray')
    m.drawcoastlines()
    m.fillcontinents()

    m.plot(true_lon,true_lat,latlon=True,linewidth=2,color='k', label='True trajectory')

    m.plot(longitudes,latitudes,latlon=True,linewidth=1.5,color='tab:blue', label='Model A')
    m.plot(longitudes_stokes,longitudes_stokes,latlon=True,linewidth=1.5,color='tab:orange', label='Model C')
    m.plot(longitudes_dd,latitudes_dd,latlon=True,linewidth=1.5,color='tab:pink', label=f'Model {model}')

    m.drawmapscale(config['min_lon']+0.4, config['min_lat']+1, config['min_lon'], config['min_lat'], 50, barstyle='fancy')
    plt.tight_layout()
    plt.savefig('figures/predicted_trajectories.png')

