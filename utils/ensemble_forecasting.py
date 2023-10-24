from utils.read_data import get_initial_position, get_closest_position, water_interpolated, wind_interpolated, get_true_drift_positions
from utils.convert_lats import new_latitude, new_longitude
from utils.param_alpha import update_alpha_GD, update_alpha_GD_stockes
from models.linear_model import u_drift_linear_matrix_randomized
from models.stockes_model import u_drift_ocean_wind_stockes_matrix_randomized, u_drift_ocean_stockes_matrix_randomized
from utils.RK4 import compute_position
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
from sklearn.neighbors import KernelDensity

# Uncertainties for all parameters
DMETRE = 100 # uncertainty in m on the position
DALPHA = 0.02
DTHETA = 0.17
DUWATER = 0.1 #m/s
DUWIND = 0.15 #m/s 
DUWAVES = 0.15 #m/s


def compute_one_trajectory(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, nhours, npoints = 3):

    pos_1, time1 = get_initial_position(path_drift)

    # Random perturbation for the initial position
    delta_x = np.random.normal(loc = 0.0, scale = DMETRE)
    delta_y = np.random.normal(loc = 0.0, scale = DMETRE)
    pos_1 = np.array([new_longitude(pos_1[1], pos_1[0],delta_x), new_latitude(pos_1[1],pos_1[0],delta_y)])

    # Random perturbation for initial value of alpha
    alpha_init = 0.03 + np.random.normal(loc=0.0, scale=DALPHA)
    theta_init = 0.349066+ np.random.normal(loc=0.0, scale = DTHETA) #0.349066 

    # GRADIENT DESCENT ALPHA
    alpha = update_alpha_GD(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,alpha = alpha_init, theta = theta_init,  step=0.1, npoints=npoints)

    # u_drift function, adds random perturbations on the wind and current velocities
    u_drift = u_drift_linear_matrix_randomized(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation,DUWATER,DUWIND)

    longitudes, latitudes, time_final = compute_position(u_drift, pos_1, time1,1, nhours)

    return longitudes, latitudes, time_final

def compute_one_trajectory_stokes(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, ust_interpolation,vst_interpolation,nhours, npoints = 3):
    pos_1, time1 = get_initial_position(path_drift)

    # Random perturbation for the initial position
    delta_x = np.random.normal(loc = 0.0, scale = DMETRE)
    delta_y = np.random.normal(loc = 0.0, scale = DMETRE)
    pos_1 = np.array([new_longitude(pos_1[1], pos_1[0],delta_x), new_latitude(pos_1[1],pos_1[0],delta_y)])

    # Random perturbation for initial value of alpha
    alpha_init = 0.03 + np.random.normal(loc=0.0, scale=DALPHA)
    theta_init = 0.349066+ np.random.normal(loc=0.0, scale = DTHETA) #0.349066 

    # GRADIENT DESCENT ALPHA
    alpha = update_alpha_GD_stockes(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,ust_interpolation,vst_interpolation, alpha = alpha_init, theta = theta_init,  step=0.1, npoints=npoints)

    # u_drift function, adds random perturbations on the wind and current velocities
    u_drift = u_drift_ocean_wind_stockes_matrix_randomized(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation,ust_interpolation,vst_interpolation,DUWATER,DUWIND,DUWAVES)

    longitudes, latitudes, time_final = compute_position(u_drift, pos_1, time1,1, nhours)

    return longitudes, latitudes, time_final

def compute_one_trajectory_wave_only(path_drift, water_u_interpolation, water_v_interpolation, ust_interpolation,vst_interpolation,nhours):
    pos_1, time1 = get_initial_position(path_drift)
    # Random perturbation for the initial position
    delta_x = np.random.normal(loc = 0.0, scale = DMETRE)
    delta_y = np.random.normal(loc = 0.0, scale = DMETRE)
    pos_1 = np.array([new_longitude(pos_1[1], pos_1[0],delta_x), new_latitude(pos_1[1],pos_1[0],delta_y)])

    # u_drift function, adds random perturbations on the wind and current velocities
    u_drift = u_drift_ocean_stockes_matrix_randomized(water_u_interpolation, water_v_interpolation,ust_interpolation,vst_interpolation, DUWATER, DUWAVES)

    longitudes, latitudes, time_final = compute_position(u_drift, pos_1, time1,1, nhours)

    return longitudes, latitudes, time_final


def compute_N_trajectories(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, ust_interpolation,vst_interpolation, Ntraj, nhours, npoints_alpha = 3, case = 'Original', save_all = False):

    if save_all: 
        lon_endpoints = np.zeros((Ntraj,nhours+1))
        lat_endpoints = np.zeros((Ntraj,nhours+1))
    else:
        lon_endpoints = np.zeros(Ntraj)
        lat_endpoints = np.zeros(Ntraj)

    for i in range(Ntraj):
        if case == 'Original':
            longitudes, latitudes, _ = compute_one_trajectory(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, nhours, npoints_alpha)
        elif case == 'Stokes':
            longitudes, latitudes, _ = compute_one_trajectory_stokes(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, ust_interpolation,vst_interpolation, nhours, npoints_alpha)
        elif case == 'WaveOnly':
            longitudes, latitudes, _ = compute_one_trajectory_wave_only(path_drift, water_u_interpolation, water_v_interpolation, ust_interpolation,vst_interpolation,nhours)
        
        if save_all:
            lon_endpoints[i,:] = longitudes
            lat_endpoints[i,:] = latitudes
        else:
            lon_endpoints[i] = longitudes[-1]
            lat_endpoints[i] = latitudes[-1]

    return lon_endpoints, lat_endpoints

def plot_probability_after_N_hours(config, Ntraj, nhours, latmin = 44, latmax = 52,lonmin = -74,lonmax = -53):
    
    u10_interpolation, v10_interpolation = wind_interpolated(config['PATH_WIND'])
    water_u_interpolation, water_v_interpolation = water_interpolated(config['PATH_WATER'])

    lon_endpoints, lat_endpoints = compute_N_trajectories(config['PATH_DRIFT'],water_u_interpolation, water_v_interpolation,u10_interpolation, v10_interpolation,Ntraj,nhours)
    true_lon, true_lat = get_closest_position(config['PATH_DRIFT'],nhours)
    lon_traj, lat_traj, _ = get_true_drift_positions(config['PATH_DRIFT'])

    pos_1, _ = get_initial_position(config['PATH_DRIFT'])
    # setting the size of the map
    fig,ax = plt.subplots(figsize=(12,9))

    # creating the map - setting latitude and longitude
    m = Basemap(projection = 'mill', llcrnrlat = latmin, urcrnrlat = latmax, llcrnrlon =lonmin, urcrnrlon = lonmax, resolution = 'i') 

    x, y  = m(lon_endpoints, lat_endpoints)

    df_xy = pd.DataFrame({'Longitude': x, 'Latitude':y})
    # drawing the coastline
    m.drawcoastlines()
    m.drawcountries(color='yellowgreen')
    m.drawstates(color='yellowgreen')
    m.drawcoastlines()
    m.fillcontinents(color='darkseagreen')

    sns.kdeplot(data = df_xy, x='Longitude',y='Latitude', fill = True)

    # plotting the map
    m.scatter(pos_1[0], pos_1[1], latlon=True, s=20, color = 'k', marker='*', label='Initial position')
    m.scatter(true_lon,true_lat,latlon=True,s=20,color='k', marker = 'o', label='True position')
    m.plot(lon_traj,lat_traj,latlon=True,linewidth=1.5,color='k', linestyle='dashdot',label='True trajectory')


    plt.legend(loc = 'lower right',framealpha=1)
    plt.title(f'Probability distribution after %3d hours' %nhours)
    plt.show()

    return


def get_distribution_stats(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,hours):
    lon_endpoints, lat_endpoints = compute_N_trajectories(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, Ntraj=100, nhours = hours, npoints_alpha = 3)
    data = np.concatenate((np.expand_dims(lon_endpoints, 1), np.expand_dims(lat_endpoints,1)), axis=1)
    kde = KernelDensity(bandwidth=1, kernel='gaussian').fit(data)

    x_grid = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    y_grid = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
    xy_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([xy_grid[0].ravel(), xy_grid[1].ravel()]).T

    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    mean_x = np.sum(grid_points[:, 0] * density) / np.sum(density)
    mean_y = np.sum(grid_points[:, 1] * density) / np.sum(density)

    std_x = np.sqrt(np.sum((grid_points[:, 0] - mean_x)**2 * density) / np.sum(density))
    std_y = np.sqrt(np.sum((grid_points[:, 1] - mean_y)**2 * density) / np.sum(density))

    return mean_x, mean_y, std_x, std_y

