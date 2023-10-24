from numpy.linalg import norm
import numpy as np
import math
from utils.read_data import get_closest_position,get_initial_position
from utils.ensemble_forecasting import get_distribution_stats, compute_N_trajectories
from haversine import haversine
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


def mahalanobis_distance(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, max_hours,step_hours = 1):

    sum_ = 0

    for hours in range(1,max_hours,step_hours):
        true_lon, true_lat = get_closest_position(path_drift, hours)
        true_pos = np.array([true_lon, true_lat])
        mean_x, mean_y, std_x, std_y = get_distribution_stats(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,hours)
        mean = np.array([mean_x, mean_y])
        std = np.array([std_x,std_y])
        sum_ = sum_ + norm(true_pos-mean)**2/norm(std)**2

    d_M = math.sqrt(sum_)

    return d_M

def mahalanobis_distance_points(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, max_hours):

    dMs = np.zeros(max_hours-1)

    for i in range(1,max_hours):
        true_lon, true_lat = get_closest_position(path_drift, i)
        true_pos = np.array([true_lon,true_lat])
        lon_endpoints, lat_endpoints = compute_N_trajectories(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, Ntraj=100, nhours = i, npoints_alpha = 3)
        data = np.stack((lon_endpoints,lat_endpoints),axis = 0)
        cov_mat = np.cov(data)
        mean = np.mean(data,axis=1)

        d_square = (true_pos-mean)@np.linalg.inv(cov_mat)@(true_pos-mean)
        dMs[i-1]=math.sqrt(d_square)

    dM_mean = np.mean(dMs)

    return dMs, dM_mean


def get_KDE_centroid(kde, lon_endpoints, lat_endpoints):

    lon_grid = np.linspace(np.min(lon_endpoints), np.max(lon_endpoints), 100)
    lat_grid = np.linspace(np.min(lat_endpoints), np.max(lat_endpoints), 100)
    xy_grid = np.meshgrid(lat_grid, lon_grid)
    grid_points = np.vstack([xy_grid[0].ravel(), xy_grid[1].ravel()]).T

    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    idx = np.argmax(density)

    centroid_lat = grid_points[idx,0]
    centroid_lon = grid_points[idx,1]

    return centroid_lon,centroid_lat

def get_distribution_stats(kde, lon_endpoints, lat_endpoints):

    lon_grid = np.linspace(np.min(lon_endpoints), np.max(lon_endpoints), 100)
    lat_grid = np.linspace(np.min(lat_endpoints), np.max(lat_endpoints), 100)
    xy_grid = np.meshgrid(lat_grid, lon_grid)
    grid_points = np.vstack([xy_grid[0].ravel(), xy_grid[1].ravel()]).T

    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    mean_x = np.sum(grid_points[:, 0] * density) / np.sum(density)
    mean_y = np.sum(grid_points[:, 1] * density) / np.sum(density)
    mean_point = (mean_x,mean_y)

    std_x = np.sqrt(np.sum((grid_points[:, 0] - mean_x)**2 * density) / np.sum(density))
    std_y = np.sqrt(np.sum((grid_points[:, 1] - mean_y)**2 * density) / np.sum(density))

    sum_std = 0
    for i in range(len(density)):
        sum_std += haversine((grid_points[i,0],grid_points[i,1]),mean_point)**2*density[i]
    std = np.sqrt(sum_std/np.sum(density))
    return mean_x, mean_y, std

def get_stats_from_N_points(longitudes,latitudes):
    mean_longitude = np.mean(longitudes)
    mean_latitude = np.mean(latitudes)

    mean_point = (mean_latitude,mean_longitude)
    sum_std = 0
    for i in range(len(longitudes)):
        point = (latitudes[i], longitudes[i])
        sum_std += haversine(point,mean_point)**2
    std = np.sqrt(sum_std/(len(longitudes)-1))
    print(std)

    return mean_longitude, mean_latitude, std


def averaged_NLL_and_centroids(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, ust_interpolation,vst_interpolation,Ntraj, nhours,case = 'Original'):
    longitudes, latitudes = compute_N_trajectories(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,ust_interpolation,vst_interpolation, Ntraj, nhours, npoints_alpha = 2,case=case, save_all = True)
    centroids_lons = np.zeros(nhours+1)
    centroids_lats = np.zeros(nhours+1)
    means_lons = np.zeros(nhours+1)
    means_lats = np.zeros(nhours+1)
    stds = np.zeros(nhours+1)
    dist_mean_true = np.zeros(nhours+1)
    dist_centroid_true = np.zeros(nhours+1)

    [centroids_lons[0],centroids_lats[0]],_ = get_initial_position(path_drift)
    means_lons[0], means_lats[0] = centroids_lons[0],centroids_lats[0]

    L_vec = np.zeros(nhours+1)
    #L = 0
    for i in range(1,nhours+1):
        data = np.zeros((Ntraj,2))
        data[:,1] = longitudes[:,i]
        data[:,0] = latitudes[:,i]
        kde = KernelDensity(bandwidth=1, kernel='gaussian',metric='haversine').fit(data)
        centroids_lons[i],centroids_lats[i] = get_KDE_centroid(kde,longitudes[:,i],latitudes[:,i])
        means_lats[i], means_lons[i], stds[i] = get_distribution_stats(kde, longitudes[:,i],latitudes[:,i])
        #means_lons[i], means_lats[i],stds[i] = get_stats_from_N_points(longitudes[:,i],latitudes[:,i])
        true_pos = np.zeros((1,2))
        true_pos[0,1], true_pos[0,0] = get_closest_position(path_drift, i) #lon, lat
        true_position = (true_pos[0,0],true_pos[0,1]) #lat,lon
        mean_position = (means_lats[i], means_lons[i])
        centroid_position = (centroids_lats[i],centroids_lons[i])
        dist_mean_true[i] = haversine(true_position,mean_position) # in km
        dist_centroid_true[i] = haversine(true_position,centroid_position)
        #L = L - kde.score_samples(true_pos)
        L_vec[i] = - kde.score_samples(true_pos)
    return L_vec, centroids_lons, centroids_lats, means_lons, means_lats, stds, dist_mean_true, dist_centroid_true


def averaged_NLL(path_drift,water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, Ntraj, nhours):
    longitudes, latitudes = compute_N_trajectories(path_drift, water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, Ntraj, nhours, npoints_alpha = 2, save_all = True)
    L = 0
    for i in range(1,nhours+1):
        data = np.zeros((Ntraj,2))
        data[:,1] = longitudes[:,i]
        data[:,0] = latitudes[:,i]
        kde = KernelDensity(bandwidth=1, kernel='gaussian',metric='haversine').fit(data)
        true_pos = np.zeros((1,2))
        true_pos[0,1], true_pos[0,0] = get_closest_position(path_drift, i)
        L = L - kde.score_samples(true_pos)
    L = L/nhours
    return L[0]


