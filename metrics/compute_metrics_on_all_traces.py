import os
import pandas as pd
import yaml
from utils.read_data import *
from utils.param_alpha import update_alpha_GD, update_alpha_GD_stockes
from models.linear_model import u_drift_linear_matrix
from models.stockes_model import u_drift_ocean_wind_stockes_matrix,u_drift_ocean_stockes_matrix
from utils.RK4 import compute_position
from metrics.metrics_trajectory import *
from metrics.metrics_probabilities import *
from tqdm import tqdm

PATH_CONFIGS = './configs_2'
stokes = False
wave_only = True

if __name__ == "__main__": 
    print('Stokes:', stokes)
    print('Waves only:', wave_only)

    #results = pd.DataFrame(columns = ['File', 'nhours', 'ssc', 'tad', 'nll', 'ssc_centroids','tad_centroids'])
    results_list_3h = []
    results_list_6h = []
    results_list_12h = []
    results_list_24h = []
    results_list_48h = []
    results_list_72h = []

    if stokes:
        case_ = 'Stokes'
    elif wave_only:
        case_ = 'WaveOnly'
    else:
        case_ = 'Original'
    
    for data_file in tqdm(os.listdir(PATH_CONFIGS),desc='Trajectories'):
    
        with open(os.path.join(PATH_CONFIGS,data_file), 'r') as f:
            config = yaml.safe_load(f)

        # Get interpolated initial data
        u10_interpolation, v10_interpolation = wind_interpolated(config['PATH_WIND'])
        water_u_interpolation, water_v_interpolation = water_interpolated(config['PATH_WATER'])
        ust_interpolation, vst_interpolation = wave_interpolated(config['PATH_WAVES'])

        # Get initial position & True positions
        pos_1, time1 = get_initial_position(config['PATH_DRIFT']) 
        true_lon, true_lat, true_time = get_true_drift_positions(config['PATH_DRIFT'])

        # Get (integer) number of hours
        nhours = get_number_hours(true_time)
        nhours = min(nhours,72)

        # Get interpolated positions
        true_lon_extrapolated, true_lat_extrapolated = get_extrapolated_true_position(config['PATH_DRIFT'], nhours)

        # Compute one trajectory (without perturbation)
        if stokes:
            alpha_stokes = update_alpha_GD_stockes(config['PATH_DRIFT'], water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation, ust_interpolation, vst_interpolation, alpha = 0.03, theta = 0.349066,  step=0.2, npoints=3)
            u_drift = u_drift_ocean_wind_stockes_matrix(alpha_stokes, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation, ust_interpolation, vst_interpolation)

        elif wave_only:
            u_drift = u_drift_ocean_stockes_matrix(water_u_interpolation,water_v_interpolation,ust_interpolation, vst_interpolation)
        else:
            alpha = update_alpha_GD(config['PATH_DRIFT'], water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,alpha = 0.03, theta = 0.349066,  step=0.1, npoints=3)
            u_drift = u_drift_linear_matrix(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation)
        

        # Compute KDE & centroids & ALL
        #print('Computing KDEs')
        L_vec, centroids_lons, centroids_lats, means_lons, means_lats, stds, dist_mean_true,dist_centroid_true = averaged_NLL_and_centroids(config['PATH_DRIFT'],water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,ust_interpolation,vst_interpolation, Ntraj=200, nhours=nhours,case=case_)
        
        longitudes, latitudes, time_final = compute_position(u_drift, pos_1, time1,1,nhours)

        anll = sum(L_vec)/nhours

        # =================== 3h =================================================
        for i in [3,6,12,24,48,72]: #,
            if nhours >= i:
                ssc = skill_score(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1], longitudes[:i+1], latitudes[:i+1])
                tad = time_averaged_distance(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1], longitudes[:i+1], latitudes[:i+1])
                dsi = separation_after_N_hours(true_lon_extrapolated, true_lat_extrapolated, longitudes,latitudes, i)
                nll = L_vec[i]
                dist_mu_true = dist_mean_true[i]
                dist_centr_true = dist_centroid_true[i]
                std = stds[i]
                ssc_centroids = skill_score(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1],centroids_lons[:i+1], centroids_lats[:i+1])
                tad_centroids = time_averaged_distance(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1],centroids_lons[:i+1], centroids_lats[:i+1])
                ssc_means = skill_score(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1],means_lons[:i+1], means_lats[:i+1])
                tad_means = time_averaged_distance(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1],means_lons[:i+1], means_lats[:i+1])
            
                new_row = {'File': data_file,'nhours':i, 'ssc':ssc,'tad':tad,'dsi':dsi, 'nll':nll,'anll':anll, 'dist_mu_true':dist_mu_true,
                           'dist_centr_true':dist_centr_true, 'std':std, 'ssc_centroids':ssc_centroids,'tad_centroids': tad_centroids,
                           'ssc_means':ssc_means,'tad_means':tad_means}
            
                
                if i == 3:
                    results_list_3h.append(new_row)
                elif i == 6:
                    results_list_6h.append(new_row)
                elif i == 12:
                    results_list_12h.append(new_row)
                elif i == 24:
                    results_list_24h.append(new_row)
                elif i == 48:
                    results_list_48h.append(new_row)
                elif i == 72:
                    results_list_72h.append(new_row)
        

    results_list_3h = pd.DataFrame.from_dict(results_list_3h)
    results_list_6h = pd.DataFrame.from_dict(results_list_6h)
    results_list_12h = pd.DataFrame.from_dict(results_list_12h)
    results_list_24h = pd.DataFrame.from_dict(results_list_24h)
    results_list_48h = pd.DataFrame.from_dict(results_list_48h)
    results_list_72h = pd.DataFrame.from_dict(results_list_72h)

    results_list_3h.to_csv('results_metrics_wind_only_3h.csv')
    results_list_6h.to_csv('results_metrics_wind_only_6h.csv')
    results_list_12h.to_csv('results_metrics_wind_only_12h.csv')
    results_list_24h.to_csv('results_metrics_wind_only_24h.csv')
    results_list_48h.to_csv('results_metrics_wind_only_48h.csv')
    results_list_72h.to_csv('results_metrics_wind_only_72h.csv')
