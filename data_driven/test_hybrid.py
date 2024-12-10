import yaml
import netCDF4 as nc
import numpy as np
import torch
import pandas as pd
import pickle
import os
from tqdm import tqdm

from utils.read_data import *
from utils.param_alpha import get_alpha, get_complex_alpha, compute_alpha_whole_trajectory,compute_alpha_matrix, get_alpha_averaged, update_alpha_GD
from models.linear_model import u_drift_linear, u_drift_linear_complex, u_drift_linear_matrix
from utils.RK4 import RK4_step, compute_position
from data_driven.compute_trajectory_hybrid import compute_trajectory_hybrid
from metrics.metrics_trajectory import *
from data_driven.models.hybrid_model import HybridDriftModule

list_test_files = '/data/manon/MasterThesis/NOAA/testing_files_1000_new.pkl'
config_path = '/data/manon/MasterThesis/configs_NOAA/all_configs'
checkpoint_path = '~/checkpoints/MasterThesis/lightning_logs/version_2/checkpoints/epoch=6-step=8246.ckpt'
config_data_driven = './configs/config_training.yml'
test_csv = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_test_ok.csv'

def bootstrap(x,alpha=0.05):
    '''This function returns the 1-alpha% Confidence interval of the mean of x 
      using a bootstrapping approach with 10'000 samples'''
    np.random.seed(0)
    mean_vals = [np.random.choice(x,len(x)).mean() for _ in range(10000)]
    
    return np.quantile(mean_vals, alpha/2), np.quantile(mean_vals, 1-alpha/2)

def convert_name_to_config(name):
    config_name = 'config_' + name[:-4] + '.yml'
    return config_name

def print_results(res):
    for colname in ['ssc','dsi','tad','ssc_dd','dsi_dd','tad_dd']:
        col = res[colname]
        m = np.mean(col)
        a,b = bootstrap(col)
        ecart = (b-a)/2
        print('     ', colname,': ', m, ' [',a,',',b,'], ', ecart)
    return

def print_results_nextpoint(res):
    for colname in ['d_phys', 'd_pred']:
        col = res[colname]
        m = np.mean(col)
        a,b = bootstrap(col)
        ecart = (b-a)/2
        print('     ', colname,': ', m, ' [',a,',',b,'], ', ecart)
    return

def compute_nextpos_and_scores_hybrid(row, model):
    
    final_lat, final_lon = row['Latitude_final'], row['Longitude_final']
    final_position = torch.tensor([final_lat, final_lon], dtype = torch.float)

    xphys_lat, xphys_lon = row['Lat_phys'], row['Lon_phys']
    xphys = torch.unsqueeze(torch.tensor([xphys_lat, xphys_lon], dtype = torch.float).cuda(), axis=0)

    # get context
    context_path = row['PATH_CONTEXT']
    with open(context_path, 'rb') as f:
        context = np.load(f)
    context = torch.from_numpy(context.astype(np.float32)).cuda()
    #context = context[0:-2,:,:]

    pos_pred = model(xphys,context)

    lon_pred = pos_pred[0,1].item()
    lat_pred = pos_pred[0,0].item()

    true_position = (final_lat, final_lon)
    pred_position = (lat_pred, lon_pred)
    phys_position = (xphys_lat, xphys_lon)

    dist_phys = haversine(true_position,phys_position)
    dist_pred = haversine(true_position,pred_position)

    row['Longitude_pred'], row['Latitude_pred'], row['d_phys'], row['d_pred'] =  lon_pred, lat_pred, dist_phys, dist_pred

    return row

if __name__ == "__main__": 

    ##################### Next point dataset ##################################

    test_df = pd.read_csv(test_csv)

    module = HybridDriftModule.load_from_checkpoint(checkpoint_path)
    model = module.model
    model.eval()

    test_df = test_df.apply(lambda row:compute_nextpos_and_scores_hybrid(row, model), axis=1)
    
    print('----------next point----------------')
    print_results_nextpoint(test_df)

    ##################### Long term performances ##############################
    with open(config_data_driven, 'r') as f:
        config_dd = yaml.safe_load(f)

    with open(list_test_files, 'rb') as f:
        list_names = pickle.load(f)
    
    list_names = [convert_name_to_config(name) for name in list_names]

    #list_names = os.listdir(config_path)
    #list_names = list_names[:10]

    results_list_3h = []
    results_list_6h = []
    results_list_12h = []
    results_list_24h = []
    results_list_48h = []
    results_list_72h = []

    for data_file in tqdm(list_names):
        
        try: 
            #print('Processing file', data_file)
            with open(os.path.join(config_path,data_file), 'r') as f:
                config = yaml.safe_load(f)
        except:
            continue

        config = {**config, **config_dd}
        config['checkpoint_test'] = checkpoint_path

        pos_1, time1 = get_initial_position(config['PATH_DRIFT'],NOAA=True)
        true_lon, true_lat, true_time = get_true_drift_positions(config['PATH_DRIFT'],NOAA=True)

        # get nhours
        #nhours = get_number_hours(true_time)
        #nhours = min(nhours,72)
        nhours = 72

        # get extrapolated true positions
        #true_lon_extrapolated, true_lat_extrapolated = get_extrapolated_true_position(config['PATH_DRIFT'], nhours)
        true_lon_extrapolated, true_lat_extrapolated = true_lon, true_lat

        # physical model
        try: 
            u10_interpolation, v10_interpolation = wind_interpolated(config['PATH_WIND'])
            water_u_interpolation, water_v_interpolation = water_interpolated(config['PATH_WATER'])
        except:
            continue

        alpha = update_alpha_GD(config['PATH_DRIFT'], water_u_interpolation, water_v_interpolation, u10_interpolation, v10_interpolation,alpha = 0.02, theta = 0.349066,  step=0.1, npoints=3,NOAA=True)
        u_drift = u_drift_linear_matrix(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation)

        longitudes, latitudes, time_final = compute_position(u_drift, pos_1, time1,1,nhours)

        # data driven (hybrid) model
        longitudes_dd, latitudes_dd = compute_trajectory_hybrid(config,nhours,NOAA=True)

        for i in [3,6,12,24,48,72]: 
            if nhours >= i:
                ssc = skill_score(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1], longitudes[:i+1], latitudes[:i+1])
                dsi = separation_after_N_hours(true_lon_extrapolated, true_lat_extrapolated, longitudes,latitudes, i)
                tad = time_averaged_distance(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1], longitudes[:i+1], latitudes[:i+1])

                ssc_dd = skill_score(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1], longitudes_dd[:i+1], latitudes_dd[:i+1])
                dsi_dd = separation_after_N_hours(true_lon_extrapolated, true_lat_extrapolated, longitudes_dd,latitudes_dd, i)
                tad_dd = time_averaged_distance(true_lon_extrapolated[:i+1], true_lat_extrapolated[:i+1], longitudes_dd[:i+1], latitudes_dd[:i+1])

                new_row = {'File': data_file,'nhours':i, 'ssc':ssc,'dsi':dsi, 'tad':tad, 'ssc_dd': ssc_dd, 'dsi_dd':dsi_dd, 'tad_dd':tad_dd}

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

    # Analyse of the df
    results_list_3h = pd.DataFrame.from_dict(results_list_3h)
    results_list_6h = pd.DataFrame.from_dict(results_list_6h)
    results_list_12h = pd.DataFrame.from_dict(results_list_12h)
    results_list_24h = pd.DataFrame.from_dict(results_list_24h)
    results_list_48h = pd.DataFrame.from_dict(results_list_48h)
    results_list_72h = pd.DataFrame.from_dict(results_list_72h)

    print('-------------3h--------------')
    print_results(results_list_3h)
    print('-------------6h--------------')
    print_results(results_list_6h)
    print('-------------12h--------------')
    print_results(results_list_12h)
    print('-------------24h--------------')
    print_results(results_list_24h)
    print('-------------48h--------------')
    print_results(results_list_48h)
    print('-------------72h--------------')
    print_results(results_list_72h)



            




        


