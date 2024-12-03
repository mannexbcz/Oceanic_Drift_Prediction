import os
import pandas as pd
from utils.read_data import read_drift_positions
from data_processing.get_24h_trajectories import get_time_continuous_path
import yaml
import numpy as np
from tqdm import tqdm
import random
import torch
from data_processing.context import *
from models.physical_model import get_physical_model
import pickle

d_context = 50
npoints = 32
seed = 42

def get_trajectory_dataset(path_configs, saving_path):
    
    dataset_train = pd.DataFrame(columns=['lat0','lon0', 'lat1','lon1', 'lat2','lon2', 'lat3','lon3', 'time_init','PATH_WIND', 'PATH_WATER', 'PATH_WAVES'])

    listdir = os.listdir(path_configs)

    for data_file in tqdm(listdir,desc='Trajectories'):
        with open(os.path.join(path_configs,data_file), 'r') as f:
            config = yaml.safe_load(f)

        drift_tab = read_drift_positions(config['PATH_DRIFT'], NOAA=True)

        n = 0

        lats = drift_tab['Latitude'].tolist()
        lons = drift_tab['Longitude'].tolist()
        times = drift_tab['hours'].tolist()

        for i in range(len(drift_tab)-3):
            n = n+1

            lat0, lon0, time_init = drift_tab['Latitude'].iloc[i], drift_tab['Longitude'].iloc[i], drift_tab['hours'].iloc[i]
            lon1 = np.interp(time_init+1, times, lons)
            lat1 = np.interp(time_init+1, times, lats)
            lon2 = np.interp(time_init+2, times, lons)
            lat2 = np.interp(time_init+2, times, lats)
            lon3 = np.interp(time_init+3, times, lons)
            lat3 = np.interp(time_init+3, times, lats)

            new_row = pd.DataFrame({'lat0':lat0,'lon0':lon0, 'lat1':lat1,'lon1':lon1, 'lat2':lat2,'lon2':lon2, 'lat3':lat3,'lon3':lon3, 'time_init': time_init, 
                                    'PATH_WIND':config['PATH_WIND'], 'PATH_WATER':config['PATH_WATER'], 'PATH_WAVES':config['PATH_WAVES'], 'CONFIG_PATH':os.path.join(path_configs,data_file)}, index=[0])
            
            dataset_train = pd.concat([new_row,dataset_train.loc[:]]).reset_index(drop=True)
        
    print('Length of train dataset:', len(dataset_train))
    dataset_train.to_csv(saving_path)



if __name__ == "__main__": 
    path_configs = '/data/manon/MasterThesis/configs_NOAA/val'
    saving_path = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_val_trajectory.csv'
    print("Computing contexts for validation dataset")
    get_trajectory_dataset(path_configs,saving_path)

    path_configs = '/data/manon/MasterThesis/configs_NOAA/train'
    saving_path = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_train_trajectory.csv'
    print("Computing contexts for train dataset")
    get_trajectory_dataset(path_configs,saving_path)

    path_configs = '/data/manon/MasterThesis/configs_NOAA/test'
    saving_path = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_test_trajectory.csv'
    print("Computing contexts for test dataset")
    get_trajectory_dataset(path_configs,saving_path)


    