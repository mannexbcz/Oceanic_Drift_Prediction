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
import faulthandler

CONFIG_PATH = './configs_2'
'''csvfile = '../data/nextpoint_ds/next_point_dataset.csv'
saving_folder_context = '../data/nextpoint_ds/contexts/pt32d50'
saving_folder_csvs = '../data/nextpoint_ds/contexts/pt32d50'''
d_context = 50
npoints = 32
seed = 42

def splitlist(listdir,ratio_train,ratio_val):
    N = len(listdir)
    split1 = int(N*ratio_train)
    split2 = int(N*(ratio_train+ratio_val))
    return [listdir[:split1], listdir[split1:split2], listdir[split2:]]

def get_context(path_water, path_wind, path_waves,init_lat, init_lon, init_time, name_file,saving_folder, d_context=1, npoints=32):

    path_save = os.path.join(saving_folder,name_file)

    if os.path.exists(path_save):
        return path_save
    else:
        context_water_u, context_water_v = get_water_context(path_water, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
        context_waves_u, context_waves_v = get_waves_context(path_waves, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
        context_wind_u, context_wind_v = get_wind_context(path_wind, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    

    #context_bathymetry, context_coasts = get_bathymetry_context(path_bathy, init_lat, init_lon, init_time, d = d_context, npoints = npoints)

    # merge contextes
    #print('Merging context')
    context = np.stack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v))
    assert np.shape(context) == (6,npoints,npoints), f"Wrong shape for the context: {np.shape(context)}"

    with open(path_save, 'wb') as f:
        np.save(f,context)

    return path_save

def compute_contexts_Canada():

    random.seed(seed)

    physical_model = get_physical_model()

    dataset_train = pd.DataFrame(columns=['Latitude_init','Longitude_init', 'time_init', 'Latitude_final', 'Longitude_final', 'Lat_phys', 'Lon_phys',
                               'PATH_CONTEXT'])
    dataset_test = pd.DataFrame(columns=['Latitude_init','Longitude_init', 'time_init', 'Latitude_final', 'Longitude_final', 'Lat_phys', 'Lon_phys',
                               'PATH_CONTEXT'])
    dataset_val = pd.DataFrame(columns=['Latitude_init','Longitude_init', 'time_init', 'Latitude_final', 'Longitude_final', 'Lat_phys', 'Lon_phys',
                               'PATH_CONTEXT'])
    
    listdir = os.listdir(CONFIG_PATH)
    random.shuffle(listdir)
    train_list, val_list, test_list = splitlist(listdir, 0.7,0.1)

    ########################### Train dataset ########################################################################
    for data_file in tqdm(train_list,desc='Trajectories (Training Dataset)'):

        with open(os.path.join(CONFIG_PATH,data_file), 'r') as f:
            config = yaml.safe_load(f)

        drift_tab = read_drift_positions(config['PATH_DRIFT'])

        list_df_ok = get_time_continuous_path(drift_tab)

        dict_path = {
            'PATH_WATER': config['PATH_WATER'], 
            'PATH_WIND' : config['PATH_WIND'], 
            'PATH_WAVES' : config['PATH_WAVES'],
            'PATH_BATHY' : config['PATH_BATHY']
        }

        n = 0

        for df in list_df_ok:

            lats = df['Latitude'].tolist()
            lons = df['Longitude'].tolist()
            times = df['hours'].tolist()

            for i in range(len(df)-1):
                n = n+1

                lat_init, lon_init, time_init = df['Latitude'].iloc[i], df['Longitude'].iloc[i], df['hours'].iloc[i]
                lon_final = np.interp(time_init+1, times, lons)
                lat_final = np.interp(time_init+1, times, lats)

                init_position = torch.tensor([lat_init, lon_init], dtype = torch.float)
                final_position = torch.tensor([lat_final, lon_final], dtype = torch.float)

                name_file = 'train/'+data_file[:-4] + '_' + str(n) + '.npy'

                context_path = get_context(config['PATH_WATER'], config['PATH_WIND'],config['PATH_WAVES'],config['PATH_BATHY'],lat_init,lon_init,time_init,name_file,d_context, npoints)

                xphys = physical_model(init_position, time_init, dict_path)
                xphys_lat = xphys[0].item()
                xphys_lon = xphys[1].item()

                new_row = pd.DataFrame({'Latitude_init':lat_init, 'Longitude_init':lon_init,'time_init': time_init, 'Latitude_final':lat_final, 'Longitude_final':lon_final,
                                        'Lat_phys':xphys_lat, 'Lon_phys':xphys_lon,'PATH_CONTEXT':context_path}, index=[0])
                
                dataset_train = pd.concat([new_row,dataset_train.loc[:]]).reset_index(drop=True)

    ########################### Test dataset ########################################################################
    for data_file in tqdm(test_list,desc='Trajectories (Test Dataset)'):
        with open(os.path.join(CONFIG_PATH,data_file), 'r') as f:
            config = yaml.safe_load(f)

        drift_tab = read_drift_positions(config['PATH_DRIFT'])

        list_df_ok = get_time_continuous_path(drift_tab)

        dict_path = {
            'PATH_WATER': config['PATH_WATER'], 
            'PATH_WIND' : config['PATH_WIND'], 
            'PATH_WAVES' : config['PATH_WAVES'],
            'PATH_BATHY' : config['PATH_BATHY']
        }

        n = 0

        for df in list_df_ok:

            lats = df['Latitude'].tolist()
            lons = df['Longitude'].tolist()
            times = df['hours'].tolist()

            for i in range(len(df)-1):
                n = n+1

                lat_init, lon_init, time_init = df['Latitude'].iloc[i], df['Longitude'].iloc[i], df['hours'].iloc[i]
                lon_final = np.interp(time_init+1, times, lons)
                lat_final = np.interp(time_init+1, times, lats)

                init_position = torch.tensor([lat_init, lon_init], dtype = torch.float)
                final_position = torch.tensor([lat_final, lon_final], dtype = torch.float)

                name_file = 'test/'+data_file[:-4] + '_' + str(n) + '.npy'

                context_path = get_context(config['PATH_WATER'], config['PATH_WIND'],config['PATH_WAVES'],config['PATH_BATHY'],lat_init,lon_init,time_init,name_file,d_context, npoints)

                xphys = physical_model(init_position, time_init, dict_path)
                xphys_lat = xphys[0].item()
                xphys_lon = xphys[1].item()

                new_row = pd.DataFrame({'Latitude_init':lat_init, 'Longitude_init':lon_init,'time_init': time_init, 'Latitude_final':lat_final, 'Longitude_final':lon_final,
                                        'Lat_phys':xphys_lat, 'Lon_phys':xphys_lon,'PATH_CONTEXT':context_path}, index=[0])
                
                dataset_test = pd.concat([new_row,dataset_test.loc[:]]).reset_index(drop=True)

    ########################### Val dataset ########################################################################
    for data_file in tqdm(val_list,desc='Trajectories (Validation Dataset)'):

        with open(os.path.join(CONFIG_PATH,data_file), 'r') as f:
            config = yaml.safe_load(f)

        drift_tab = read_drift_positions(config['PATH_DRIFT'])

        list_df_ok = get_time_continuous_path(drift_tab)

        dict_path = {
            'PATH_WATER': config['PATH_WATER'], 
            'PATH_WIND' : config['PATH_WIND'], 
            'PATH_WAVES' : config['PATH_WAVES'],
            'PATH_BATHY' : config['PATH_BATHY']
        }

        n = 0

        for df in list_df_ok:

            lats = df['Latitude'].tolist()
            lons = df['Longitude'].tolist()
            times = df['hours'].tolist()

            for i in range(len(df)-1):
                n = n+1
                lat_init, lon_init, time_init = df['Latitude'].iloc[i], df['Longitude'].iloc[i], df['hours'].iloc[i]
                lon_final = np.interp(time_init+1, times, lons)
                lat_final = np.interp(time_init+1, times, lats)

                init_position = torch.tensor([lat_init, lon_init], dtype = torch.float)
                final_position = torch.tensor([lat_final, lon_final], dtype = torch.float)

                name_file = 'val/'+data_file[:-4] + '_' + str(n) + '.npy'

                context_path = get_context(config['PATH_WATER'], config['PATH_WIND'],config['PATH_WAVES'],config['PATH_BATHY'],lat_init,lon_init,time_init,name_file,d_context, npoints)

                xphys = physical_model(init_position, time_init, dict_path)
                xphys_lat = xphys[0].item()
                xphys_lon = xphys[1].item()

                new_row = pd.DataFrame({'Latitude_init':lat_init, 'Longitude_init':lon_init,'time_init': time_init, 'Latitude_final':lat_final, 'Longitude_final':lon_final,
                                        'Lat_phys':xphys_lat, 'Lon_phys':xphys_lon,'PATH_CONTEXT':context_path}, index=[0])
                
                dataset_val = pd.concat([new_row,dataset_val.loc[:]]).reset_index(drop=True)

    
    print('Length of train dataset:', len(dataset_train))
    dataset_train.to_csv(os.path.join(saving_folder_csvs, 'next_point_dataset_train.csv'))

    print('Length of test dataset:', len(dataset_test))
    dataset_test.to_csv(os.path.join(saving_folder_csvs, 'next_point_dataset_test.csv'))

    print('Length of val dataset:', len(dataset_val))
    dataset_val.to_csv(os.path.join(saving_folder_csvs, 'next_point_dataset_val.csv'))

    with open(os.path.join(saving_folder_csvs, 'list_config_test.txt'), 'wb') as f:
        pickle.dump(test_list, f)

    print('Done!')

    return


def compute_contexts_from_config_list_NOAA(path_configs, saving_path, saving_folder_context, set='train'):
    
    physical_model = get_physical_model()

    dataset_train = pd.DataFrame(columns=['Latitude_init','Longitude_init', 'time_init', 'Latitude_final', 'Longitude_final', 'Lat_phys', 'Lon_phys',
                               'PATH_CONTEXT'])

    listdir = os.listdir(path_configs)

    for data_file in tqdm(listdir,desc='Trajectories'):
        with open(os.path.join(path_configs,data_file), 'r') as f:
            config = yaml.safe_load(f)

        drift_tab = read_drift_positions(config['PATH_DRIFT'], NOAA=True)

        dict_path = {
            'PATH_WATER': config['PATH_WATER'], 
            'PATH_WIND' : config['PATH_WIND'], 
            'PATH_WAVES' : config['PATH_WAVES']
        }

        n = 0

        lats = drift_tab['Latitude'].tolist()
        lons = drift_tab['Longitude'].tolist()
        times = drift_tab['hours'].tolist()

        for i in range(len(drift_tab)-1):
            n = n+1

            lat_init, lon_init, time_init = drift_tab['Latitude'].iloc[i], drift_tab['Longitude'].iloc[i], drift_tab['hours'].iloc[i]
            lon_final = np.interp(time_init+1, times, lons)
            lat_final = np.interp(time_init+1, times, lats)

            init_position = torch.tensor([lat_init, lon_init], dtype = torch.float)

            name_file = set +'/'+data_file[:-4] + '_' + str(n) + '.npy'

            try:
                context_path = get_context(config['PATH_WATER'], config['PATH_WIND'],config['PATH_WAVES'],lat_init,lon_init,time_init,name_file,saving_folder_context,d_context, npoints)
            except:
                print('notworking')
                continue

            xphys = physical_model(init_position, time_init, dict_path)
            xphys_lat = xphys[0].item()
            xphys_lon = xphys[1].item()

            new_row = pd.DataFrame({'Latitude_init':lat_init, 'Longitude_init':lon_init,'time_init': time_init, 'Latitude_final':lat_final, 'Longitude_final':lon_final,
                                    'Lat_phys':xphys_lat, 'Lon_phys':xphys_lon,'PATH_CONTEXT':context_path}, index=[0])
            
            dataset_train = pd.concat([new_row,dataset_train.loc[:]]).reset_index(drop=True)
        
    print('Length of train dataset:', len(dataset_train))
    dataset_train.to_csv(saving_path)



if __name__ == "__main__": 

    faulthandler.enable()
    
    path_configs = '/data/manon/MasterThesis/configs_NOAA/val'
    saving_path = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_val.csv'
    saving_folder_context = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50'
    set = 'val'

    print("Computing contexts for validation dataset")
    compute_contexts_from_config_list_NOAA(path_configs, saving_path, saving_folder_context, set)

    path_configs = '/data/manon/MasterThesis/configs_NOAA/train'
    saving_path = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_train.csv'
    saving_folder_context = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50'
    set = 'train'

    print("Computing contexts for training dataset")
    compute_contexts_from_config_list_NOAA(path_configs, saving_path, saving_folder_context, set)

    path_configs = '/data/manon/MasterThesis/configs_NOAA/test'
    saving_path = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_test.csv'
    saving_folder_context = '/data/manon/MasterThesis/NOAA/nextpoint_ds/contexts/pt32d50'
    set = 'test'

    print("Computing contexts for test dataset")
    compute_contexts_from_config_list_NOAA(path_configs, saving_path, saving_folder_context, set)


    