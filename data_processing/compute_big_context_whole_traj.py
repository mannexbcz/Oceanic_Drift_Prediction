from utils.read_data import water_interpolated, wind_interpolated, wave_interpolated, bathymetry_interpolated
import numpy as np
import pickle
from tqdm import tqdm
import os
import yaml
from utils.read_data import get_initial_position

config_path = '/data/manon/MasterThesis/configs_NOAA/all_configs'
saving_path = '/data/manon/MasterThesis/NOAA/bigcontexts'
point_per_deg = 10


def convert_name_to_config(name):
    config_name = 'config_' + name[:-4] + '.yml'
    return config_name

def get_water_context(path_water, lat_min, lon_min, time_init, lat_max, lon_max, points_per_degree=100):

    water_u_interpolation,water_v_interpolation = water_interpolated(path_water)

    npoint_lats = (lat_max-lat_min)*points_per_degree
    npoint_lons = (lon_max-lon_min)*points_per_degree

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoint_lats)
    lons = np.linspace(lon_min, lon_max, npoint_lons)
    
    # get tensor of interpolated values
    context_water_u = np.zeros([72,npoint_lons, npoint_lats])
    context_water_v = np.zeros([72,npoint_lons, npoint_lats])

    for i in range(npoint_lons):
        for j in range(npoint_lats):
            for k in range(72):
                context_water_u[k,i,j] = water_u_interpolation([time_init+k,0,lats[j],lons[i]])
                context_water_v[k,i,j] = water_v_interpolation([time_init+k,0,lats[j],lons[i]])

    return context_water_u, context_water_v

def get_wind_context(path_wind, lat_min, lon_min, time_init, lat_max, lon_max, points_per_degree=100):

    wind_u_interpolation,wind_v_interpolation = wind_interpolated(path_wind)
    
    npoint_lats = (lat_max-lat_min)*points_per_degree
    npoint_lons = (lon_max-lon_min)*points_per_degree

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoint_lats)
    lons = np.linspace(lon_min, lon_max, npoint_lons)
    
    # get tensor of interpolated values
    context_wind_u = np.zeros([72,npoint_lons, npoint_lats])
    context_wind_v = np.zeros([72,npoint_lons, npoint_lats])
    
    for i in range(npoint_lons):
        for j in range(npoint_lats):
            for k in range(72):
                context_wind_u[k,i,j] = wind_u_interpolation([time_init+k,lats[j],lons[i]])
                context_wind_v[k,i,j] = wind_v_interpolation([time_init+k,lats[j],lons[i]])

    return context_wind_u, context_wind_v


def get_waves_context(path_waves, lat_min, lon_min, time_init, lat_max, lon_max, points_per_degree=100):

    ust_interpolation,vst_interpolation = wave_interpolated(path_waves)
    
    npoint_lats = (lat_max-lat_min)*points_per_degree
    npoint_lons = (lon_max-lon_min)*points_per_degree

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoint_lats)
    lons = np.linspace(lon_min, lon_max, npoint_lons)
    
    # get tensor of interpolated values
    context_waves_u = np.zeros([72,npoint_lons, npoint_lats])
    context_waves_v = np.zeros([72,npoint_lons, npoint_lats])


    for i in range(npoint_lons):
        for j in range(npoint_lats):
            for k in range(72):
                context_waves_u[k,i,j] = ust_interpolation([time_init+k,lats[j],lons[i]])
                context_waves_v[k,i,j] = vst_interpolation([time_init+k,lats[j],lons[i]])

    return context_waves_u, context_waves_v

def get_context(path_water, path_wind, path_waves,lat_min, lon_min, time_init, lat_max, lon_max, name_file,saving_folder,points_per_degree=100):

    path_save = os.path.join(saving_folder,name_file)

    '''if os.path.exists(path_save):
        return path_save
    else:'''
    context_water_u, context_water_v = get_water_context(path_water, lat_min, lon_min, time_init, lat_max, lon_max, points_per_degree)
    context_waves_u, context_waves_v = get_waves_context(path_waves, lat_min, lon_min, time_init, lat_max, lon_max, points_per_degree)
    context_wind_u, context_wind_v = get_wind_context(path_wind, lat_min, lon_min, time_init, lat_max, lon_max, points_per_degree)    

    context = np.stack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v))
    #assert np.shape(context) == (6,npoints,npoints), f"Wrong shape for the context: {np.shape(context)}"

    with open(path_save, 'wb') as f:
        np.save(f,context)

    return path_save

if __name__ == "__main__": 
    with open("/data/manon/MasterThesis/NOAA/training_files_1000_new.pkl", "rb") as fp:   #Pickling
        train_files = pickle.load(fp)
    with open("/data/manon/MasterThesis/NOAA/testing_files_1000_new.pkl", "rb") as fp:   #Pickling
        test_files = pickle.load(fp)
    with open("/data/manon/MasterThesis/NOAA/validation_files_1000_new.pkl", "rb") as fp:   #Pickling
        val_files = pickle.load(fp)

    all_files = train_files + test_files + val_files

    for data_file in tqdm(all_files):
        config_name = convert_name_to_config(data_file)
        filename = 'context_' + data_file[:-4] + '.npy'
        try: 
            #print('Processing file', data_file)
            with open(os.path.join(config_path,config_name), 'r') as f:
                config = yaml.safe_load(f)
        except:
            continue

        try:
            _, init_time = get_initial_position(config['PATH_DRIFT'], NOAA = True)

            final_path = get_context(config['PATH_WATER'],config['PATH_WIND'],config['PATH_WAVES'],config['min_lat'], config['min_lon'],init_time, config['max_lat'], config['max_lon'],filename, saving_path, points_per_degree=point_per_deg)
        except:
            print('Error with file:', data_file)
            continue