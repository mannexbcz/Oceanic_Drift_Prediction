import os 
import pandas as pd
from utils.read_data import get_true_drift_positions, convert_date_back
from datetime import datetime, timedelta
import math
import time
import yaml
from tqdm import tqdm
import pickle
import requests
import cdsapi

PATH_CONFIGS = '/data/manon/MasterThesis/configs_NOAA'
PATH_CONFIGS_TRAIN = '/data/manon/MasterThesis/configs_NOAA/train'
PATH_CONFIGS_TEST = '/data/manon/MasterThesis/configs_NOAA/test'
PATH_CONFIGS_VAL = '/data/manon/MasterThesis/configs_NOAA/val'
LIST_TRAIN = '/data/manon/MasterThesis/NOAA/training_files_1000.pkl'
LIST_TEST = '/data/manon/MasterThesis/NOAA/testing_files_1000.pkl'
LIST_VAL = '/data/manon/MasterThesis/NOAA/validation_files_1000.pkl'
PATH_TRAJ = '/data/manon/MasterThesis/NOAA/trajectories'
SAVE_PATH_WATER = '/data/manon/MasterThesis/HYCOM'
SAVE_PATH_WIND = '/data/manon/MasterThesis/ERA5_Wind'
SAVE_PATH_WAVES = '/data/manon/MasterThesis/ERA5_Waves'

def get_days(start, end):
    span = end - start
    for i in range(span.days):
        datei = start+timedelta(days=i)
        yield str(int(datei.day)).zfill(2)


def create_all_configs(list_to_config, path_configs = PATH_CONFIGS):

    c = cdsapi.Client()

    with open(list_to_config, 'rb') as f:
        list_train = pickle.load(f)

    for data_file in tqdm(list_train):

        # test if config already exists
        data_name = data_file
        config_name = path_configs + '/config_' + data_name[:-4]+'.yml'
        if os.path.exists(config_name):
            continue
        
        # 1st step: read trace
        data_file = os.path.join(PATH_TRAJ, data_file)
        lon, lat, time_drift = get_true_drift_positions(data_file, NOAA=True)

        # get frame
        lat_max = math.ceil(max(lat)+1)
        lat_min = math.floor(min(lat)-1)
        lon_max = math.ceil(max(lon)+1)
        lon_min = math.floor(min(lon)-1)

        # get time
        min_time = min(time_drift) - 1
        max_time = max(time_drift) + 1
        # Convert time from hours back to date+time
        min_date = convert_date_back(min_time)
        max_date = convert_date_back(max_time) + timedelta(days=1)
        date_str_min = str(min_date.year)+'-'+str(min_date.month)+'-'+str(min_date.day)
        date_str_max = str(max_date.year)+'-'+str(max_date.month)+'-'+str(max_date.day)

        #print('     Opening URL and downloading data...')

        ################################### HYCOM DATA #######################################################3

        #print('Downloading HYCOM data...')

        if min_date < datetime(year=2014,month=7,day=1):
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_53.X/data/'+str(min_date.year)+'?var=water_u&var=water_v&north='+str(lat_max) +'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        elif min_date < datetime(year=2016, month = 4, day = 30):
            if max_date > datetime(year=2016, month = 4, day = 30):
                continue
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_56.3?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        elif min_date < datetime(year=2017, month = 1, day = 31):
            if max_date > datetime(year=2017, month = 1, day = 31):
                continue
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_57.2?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        elif min_date < datetime(year=2017, month = 5, day = 31):
            if max_date > datetime(year=2017, month = 5, day = 31):
                continue
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.8?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        elif min_date < datetime(year=2017, month = 9, day = 30):
            if max_date > datetime(year=2017, month = 9, day = 30):
                continue
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_57.7?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        elif min_date < datetime(year=2017, month = 12, day = 31):
            if max_date > datetime(year=2017, month = 12, day = 31):
                continue
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.9?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        elif min_date < datetime(year=2020, month = 2, day = 18):
            if max_date > datetime(year=2020, month = 2, day = 18):
                continue
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_93.0?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        else: 
            url = 'https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/uv3z?var=water_u&var=water_v&north'+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        
        
        response = requests.get(url)
        path_water = SAVE_PATH_WATER + '/HYCOM_' + data_name[:-4] + '.nc4'
        open(path_water, "wb").write(response.content)

        ################################## WIND DATA (ERA5) ############################################################

        #print('Downloading Wind data...')

        # get year(s)
        years = list()
        if min_date.year != max_date.year:
            years.append(str(min_date.year))
            years.append(str(max_date.year))
        else:
            years.append(str(min_date.year))
        
        # get month(s)
        months = list()
        if min_date.month != max_date.month:
            months.append(str(min_date.month))
            months.append(str(max_date.month))
        else:
            months.append(str(min_date.month))

        # get days
        days = list(get_days(start=min_date, end = max_date))

        # filename 
        path_wind = SAVE_PATH_WIND + '/Wind_' + data_name[:-4] + '.nc'

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'area': [
                    lat_max, lon_min, lat_min,lon_max
                ],
                'product_type': ['reanalysis'],
                'data_format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind',
                ],
                'year': years,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'day': days,
                'month': months,
                'area': [
                    lat_max, lon_min, lat_min,lon_max
                ]
            },
            path_wind)
    
        ################################## WAVE DATA (ERA5) ###########################################################

        #print('Downloading Waves data...')

        path_waves = SAVE_PATH_WAVES + '/Waves_' + data_name[:-4] + '.nc'

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'area': [
                    lat_max, lon_min, lat_min,lon_max
                ],
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': [
                    'u_component_stokes_drift', 'v_component_stokes_drift',
                ],
                'year': years,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'day': days,
                'month': months,
            },
            path_waves)
        
        ################################# BATHYMETRY #############################################################

        '''if (((lat_max >= 0 and lat_max <= 90) and (lat_min >= 0 and lat_min <= 90)) and ((lon_max >= -180 and lon_max <= -60) and (lon_min >= -180 and lon_min <= -60))):
            path_bathy = '../data/GEBCO_Bathymetry/gebco_2023_n90.0_s0.0_w-180.0_e-60.0.nc'

        elif (((lat_max >= 0 and lat_max <= 90) and (lat_min >= 0 and lat_min <= 90)) and ((lon_max >= -60 and lon_max <= 60) and (lon_min >= -60 and lon_min <= 60))):
            path_bathy = '../data/GEBCO_Bathymetry/gebco_2023_n90.0_s0.0_w-60.0_e60.0.nc'

        elif (((lat_max >= 0 and lat_max <= 90) and (lat_min >= 0 and lat_min <= 90)) and ((lon_max >= 60 and lon_max <= 180) and (lon_min >= 60 and lon_min <= 180))):
            path_bathy = '../data/GEBCO_Bathymetry/gebco_2023_n90.0_s0.0_w60.0_e180.0.nc'

        elif (((lat_max >= -90 and lat_max <= 0) and (lat_min >= -90 and lat_min <= 0)) and ((lon_max >= -180 and lon_max <= -60) and (lon_min >= -180 and lon_min <= -60))):
            path_bathy = '../data/GEBCO_Bathymetry/gebco_2023_n0.0_s-90.0_w-180.0_e-60.0.nc'

        elif (((lat_max >= -90 and lat_max <= 0) and (lat_min >= -90 and lat_min <= 0)) and ((lon_max >= -60 and lon_max <= 60) and (lon_min >= -60 and lon_min <= 60))):
            path_bathy = '../data/GEBCO_Bathymetry/gebco_2023_n0.0_s-90.0_w-60.0_e60.0.nc'

        elif (((lat_max >= -90 and lat_max <= 0) and (lat_min >= -90 and lat_min <= 0)) and ((lon_max >= 60 and lon_max <= 180) and (lon_min >= 60 and lon_min <= 180))):
            path_bathy = '../data/GEBCO_Bathymetry/gebco_2023_n0.0_s-90.0_w60.0_e180.0.nc' '''

        ################################## CONFIG ################################################################

        dic = {'PATH_DRIFT': data_file, 
            'PATH_WIND': path_wind,
            'PATH_WATER' : path_water,
            'PATH_WAVES': path_waves,
            #'PATH_BATHY': path_bathy,
            'min_lon': lon_min,
            'max_lon': lon_max,
            'min_lat': lat_min,
            'max_lat': lat_max}
        
        #print('     Saving config file...')

        with open(config_name, 'w') as yaml_file:
            yaml.dump(dic, yaml_file, default_flow_style=False)
        #print('     Done!')
    return



if __name__ == "__main__": 

    create_all_configs(list_to_config=LIST_TEST, path_configs=PATH_CONFIGS_TEST)
        



    