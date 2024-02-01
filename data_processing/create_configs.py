import os 
import webbrowser
import pandas as pd
from utils.read_data import read_drift_positions, convert_date_back
from datetime import datetime, timedelta
import math
import time
import yaml

# This is the path where all the files are stored.
FOLDER_PATH = '../data/Dumont-etal_2019/datasets2'

def create_all_configs():

    for data_file in os.listdir(FOLDER_PATH):

        data_name = data_file
        # 1st step: read trace
        print('Reading ' + data_file + ' ...')
        data_file = os.path.join(FOLDER_PATH, data_file)
        drift_tab = read_drift_positions(data_file)

        # creating variable for latitude and longitude to list
        lat = drift_tab['Latitude'].tolist()
        lon = drift_tab['Longitude'].tolist()

        # get frame
        lat_max = math.ceil(max(lat)+1)
        lat_min = math.floor(min(lat)-1)
        lon_max = math.ceil(max(lon)+1)
        lon_min = math.floor(min(lon)-1)

        # get time
        time_drift = drift_tab['hours'].tolist()
        min_time = min(time_drift) - 1
        max_time = max(time_drift) + 1
        # Convert time from hours back to date+time
        min_date = convert_date_back(min_time)
        max_date = convert_date_back(max_time) + timedelta(days=1)
        date_str_min = str(min_date.year)+'-'+str(min_date.month)+'-'+str(min_date.day)
        date_str_max = str(max_date.year)+'-'+str(max_date.month)+'-'+str(max_date.day)

        print('     Opening URL and downloading data...')

        if min_date < datetime(year=2014,month=7,day=1):
            name_data = 'data_2014'
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_53.X/data/'+str(min_date.year)+'?var=water_u&var=water_v&north='+str(lat_max) +'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
        else:
            name_data = 'GLBv0_expt_56'
            url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_56.3?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'

        webbrowser.open(url,new=0)

        while not os.path.exists('c:/Users/manon/Downloads/'+name_data+'.nc4'):
            time.sleep(1)
        
        time.sleep(5)


        print('     Renaming data...')
        os.rename('c:/Users/manon/Downloads/'+name_data+'.nc4', 'c:/Users/manon/Downloads/data_'+data_name[:-4]+'.nc4')

        dic = {'PATH_DRIFT': data_file, 
               'PATH_WIND':'C:/Users/manon/Desktop/Thesis/data/ERA5_Wind/WindData.nc',
               'PATH_WATER' : 'C:/Users/manon/Desktop/Thesis/data/HYCOM/data_'+data_name[:-4]+'.nc4',
               'PATH_WAVES': 'C:/Users/manon/Desktop/Thesis/data/ERA5_Wind/WaveData.nc',
               'min_lon': lon_min,
               'max_lon': lon_max,
               'min_lat': lat_min,
               'max_lat': lat_max}
        
        print('     Saving config file...')
        with open('C:/Users/manon/Desktop/Thesis/code/configs/trace_'+data_name[:-4]+'.yml', 'w') as yaml_file:
            yaml.dump(dic, yaml_file, default_flow_style=False)
        print('     Done!')
        time.sleep(10)
    return


def create_one_configs(data_file):

    data_name = data_file
    # 1st step: read trace
    print('Reading ' + data_file + ' ...')
    data_file = os.path.join(FOLDER_PATH, data_file)
    drift_tab = read_drift_positions(data_file)

    # creating variable for latitude and longitude to list
    lat = drift_tab['Latitude'].tolist()
    lon = drift_tab['Longitude'].tolist()

    # get frame
    lat_max = math.ceil(max(lat)+1)
    lat_min = math.floor(min(lat)-1)
    lon_max = math.ceil(max(lon)+1)
    lon_min = math.floor(min(lon)-1)

    # get time
    time_drift = drift_tab['hours'].tolist()
    min_time = min(time_drift) - 1
    max_time = max(time_drift) + 1
    # Convert time from hours back to date+time
    min_date = convert_date_back(min_time)
    max_date = convert_date_back(max_time) + timedelta(days=1)
    date_str_min = str(min_date.year)+'-'+str(min_date.month)+'-'+str(min_date.day)
    date_str_max = str(max_date.year)+'-'+str(max_date.month)+'-'+str(max_date.day)

    print('     Opening URL and downloading data...')

    if min_date < datetime(year=2014,month=7,day=1):
        name_data = 'data_2014'
        url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_53.X/data/'+str(min_date.year)+'?var=water_u&var=water_v&north='+str(lat_max) +'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'
    else:
        name_data = 'GLBv0_expt_56'
        url = 'https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_56.3?var=water_u&var=water_v&north='+str(lat_max)+'&west='+str(lon_min)+'&east='+str(lon_max)+'&south='+str(lat_min)+'&disableProjSubset=on&horizStride=1&time_start='+date_str_min+'T00%3A00%3A00Z&time_end='+date_str_max+'T00%3A00%3A00Z&timeStride=1&vertCoord=0&accept=netcdf4'

    webbrowser.open(url,new=0)

    while not os.path.exists('c:/Users/manon/Downloads/'+name_data+'.nc4'):
        time.sleep(1)

    time.sleep(5)
    print('     Renaming data...')
    os.rename('c:/Users/manon/Downloads/'+name_data+'.nc4', 'c:/Users/manon/Downloads/data_'+data_name[:-4]+'.nc4')

    dic = {'PATH_DRIFT': data_file, 
            'PATH_WIND':'C:/Users/manon/Desktop/Thesis/data/ERA5_Wind/WindData.nc',
            'PATH_WATER' : 'C:/Users/manon/Desktop/Thesis/data/HYCOM/data_'+data_name[:-4]+'.nc4',
            'PATH_WAVES': 'C:/Users/manon/Desktop/Thesis/data/ERA5_Wind/WaveData.nc',
            'min_lon': lon_min,
            'max_lon': lon_max,
            'min_lat': lat_min,
            'max_lat': lat_max}
    
    print('     Saving config file...')
    with open('C:/Users/manon/Desktop/Thesis/code/configs/trace_'+data_name[:-4]+'.yml', 'w') as yaml_file:
        yaml.dump(dic, yaml_file, default_flow_style=False)
    print('     Done!')
    return



if __name__ == "__main__": 

    #create_one_configs('ISMER_20141022_spot243_drift.tab')
    create_all_configs()
        



    