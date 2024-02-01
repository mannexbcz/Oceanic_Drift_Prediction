import os
import pandas as pd
from utils.read_data import read_drift_positions
from data_processing.get_24h_trajectories import get_time_continuous_path
import yaml
import numpy as np
from tqdm import tqdm

CONFIG_PATH = './configs_2'
#PATH_BATHY = '../data/GEBCO_Bathymetry/GEBCO_25_Oct_2023_ba927c4c060f/gebco_2023_n52.0_s44.0_w-74.0_e-53.0.nc'

if __name__ == "__main__": 

    dataset = pd.DataFrame(columns=['Latitude_init','Longitude_init', 'time_init', 'Latitude_final', 'Longitude_final', 
                               'PATH_WATER', 'PATH_WIND', 'PATH_WAVES'])
    
    for data_file in tqdm(os.listdir(CONFIG_PATH),desc='Trajectories'):

        with open(os.path.join(CONFIG_PATH,data_file), 'r') as f:
            config = yaml.safe_load(f)

        drift_tab = read_drift_positions(config['PATH_DRIFT'])

        list_df_ok = get_time_continuous_path(drift_tab)

        for df in list_df_ok:

            lats = df['Latitude'].tolist()
            lons = df['Longitude'].tolist()
            times = df['hours'].tolist()

            for i in range(len(df)-1):
                lat_init, lon_init, time_init = df['Latitude'].iloc[i], df['Longitude'].iloc[i], df['hours'].iloc[i]
                lon_final = np.interp(time_init+1, times, lons)
                lat_final = np.interp(time_init+1, times, lats)
                
                new_row = pd.DataFrame({'Latitude_init':lat_init, 'Longitude_init':lon_init,'time_init': time_init, 'Latitude_final':lat_final, 'Longitude_final':lon_final,
                                        'PATH_WATER': config['PATH_WATER'], 'PATH_WIND':config['PATH_WIND'], 'PATH_WAVES':config['PATH_WAVES'], 'PATH_BATHY': config['PATH_BATHY']}, index=[0])
                dataset = pd.concat([new_row,dataset.loc[:]]).reset_index(drop=True)

    print('Length of dataset:', len(dataset))
    dataset.to_csv('../data/nextpoint_ds/next_point_dataset.csv')



