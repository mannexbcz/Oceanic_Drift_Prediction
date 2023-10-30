import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_processing.context import *

class DriftPairDataset(Dataset):
    def __init__(self, csvfile, d_context = 1, npoints = 32):
        self.tab = pd.read_csv(csvfile)
        self.d_context = d_context
        self.npoints = npoints

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):

        init_lat, init_lon = self.tab['Latitude_init'].iloc[idx], self.tab['Longitude_init'].iloc[idx]
        init_position = torch.tensor([[init_lat], [init_lon]], dtype=torch.double)
        final_lat, final_lon = self.tab['Latitude_final'].iloc[idx], self.tab['Longitude_final'].iloc[idx]
        final_position = torch.tensor([[final_lat], [final_lon]], dtype=torch.double)
        init_time = self.tab['time_init'].iloc[idx]

        # Paths for the context
        path_water = self.tab['PATH_WATER'].iloc[idx]
        path_wind = self.tab['PATH_WIND'].iloc[idx]
        path_waves = self.tab['PATH_WAVES'].iloc[idx]
        path_bathy = self.tab['PATH_BATHY'].iloc[idx]

        # Dictionnary with the paths
        dict_path = {
            'PATH_WATER': path_water, 
            'PATH_WIND' : path_wind, 
            'PATH_WAVES' : path_waves,
            'PATH_BATHY' : path_bathy
        }

        # get contexts
        context_water_u, context_water_v = get_water_context(path_water, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)
        context_wind_u, context_wind_v = get_wind_context(path_wind, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)
        context_waves_u, context_waves_v = get_waves_context(path_waves, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)
        context_bathymetry, context_coasts = get_bathymetry_context(path_bathy, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)

        # merge contextes
        context = np.dstack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v, context_bathymetry, context_coasts))
        assert np.shape(context) == (self.npoints,self.npoints,8), f"Wrong shape for the context: {np.shape(context)}"
        context = torch.from_numpy(context)
        
        return init_position, final_position, init_time, context, dict_path
