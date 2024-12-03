import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_processing.context import *
from models.physical_model import get_physical_model
import yaml

class DriftPairDataset(Dataset):
    def __init__(self, csvfile, d_context = 1, npoints = 32):
        self.tab = pd.read_csv(csvfile)
        self.d_context = d_context
        self.npoints = npoints
        self.physical_model = get_physical_model()

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):

        #print('Getting positions')
        init_lat, init_lon = self.tab['Latitude_init'].iloc[idx], self.tab['Longitude_init'].iloc[idx]
        init_position = torch.tensor([init_lat,init_lon], dtype = torch.float)
        final_lat, final_lon = self.tab['Latitude_final'].iloc[idx], self.tab['Longitude_final'].iloc[idx]
        final_position = torch.tensor([final_lat, final_lon], dtype = torch.float)
        init_time = self.tab['time_init'].iloc[idx]

        # Paths for the context
        #print('Getting paths')
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
        #print('Getting context')
        context_water_u, context_water_v = get_water_context(path_water, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)
        context_wind_u, context_wind_v = get_wind_context(path_wind, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)
        context_waves_u, context_waves_v = get_waves_context(path_waves, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)
        context_bathymetry, context_coasts = get_bathymetry_context(path_bathy, init_lat, init_lon, init_time, d = self.d_context, npoints = self.npoints)

        # merge contextes
        #print('Merging context')
        context = np.stack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v, context_bathymetry, context_coasts))
        assert np.shape(context) == (8,self.npoints,self.npoints), f"Wrong shape for the context: {np.shape(context)}"
        context = torch.from_numpy(context.astype(np.float32))
        #context.requires_grad=True

        #print('Computing xphys')
        xphys = self.physical_model(init_position, init_time, dict_path)
        #xphys = torch.Tensor([49,-65])
        
        return xphys, final_position, context



class DriftPairDataset_Wo_Computation(Dataset):
    def __init__(self, csvfile):
        self.tab = pd.read_csv(csvfile)
        
    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):

        initial_lat, initial_lon = self.tab['Latitude_init'].iloc[idx], self.tab['Longitude_init'].iloc[idx]
        initial_position = torch.tensor([initial_lat, initial_lon], dtype = torch.float)

        final_lat, final_lon = self.tab['Latitude_final'].iloc[idx], self.tab['Longitude_final'].iloc[idx]
        final_position = torch.tensor([final_lat, final_lon], dtype = torch.float)

        xphys_lat, xphys_lon = self.tab['Lat_phys'].iloc[idx], self.tab['Lon_phys'].iloc[idx]
        xphys = torch.tensor([xphys_lat, xphys_lon], dtype = torch.float)

        # get context
        context_path = self.tab['PATH_CONTEXT'].iloc[idx]
        with open(context_path, 'rb') as f:
            context = np.load(f)
        context = torch.from_numpy(context.astype(np.float32))
        #context.requires_grad=True
        
        return initial_position, xphys, final_position, context #[0:-2,:,:]


class DriftPairDataset_W_Previous(Dataset):
    def __init__(self, csvfile):
        self.tab = pd.read_csv(csvfile)
        
    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):

        final_lat, final_lon = self.tab['Latitude_final'].iloc[idx], self.tab['Longitude_final'].iloc[idx]
        final_position = torch.tensor([final_lat, final_lon], dtype = torch.float)

        prev_lat, prev_lon = self.tab['Latitude_prev'].iloc[idx], self.tab['Longitude_prev'].iloc[idx]
        prev_position = torch.tensor([prev_lat, prev_lon], dtype = torch.float)

        xphys_lat, xphys_lon = self.tab['Lat_phys'].iloc[idx], self.tab['Lon_phys'].iloc[idx]
        xphys = torch.tensor([xphys_lat, xphys_lon], dtype = torch.float)

        # get context
        context_path = self.tab['PATH_CONTEXT'].iloc[idx]
        with open(context_path, 'rb') as f:
            context = np.load(f)
        context = torch.from_numpy(context.astype(np.float32))
        #context = context[0:-2,:,:]

        prev_context_path = self.tab['PREVIOUS_PATH_CONTEXT'].iloc[idx]
        with open(prev_context_path, 'rb') as f:
            prev_context = np.load(f)
        prev_context = torch.from_numpy(prev_context.astype(np.float32))
        #prev_context = prev_context[0:-2,:,:]

        final_context = torch.cat((context, prev_context), 0)

        return xphys, final_position, prev_position, final_context


class TrajectoryDataset(Dataset):
    def __init__(self, csvfile):
        self.tab = pd.read_csv(csvfile)
        
    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):

        lat0, lon0 = self.tab['Latitude_init'].iloc[idx], self.tab['Longitude_init'].iloc[idx]
        pos0 = torch.tensor([lat0, lon0], dtype = torch.float)

        lat1, lon1 = self.tab['Latitude_1'].iloc[idx], self.tab['Longitude_1'].iloc[idx]
        pos1 = torch.tensor([lat1, lon1], dtype = torch.float)

        lat2, lon2 = self.tab['Latitude_2'].iloc[idx], self.tab['Longitude_2'].iloc[idx]
        pos2 = torch.tensor([lat2, lon2], dtype = torch.float)

        lat3, lon3 = self.tab['Latitude_3'].iloc[idx], self.tab['Longitude_3'].iloc[idx]
        pos3 = torch.tensor([lat3, lon3], dtype = torch.float)

        # get context
        context_path = self.tab['PATH_CONTEXT'].iloc[idx]
        with open(context_path, 'rb') as f:
            context = np.load(f)
        context = torch.from_numpy(context.astype(np.float32))
        context = context[0:-2,:,:]

        time0 = self.tab['time_init'].iloc[idx]

        with open(self.tab['CONFIG_PATH'].iloc[idx], 'r') as f:
            config = yaml.safe_load(f)

        return pos0, pos1, pos2, pos3, time0, config, context


class TrajectoryDataset_Dynamic(Dataset):
    def __init__(self, csvfile):
        self.tab = pd.read_csv(csvfile)
        
    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):

        ['', 'PATH_WATER', 'PATH_WAVES']

        lat0, lon0 = self.tab['lat0'].iloc[idx], self.tab['lon0'].iloc[idx]
        pos0 = torch.tensor([lat0, lon0], dtype = torch.float)

        lat1, lon1 = self.tab['lat1'].iloc[idx], self.tab['lon1'].iloc[idx]
        pos1 = torch.tensor([lat1, lon1], dtype = torch.float)

        lat2, lon2 = self.tab['lat2'].iloc[idx], self.tab['lon2'].iloc[idx]
        pos2 = torch.tensor([lat2, lon2], dtype = torch.float)

        lat3, lon3 = self.tab['lat3'].iloc[idx], self.tab['lon3'].iloc[idx]
        pos3 = torch.tensor([lat3, lon3], dtype = torch.float)

        time0 = self.tab['time_init'].iloc[idx]

        with open(self.tab['CONFIG_PATH'].iloc[idx], 'r') as f:
            config = yaml.safe_load(f)

        '''PATH_WIND = self.tab['PATH_WIND'].iloc[idx]
        PATH_WATER = self.tab['PATH_WATER'].iloc[idx]
        PATH_WAVES = self.tab['PATH_WAVES'].iloc[idx]'''

        return pos0, pos1, pos2, pos3, time0, config #, PATH_WATER, PATH_WIND, PATH_WAVES