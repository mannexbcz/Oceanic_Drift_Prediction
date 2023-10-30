import os 
import pandas as pd
import yaml
import netCDF4 as nc
import numpy as np
from utils.read_data import *
from utils.param_alpha import get_alpha, get_complex_alpha, compute_alpha_whole_trajectory,compute_alpha_matrix, get_alpha_averaged, update_alpha_GD
from models.linear_model import u_drift_linear, u_drift_linear_complex, u_drift_linear_matrix
from utils.RK4 import RK4_step, compute_position
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pickle

FOLDER_PATH = './configs_2'

def check_HYCOM_data():

    for data_file in os.listdir(FOLDER_PATH):
        with open(os.path.join(FOLDER_PATH,data_file), 'r') as f:
            config = yaml.safe_load(f)

        print('Opening ',data_file)

        true_lon, true_lat, true_time = get_true_drift_positions(config['PATH_DRIFT'])

        ds = nc.Dataset(config['PATH_WATER'])

        lat, lon = ds.variables['lat'], ds.variables['lon']
        water_u = ds.variables['water_u']
        water_v = ds.variables['water_v']

        x, y = np.meshgrid(list(lon),list(lat)) 

        time = ds.variables['time']
        time[:].data

        print(time[0], time[-1], '(must be wider)')
        print(true_time[0], true_time[-1])

        if time[0]<true_time[0] and time[-1]>true_time[-1]:
            print('Times OK')
        else:
            print('TIMES NOT OK!')

        print('Length of the trace: ', len(true_time))

        # setting the size of the map
        fig,ax = plt.subplots(figsize=(8,5))

        # creating the map - setting latitude and longitude
        m = Basemap(projection = 'mill', llcrnrlat = 44, urcrnrlat = 52, llcrnrlon = -74, urcrnrlon = -53, resolution = 'i') 

        # drawing the coastline
        m.drawcoastlines()
        m.drawcoastlines()
        m.fillcontinents(color='darkseagreen')

        x_,y_ = m(x,y)

        Q = ax.quiver(x_, y_, water_u[3,0,:,:], water_v[3,0,:,:]) #, scale=5, width = 0.005) 
        m.plot(true_lon,true_lat,latlon=True,linewidth=1.5,color='r', label='True trajectory')

        plt.legend(loc = 'lower right',framealpha=1)
        plt.tight_layout()
        plt.show()

        ok = input('Can we go to the next iteration?')
    
    return

def check_near_shore_data():
    listok = []
    for data_file in os.listdir(FOLDER_PATH):
        with open(os.path.join(FOLDER_PATH,data_file), 'r') as f:
            config = yaml.safe_load(f)

        print('Opening ',data_file)

        true_lon, true_lat, true_time = get_true_drift_positions(config['PATH_DRIFT'])

        # setting the size of the map
        fig,ax = plt.subplots(figsize=(8,5))

        # creating the map - setting latitude and longitude
        m = Basemap(projection = 'mill', llcrnrlat = 44, urcrnrlat = 52, llcrnrlon = -74, urcrnrlon = -53, resolution = 'i') 

        # drawing the coastline
        m.drawcoastlines()
        m.drawcoastlines()
        m.fillcontinents(color='darkseagreen')

        m.plot(true_lon,true_lat,latlon=True,linewidth=1.5,color='r', label='True trajectory')

        plt.legend(loc = 'lower right',framealpha=1)
        plt.tight_layout()
        plt.show()

        ok = input('Are we keeping this trace?')

        if ok == 'Y':
            listok.append(data_file)

    with open("traces_wo_shoring.txt", "wb") as fp:   #Pickling
        pickle.dump(listok, fp)
    
    return



if __name__ == "__main__": 
    check_near_shore_data()