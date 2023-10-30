import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utils.convert_lats import dist_latitudes,dist_longitudes
import math

######## Homogenization of time format ########################################

def convert_date(strdate):
    try:
        date_datetime = datetime.strptime(strdate,'%Y-%m-%dT%H:%M:%S')
    except:
        date_datetime = datetime.strptime(strdate,'%Y-%m-%dT%H:%M')
    reference = datetime(2000,1,1,0,0,0)
    duration = date_datetime-reference
    hours = duration.total_seconds()/3600
    return hours

def convert_date_back(nhours):
    # returns day but not hour
    reference = datetime(2000,1,1,0,0,0)
    time_delta = timedelta(seconds=nhours*3600)
    date = reference + time_delta
    return date

def convert_hours_1900_2000(hours1900):
    ref1900 = datetime(1900,1,1,0,0,0)
    ref2000 = datetime(2000,1,1,0,0,0)
    century_diff = ref2000-ref1900
    century_h = century_diff.total_seconds()/3600
    return hours1900-century_h

############# Drift position from Canada Data ##################################

def read_drift_positions(path_drift):
    drift_tab = pd.read_csv(path_drift,sep='\t',skiprows=13)
    drift_tab['hours'] = drift_tab['Date/Time'].apply(lambda x: convert_date(x))
    return drift_tab

def get_initial_position(path_drift):
    drift_tab = read_drift_positions(path_drift)
    pos_1 = np.array([drift_tab._get_value(0,'Longitude'), drift_tab._get_value(0,'Latitude')])
    #pos_2 = np.array([drift_tab._get_value(1,'Longitude'), drift_tab._get_value(1,'Latitude')])
    time_1 = drift_tab._get_value(0,'hours')
    #time_2 = drift_tab._get_value(1,'hours')
    return pos_1, time_1

def get_true_drift_positions(path_drift):
    drift_tab = read_drift_positions(path_drift)

    lat = drift_tab['Latitude'].tolist()
    lon = drift_tab['Longitude'].tolist()
    time = drift_tab['hours'].tolist()

    return lon, lat, time

def get_closest_position(path_drift, Nhours):
    drift_tab = read_drift_positions(path_drift)
    time_1 = drift_tab._get_value(0,'hours')
    time_N = time_1 + Nhours
    lon, lat, time = get_true_drift_positions(path_drift)
    lon_N = np.interp(time_N, time, lon)
    lat_N = np.interp(time_N, time, lat)
    return lon_N, lat_N

def get_true_velocity(path_drift):
    lons, lats, times = get_true_drift_positions(path_drift)
    velocity = np.zeros((len(lons)-1,2))
    for i in range(len(lons)-1):
        velocity[i,0] = dist_longitudes(lons[i+1],lons[i],lats[i+1],lats[i])/((times[i+1]-times[i])*3600)
        velocity[i,1] = dist_latitudes(lats[i+1],lats[i])/((times[i+1]-times[i])*3600)
    hours = [t- times[0] for t in times]
    return velocity, hours

def get_number_hours(times):
    return math.floor(times[-1]-times[0])

def get_extrapolated_true_position(path_drift, nhours):
    true_lat_extrapolated = np.zeros(nhours+1)
    true_lon_extrapolated = np.zeros(nhours+1)

    posinit, _ = get_initial_position(path_drift)
    true_lon_extrapolated[0], true_lat_extrapolated[0] = posinit[0], posinit[1]

    for i in range(nhours):
        true_lon_extrapolated[i+1], true_lat_extrapolated[i+1] = get_closest_position(path_drift,i+1)

    return true_lon_extrapolated, true_lat_extrapolated

############# Wind Data ########################################################

def get_wind_data(path_wind):
    ds = nc.Dataset(path_wind)
    return ds

def wind_interpolated(path_wind):
    ds = nc.Dataset(path_wind)

    lon = ds.variables['longitude']
    lon = lon[:].data

    lat = ds.variables['latitude']
    lat = lat[:].data

    time = ds.variables['time']
    time = time[:].data
    vfunc = np.vectorize(convert_hours_1900_2000) # convert time to fit the rest of the data
    time = vfunc(time)

    # Wind data
    u10 = ds.variables['u10']
    u10 = u10[:].data
    v10 = ds.variables['v10']
    v10 = v10[:].data

    # Interpolating functions for u10 and v10
    u10_interpolation = RegularGridInterpolator((time, lat, lon), u10, bounds_error=False)
    v10_interpolation = RegularGridInterpolator((time, lat, lon), v10, bounds_error=False)

    return u10_interpolation, v10_interpolation

############# Wave Data ########################################################

def get_wave_data(path_wave):
    ds = nc.Dataset(path_wave)
    return ds

def wave_interpolated(path_wave):
    ds = nc.Dataset(path_wave)

    lon = ds.variables['longitude']
    lon = lon[:].data

    lat = ds.variables['latitude']
    lat = lat[:].data

    time = ds.variables['time']
    time = time[:].data
    vfunc = np.vectorize(convert_hours_1900_2000) # convert time to fit the rest of the data
    time = vfunc(time)

    # Wind data
    ust = ds.variables['ust']
    ust = ust[:].data
    ust = np.where(ust == -32767, np.nan, ust)
    vst = ds.variables['vst']
    vst = vst[:].data
    vst = np.where(vst == -32767, np.nan, vst)

    # Interpolating functions for ust and vst
    ust_interpolation = RegularGridInterpolator((time, lat, lon), ust, bounds_error=False)
    vst_interpolation = RegularGridInterpolator((time, lat, lon), vst, bounds_error=False)

    return ust_interpolation, vst_interpolation


############# Ocean Data #######################################################


def water_interpolated(path_water):
    ds = nc.Dataset(path_water)

    lon = ds.variables['lon']
    lon = lon[:].data

    lat = ds.variables['lat']
    lat = lat[:].data

    time = ds.variables['time']
    time = time[:].data

    depth = ds.variables['depth']
    depth = depth[:].data

    # Ocean current data
    water_u = ds.variables['water_u']
    water_u = water_u[:].data
    water_v = ds.variables['water_v']
    water_v = water_v[:].data

    water_u = np.where(water_u < -10000, np.nan, water_u)
    water_v = np.where(water_v < -10000, np.nan, water_v)


    # Interpolating functions for u10 and v10
    water_u_interpolation = RegularGridInterpolator((time, depth, lat, lon), water_u, bounds_error=False)
    water_v_interpolation = RegularGridInterpolator((time, depth, lat, lon), water_v, bounds_error=False)

    return water_u_interpolation,water_v_interpolation

############# Bathymetry/Coasts Data #######################################################

def bathymetry_interpolated(path_bathy):
    ds = nc.Dataset(path_bathy)

    lon = ds.variables['lon']
    lon = lon[:].data

    lat = ds.variables['lat']
    lat = lat[:].data

    elevation = ds.variables['elevation']
    elevation = elevation[:].data

    # Interpolating functions for u10 and v10
    elevation_interpolation = RegularGridInterpolator((lat, lon), elevation, bounds_error=False)

    return elevation_interpolation