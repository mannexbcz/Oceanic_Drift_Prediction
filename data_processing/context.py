from utils.read_data import water_interpolated, wind_interpolated, wave_interpolated, bathymetry_interpolated
from utils.convert_lats import new_latitude, new_longitude
import numpy as np

def get_water_context(path_water, lat_init, lon_init, time_init, d = 1, npoints = 32):

    water_u_interpolation,water_v_interpolation = water_interpolated(path_water)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_water_u = np.zeros([npoints, npoints])
    context_water_v = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_water_u[i,j] = water_u_interpolation([time_init,0,lats[j],lons[i]])
            context_water_v[i,j] = water_v_interpolation([time_init,0,lats[j],lons[i]])

    return context_water_u, context_water_v

def get_wind_context(path_wind, lat_init, lon_init, time_init, d = 1, npoints = 32):

    wind_u_interpolation,wind_v_interpolation = wind_interpolated(path_wind)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_wind_u = np.zeros([npoints, npoints])
    context_wind_v = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_wind_u[i,j] = wind_u_interpolation([time_init,lats[j],lons[i]])
            context_wind_v[i,j] = wind_v_interpolation([time_init,lats[j],lons[i]])

    return context_wind_u, context_wind_v


def get_waves_context(path_waves, lat_init, lon_init, time_init, d = 1, npoints = 32):

    ust_interpolation,vst_interpolation = wave_interpolated(path_waves)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_waves_u = np.zeros([npoints, npoints])
    context_waves_v = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_waves_u[i,j] = ust_interpolation([time_init,lats[j],lons[i]])
            context_waves_v[i,j] = vst_interpolation([time_init,lats[j],lons[i]])

    return context_waves_u, context_waves_v

def get_bathymetry_context(path_baythy, lat_init, lon_init, time_init, d = 1, npoints = 32):

    elevation_interpolation = bathymetry_interpolated(path_baythy)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_bathymetry = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_bathymetry[i,j] = elevation_interpolation([lats[j],lons[i]])

    context_coasts = np.where(context_bathymetry>0, 0,1)

    return context_bathymetry, context_coasts


def get_context(path_water, path_wind, path_waves, path_bathy, init_lat, init_lon, init_time, d_context=1, npoints=32):

    context_water_u, context_water_v = get_water_context(path_water, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_wind_u, context_wind_v = get_wind_context(path_wind, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_waves_u, context_waves_v = get_waves_context(path_waves, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_bathymetry, context_coasts = get_bathymetry_context(path_bathy, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    # merge contextes
    #print('Merging context')
    context = np.stack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v, context_bathymetry, context_coasts))
    assert np.shape(context) == (8,npoints,npoints), f"Wrong shape for the context: {np.shape(context)}"

    return context