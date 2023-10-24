# Implementation of Runge Kutta 4 scheme
from utils.convert_lats import new_latitude, new_longitude,dist_latitudes,dist_longitudes
from utils.read_data import get_closest_position
import numpy as np

def RK4_step(u_drift, pos_init, time_init, dt):
    dt_h = dt
    dt = dt*3600

    lon_init,lat_init = pos_init

    k1 = u_drift(time_init,lon_init,lat_init)
    k2 = u_drift(time_init+dt_h/2, new_longitude(lat_init,lon_init,k1[0]*dt/2), new_latitude(lat_init, lon_init,k1[1]*dt/2))
    k3 = u_drift(time_init+dt_h/2, new_longitude(lat_init,lon_init,k2[0]*dt/2), new_latitude(lat_init, lon_init,k2[1]*dt/2))
    k4 = u_drift(time_init+dt_h, new_longitude(lat_init,lon_init,k3[0]*dt), new_latitude(lat_init,lon_init,k3[1]*dt))

    lon_new = new_longitude(lat_init,lon_init, dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6)
    lat_new = new_latitude(lat_init, lon_init, dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6)

    pos_new = np.array([lon_new,lat_new])
    
    time_new = time_init + dt_h

    return pos_new, time_new


def compute_position(u_drift, pos_init, time_init,dt, n_steps):
    longitudes = np.zeros(n_steps+1)
    latitudes = np.zeros(n_steps+1)

    longitudes[0] = pos_init[0]
    latitudes[0] = pos_init[1]

    for i in range(n_steps):
        pos_new, time_new = RK4_step(u_drift, pos_init, time_init, dt)
        if np.isnan(pos_new).any():
            #print('Trajectory stopped after ', i, ' hours, Nan values')
            longitudes[i+1:] = longitudes[i]
            latitudes[i+1:] = latitudes[i]
            #print(latitudes)
            #print(longitudes)
            break
        longitudes[i+1] = pos_new[0]
        latitudes[i+1] = pos_new[1]
        pos_init = pos_new
        time_init = time_new
        

    return longitudes, latitudes,time_init


def get_velocities_over_time(path_drift, u_drift, u_drift_decomposition, pos_init, time_init, n_steps):

    u_drift_true = np.zeros((n_steps+1,2))
    u_drift_pred = np.zeros((n_steps+1,2))
    u_water = np.zeros((n_steps+1,2))
    alpha_u_wind = np.zeros((n_steps+1,2))
    u_wind = np.zeros((n_steps+1,2))
    u_stockes = np.zeros((n_steps+1,2))

    ud_pred, water_u, alpha_wind_u, wind_u, wave_u = u_drift_decomposition(time_init, pos_init[0],pos_init[1])
    u_drift_pred[0,:] = ud_pred
    u_water[0,:] = water_u
    alpha_u_wind[0,:] = alpha_wind_u
    u_wind[0,:] = wind_u
    u_stockes[0,:] = wave_u

    lon_1h,lat_1h = get_closest_position(path_drift, 1)
    u_drift_true[0,0] = dist_longitudes(pos_init[0],lon_1h,pos_init[1],lat_1h)/3600
    u_drift_true[0,1] = dist_latitudes(pos_init[1],lat_1h)/3600

    for i in range(n_steps):
        pos_new, time_new = RK4_step(u_drift, pos_init, time_init, 1)

        if np.isnan(pos_new).any():
            print('Trajectory stopped after ', i, ' hours, Nan values')
            break


        ud_pred, water_u, alpha_wind_u, wind_u, wave_u = u_drift_decomposition(time_new, pos_new[0],pos_new[1])
        u_drift_pred[i+1,:] = ud_pred
        u_water[i+1,:] = water_u
        alpha_u_wind[i+1,:] = alpha_wind_u
        u_wind[i+1,:] = wind_u
        u_stockes[i+1,:] = wave_u

        lon_1h,lat_1h = get_closest_position(path_drift, i+1)
        u_drift_true[i+1,0] = dist_longitudes(pos_new[0],lon_1h,pos_new[1],lat_1h)/3600
        u_drift_true[i+1,1] = dist_latitudes(pos_new[1],lat_1h)/3600

        pos_init = pos_new
        time_init = time_new

    
    return u_drift_true,u_drift_pred,u_water,alpha_u_wind,u_wind, u_stockes