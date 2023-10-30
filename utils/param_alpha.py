from utils.read_data import get_true_drift_positions
from utils.complex_lin_regression import complex_linear_regression
from utils.convert_lats import dist_latitudes, dist_longitudes
import numpy as np
from sklearn.linear_model import LinearRegression
import math

#################### Non personalized alpha ##########################################################3

def general_alpha(alpha = 0.02, theta = 0.349066):
    
    alpha_mat = np.zeros((2,2))

    alpha_mat[0,0] = alpha * math.cos(theta)
    alpha_mat[1,0] = -alpha * math.sin(theta)
    alpha_mat[0,1] = alpha * math.sin(theta)
    alpha_mat[1,1] = alpha * math.cos(theta)

    return alpha_mat


########################################################################################################

def get_alpha(lats,lons,times, water_u, water_v, wind_u, wind_v):
    lat1 = lats[0]
    lat2 = lats[1]
    lat3 = lats[2]
    lon1 = lons[0]
    lon2 = lons[1]
    lon3 = lons[2]

    dt = (times[2]-times[0])*3600 #original times are in h but velocities are in m/s

    water_u_init = water_u([times[1],0,lat2,lon2])
    water_v_init = water_v([times[1],0,lat2,lon2])

    wind_u_init = wind_u([times[1],lat2,lon2])
    wind_v_init = wind_v([times[1],lat2,lon2])

    ud_u = dist_longitudes(lon1,lon3,lat1,lat3)/dt #haversine function to get distance
    ud_v = dist_latitudes(lat1,lat3)/dt

    print('Dist Long',dist_longitudes(lon1,lon2,lat1,lat2))

    alpha1 = (ud_u-water_u_init)/wind_u_init
    alpha2 = (ud_v-water_v_init)/wind_v_init

    '''alpha1 = 0
    alpha2 = 0'''

    print('Standard regression')
    print(alpha1)
    print(alpha2)

    #return [alpha1,alpha2]
    return [alpha1.item(),alpha2.item()]


def get_alpha_averaged(lats,lons,times, water_u, water_v, wind_u, wind_v, npoints):

    alpha1_sum = 0
    alpha2_sum = 0

    for i in range(npoints):

        lat1 = lats[i]
        lat2 = lats[i+1]
        lat3 = lats[i+2]
        lon1 = lons[i]
        lon2 = lons[i+1]
        lon3 = lons[i+2]

        dt = (times[i+2]-times[i])*3600 #original times are in h but velocities are in m/s

        water_u_init = water_u([times[i+1],0,lat2,lon2])
        water_v_init = water_v([times[i+1],0,lat2,lon2])

        wind_u_init = wind_u([times[i+1],lat2,lon2])
        wind_v_init = wind_v([times[i+1],lat2,lon2])

        ud_u = dist_longitudes(lon1,lon3,lat1,lat3)/dt #haversine function to get distance
        ud_v = dist_latitudes(lat1,lat3)/dt

        alpha1 = (ud_u-water_u_init)/wind_u_init
        alpha2 = (ud_v-water_v_init)/wind_v_init

        alpha1_sum = alpha1_sum + alpha1
        alpha2_sum = alpha2_sum + alpha2
    
    alpha1 = alpha1_sum/npoints
    alpha2 = alpha2_sum/npoints

    print('Standard regression')
    print(alpha1)
    print(alpha2)

    #return [alpha1,alpha2]
    return [alpha1.item(),alpha2.item()]



def compute_alpha_matrix(path, water_u, water_v, wind_u, wind_v):
    true_lon, true_lat, true_time = get_true_drift_positions(path)

    u_diff = np.zeros((2,2))
    u_A = np.zeros((2,2))

    for i in range(2):
        ud_u = dist_longitudes(true_lon[i+1],true_lon[i],true_lat[i+1],true_lat[i])/((true_time[i+1]-true_time[i])*3600) #haversine function to get distance
        ud_v = dist_latitudes(true_lat[i+1],true_lat[i])/((true_time[i+1]-true_time[i])*3600)

        water_u_init = water_u([true_time[i],0,true_lat[i],true_lon[i]])
        water_v_init = water_v([true_time[i],0,true_lat[i],true_lon[i]])

        wind_u_init = wind_u([true_time[i],true_lat[i],true_lon[i]])
        wind_v_init = wind_v([true_time[i],true_lat[i],true_lon[i]])

        u_diff[0,i] = ud_u - water_u_init
        u_diff[1,i] = ud_v - water_v_init
        u_A[0,i] = wind_u_init
        u_A[1,i] = wind_v_init

    print('U_A', u_A)

    alpha = u_diff@np.linalg.inv(u_A)

    return alpha

#===========================================================================================================================================
def update_alpha_GD(path, water_u, water_v, wind_u, wind_v, alpha = 0.02, theta = 0.349066, step=0.1, npoints=2):
    true_lon, true_lat, true_time = get_true_drift_positions(path)

    #alpha = 0.02
    #theta = 0.349066 # in rad, correspond to ~20°

    step_alpha = step*alpha
    step_theta = step*theta #this way, step correspond to a percentage of change, change is similar for both variables

    alpha_mat = np.zeros((2,2))

    for i in range(npoints-1):

        ud_u = dist_longitudes(true_lon[i],true_lon[i+1],true_lat[i],true_lat[i+1])/((true_time[i+1]-true_time[i])*3600) 
        ud_v = dist_latitudes(true_lat[i],true_lat[i+1])/((true_time[i+1]-true_time[i])*3600)

        water_u_init = water_u([true_time[i],0,true_lat[i],true_lon[i]])
        water_v_init = water_v([true_time[i],0,true_lat[i],true_lon[i]])

        wind_u_init = wind_u([true_time[i],true_lat[i],true_lon[i]])
        wind_v_init = wind_v([true_time[i],true_lat[i],true_lon[i]])

        u_pred = water_u_init + alpha*(math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)
        v_pred = water_v_init + alpha*(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)

        #print('Gradf_alpha:', ((math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)*(u_pred-ud_u)+(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(v_pred-ud_v)))
        #print('Gradf_theta:', alpha * ((-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(u_pred-ud_u)+(-math.cos(theta)*wind_u_init - math.sin(theta)*wind_v_init)*(v_pred-ud_v)))
        '''print('u_pred:', u_pred)
        print('u_true:', ud_u)
        print('v_pred:', v_pred)
        print('v_true:', ud_v)'''

        #print('udiff:', (u_pred-ud_u))
        #print('vdiff:', (v_pred-ud_v))

        alpha_new = alpha - step_alpha *((math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)*(u_pred-ud_u)+(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(v_pred-ud_v))
        theta_new = theta - step_theta * alpha * ((-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(u_pred-ud_u)+(-math.cos(theta)*wind_u_init - math.sin(theta)*wind_v_init)*(v_pred-ud_v))

        alpha = alpha_new
        theta = theta_new
        #print('New alpha:', alpha)
        #print('New theta:', math.degrees(theta))

    alpha_mat[0,0] = alpha * math.cos(theta)
    alpha_mat[1,0] = -alpha * math.sin(theta)
    alpha_mat[0,1] = alpha * math.sin(theta)
    alpha_mat[1,1] = alpha * math.cos(theta)

    return alpha_mat

def update_alpha_GD_stockes(path, water_u, water_v, wind_u, wind_v,wave_u,wave_v, alpha = 0.02, theta = 0.349066, step=0.1, npoints=2):
    true_lon, true_lat, true_time = get_true_drift_positions(path)

    step_alpha = step*alpha
    step_theta = step*theta #this way, step correspond to a percentage of change, change is similar for both variables

    alpha_mat = np.zeros((2,2))

    for i in range(npoints-1):

        ud_u = dist_longitudes(true_lon[i],true_lon[i+1],true_lat[i],true_lat[i+1])/((true_time[i+1]-true_time[i])*3600) 
        ud_v = dist_latitudes(true_lat[i],true_lat[i+1])/((true_time[i+1]-true_time[i])*3600)

        water_u_init = water_u([true_time[i],0,true_lat[i],true_lon[i]])
        water_v_init = water_v([true_time[i],0,true_lat[i],true_lon[i]])

        wind_u_init = wind_u([true_time[i],true_lat[i],true_lon[i]])
        wind_v_init = wind_v([true_time[i],true_lat[i],true_lon[i]])

        wave_u_init = wave_u([true_time[i],true_lat[i],true_lon[i]])
        wave_v_init = wave_v([true_time[i],true_lat[i],true_lon[i]])

        u_pred = water_u_init + wave_u_init + alpha*(math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)
        v_pred = water_v_init + wave_v_init + alpha*(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)

        alpha_new = alpha - step_alpha *((math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)*(u_pred-ud_u)+(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(v_pred-ud_v))
        theta_new = theta - step_theta * alpha * ((-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(u_pred-ud_u)+(-math.cos(theta)*wind_u_init - math.sin(theta)*wind_v_init)*(v_pred-ud_v))

        alpha = alpha_new
        theta = theta_new

    alpha_mat[0,0] = alpha * math.cos(theta)
    alpha_mat[1,0] = -alpha * math.sin(theta)
    alpha_mat[0,1] = alpha * math.sin(theta)
    alpha_mat[1,1] = alpha * math.cos(theta)

    return alpha_mat


def update_alpha_GD_wind_only(path, wind_u, wind_v, alpha = 0.02, theta = 0.349066, step=0.1, npoints=2):
    true_lon, true_lat, true_time = get_true_drift_positions(path)

    step_alpha = step*alpha
    step_theta = step*theta #this way, step correspond to a percentage of change, change is similar for both variables

    alpha_mat = np.zeros((2,2))

    for i in range(npoints-1):

        ud_u = dist_longitudes(true_lon[i],true_lon[i+1],true_lat[i],true_lat[i+1])/((true_time[i+1]-true_time[i])*3600) 
        ud_v = dist_latitudes(true_lat[i],true_lat[i+1])/((true_time[i+1]-true_time[i])*3600)

        wind_u_init = wind_u([true_time[i],true_lat[i],true_lon[i]])
        wind_v_init = wind_v([true_time[i],true_lat[i],true_lon[i]])

        u_pred = alpha*(math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)
        v_pred = alpha*(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)

        alpha_new = alpha - step_alpha *((math.cos(theta)*wind_u_init + math.sin(theta)*wind_v_init)*(u_pred-ud_u)+(-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(v_pred-ud_v))
        theta_new = theta - step_theta * alpha * ((-math.sin(theta)*wind_u_init + math.cos(theta)*wind_v_init)*(u_pred-ud_u)+(-math.cos(theta)*wind_u_init - math.sin(theta)*wind_v_init)*(v_pred-ud_v))

        alpha = alpha_new
        theta = theta_new

    alpha_mat[0,0] = alpha * math.cos(theta)
    alpha_mat[1,0] = -alpha * math.sin(theta)
    alpha_mat[0,1] = alpha * math.sin(theta)
    alpha_mat[1,1] = alpha * math.cos(theta)

    return alpha_mat


#============================================================================================================

def compute_alpha_whole_trajectory(path, water_u, water_v, wind_u, wind_v):
    true_lon, true_lat, true_time = get_true_drift_positions(path)

    u_diff = np.zeros((len(true_lon)-2,2))
    u_A = np.zeros((len(true_lon)-2,2))

    for i in range(len(true_lon)-2):
        ud_u = dist_longitudes(true_lon[i+2],true_lon[i],true_lat[i+2],true_lat[i])/((true_time[i+2]-true_time[i])*3600) #haversine function to get distance
        ud_v = dist_latitudes(true_lat[i+2],true_lat[i])/((true_time[i+2]-true_time[i])*3600)

        water_u_init = water_u([true_time[i+1],0,true_lat[i+1],true_lon[i+1]])
        water_v_init = water_v([true_time[i+1],0,true_lat[i+1],true_lon[i+1]])

        wind_u_init = wind_u([true_time[i+1],true_lat[i+1],true_lon[i+1]])
        wind_v_init = wind_v([true_time[i+1],true_lat[i+1],true_lon[i+1]])

        u_diff[i,0] = ud_u - water_u_init
        u_diff[i,1] = ud_v - water_v_init
        u_A[i,0] = wind_u_init
        u_A[i,1] = wind_v_init

    u_diff_wo_nan = u_diff[np.logical_and(~np.isnan(u_diff).any(axis=1),~np.isnan(u_A).any(axis=1))]
    u_A_wo_nan = u_A[np.logical_and(~np.isnan(u_diff).any(axis=1),~np.isnan(u_A).any(axis=1))]

    reg1 = LinearRegression().fit(u_A_wo_nan,u_diff_wo_nan)
    #reg2 = LinearRegression().fit(u_A_v_wo_nans.reshape(-1, 1),u_diff_v_wo_nans)

    print(reg1.coef_)
    #print(reg2.coef_)

    return reg1.coef_







def get_complex_alpha(lats,lons,times, water_u, water_v, wind_u, wind_v):
    lat1 = lats[0]
    lat2 = lats[1]
    lat3 = lats[2]
    lon1 = lons[0]
    lon2 = lons[1]
    lon3 = lons[2]

    dt = (times[2]-times[0])*3600

    water_u_init = water_u([times[1],0,lat2,lon2])
    water_v_init = water_v([times[1],0,lat2,lon2])

    wind_u_init = wind_u([times[1],lat2,lon2])
    wind_v_init = wind_v([times[1],lat2,lon2])

    '''print(water_u_init)
    print(water_v_init)
    print(wind_u_init)
    print(wind_v_init)'''

    ud_u = dist_longitudes(lon3,lon1,lat1,lat3)/dt #haversine function to get distance
    ud_v = dist_latitudes(lat3,lat1)/dt

    #ud_u = (lon3-lon1)/dt
    #ud_v = (lat3-lat1)/dt

    #Complex regression
    complex_water_u = complex(water_u_init,-water_v_init)
    complex_wind_u = complex(wind_u_init,-wind_v_init)
    complex_ud = complex(ud_u,-ud_v)

    complex_alpha = (complex_ud-complex_water_u)/complex_wind_u

    print('Complex regression')
    print(complex_alpha)

    return complex_alpha