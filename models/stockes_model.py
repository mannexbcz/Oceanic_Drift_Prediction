import numpy as np

def u_drift_ocean_wind_stockes_matrix(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation,ust_interpolation,vst_interpolation):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)
        wind_u = np.zeros(2)
        wave_u = np.zeros(2)

        water_u[0] = water_u_interpolation([time,0,lat,lon])
        water_u[1] = water_v_interpolation([time,0,lat,lon])
        wind_u[0] = u10_interpolation([time,lat,lon])
        wind_u[1] = v10_interpolation([time,lat,lon])
        wave_u[0] = ust_interpolation([time,lat,lon])
        wave_u[1] = vst_interpolation([time,lat,lon])

        ud = water_u + alpha@wind_u + wave_u

        return ud
    
    return u_drift

def u_drift_stockes_only(ust_interpolation,vst_interpolation):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        wave_u = np.zeros(2)

        wave_u[0] = ust_interpolation([time,lat,lon])
        wave_u[1] = vst_interpolation([time,lat,lon])

        ud = wave_u

        return ud
    
    return u_drift


def u_drift_ocean_stockes_matrix(water_u_interpolation,water_v_interpolation,ust_interpolation,vst_interpolation):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)
        wave_u = np.zeros(2)

        water_u[0] = water_u_interpolation([time,0,lat,lon])
        water_u[1] = water_v_interpolation([time,0,lat,lon])
        wave_u[0] = ust_interpolation([time,lat,lon])
        wave_u[1] = vst_interpolation([time,lat,lon])

        ud = water_u + wave_u

        return ud
    
    return u_drift


def get_udrift_decomposition(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation,ust_interpolation,vst_interpolation):

    def udrift_decomposition(time, lon,lat):

        water_u = np.zeros(2)
        wind_u = np.zeros(2)
        wave_u = np.zeros(2)

        water_u[0] = water_u_interpolation([time,0,lat,lon])
        water_u[1] = water_v_interpolation([time,0,lat,lon])
        wind_u[0] = u10_interpolation([time,lat,lon])
        wind_u[1] = v10_interpolation([time,lat,lon])
        wave_u[0] = ust_interpolation([time,lat,lon])
        wave_u[1] = vst_interpolation([time,lat,lon])

        ud = water_u + alpha@wind_u + wave_u

        return ud, water_u, alpha@wind_u, wind_u, wave_u
    
    return udrift_decomposition

############## Randomized perturbations ##################################################

def u_drift_ocean_stockes_matrix_randomized(water_u_interpolation, water_v_interpolation,ust_interpolation,vst_interpolation, DUWATER, DUWAVES):

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)
        wave_u = np.zeros(2)

        dwater_u = np.random.normal(loc=0.0, scale=DUWATER/2)
        dwater_v = np.random.normal(loc=0.0, scale=DUWATER/2)
        dwaves_u = np.random.normal(loc=0.0, scale=DUWAVES/2)
        dwaves_v = np.random.normal(loc=0.0, scale=DUWAVES/2)

        water_u[0] = water_u_interpolation([time,0,lat,lon]) + dwater_u
        water_u[1] = water_v_interpolation([time,0,lat,lon]) + dwater_v
        wave_u[0] = ust_interpolation([time,lat,lon]) + dwaves_u
        wave_u[1] = vst_interpolation([time,lat,lon]) + dwaves_v

        ud = water_u + wave_u

        return ud
    
    return u_drift

def u_drift_ocean_wind_stockes_matrix_randomized(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation,ust_interpolation,vst_interpolation,DUWATER, DUWIND,DUWAVES):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)
        wind_u = np.zeros(2)
        wave_u = np.zeros(2)

        dwater_u = np.random.normal(loc=0.0, scale=DUWATER/2)
        dwater_v = np.random.normal(loc=0.0, scale=DUWATER/2)
        dwind_u = np.random.normal(loc=0.0, scale=DUWIND/2)
        dwind_v = np.random.normal(loc=0.0, scale=DUWIND/2)
        dwaves_u = np.random.normal(loc=0.0, scale=DUWAVES/2)
        dwaves_v = np.random.normal(loc=0.0, scale=DUWAVES/2)

        water_u[0] = water_u_interpolation([time,0,lat,lon]) + dwater_u 
        water_u[1] = water_v_interpolation([time,0,lat,lon]) + dwater_v
        wind_u[0] = u10_interpolation([time,lat,lon]) + dwind_u
        wind_u[1] = v10_interpolation([time,lat,lon]) + dwind_v
        wave_u[0] = ust_interpolation([time,lat,lon]) + dwaves_u
        wave_u[1] = vst_interpolation([time,lat,lon]) + dwaves_v

        ud = water_u + alpha@wind_u + wave_u

        return ud
    
    return u_drift