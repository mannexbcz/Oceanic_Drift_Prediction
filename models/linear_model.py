
import numpy as np

def u_drift_linear(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation):
    alpha1,alpha2 = alpha

    def u_drift(time,lon,lat):

        ud_u = water_u_interpolation([time,0,lat,lon]) + alpha1 * u10_interpolation([time,lat,lon])
        ud_v = water_v_interpolation([time,0,lat,lon]) + alpha2 * v10_interpolation([time,lat,lon])
        return np.array([ud_u.item(),ud_v.item()])
    
    return u_drift


def u_drift_linear_matrix(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)
        wind_u = np.zeros(2)

        water_u[0] = water_u_interpolation([time,0,lat,lon])
        water_u[1] = water_v_interpolation([time,0,lat,lon])
        wind_u[0] = u10_interpolation([time,lat,lon])
        wind_u[1] = v10_interpolation([time,lat,lon])

        ud = water_u + alpha@wind_u

        return ud
    
    return u_drift



def u_drift_linear_complex(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation):

    def u_drift(time,lon,lat):

        water_u_complex = complex(water_u_interpolation([time,0,lat,lon]),-water_v_interpolation([time,0,lat,lon]))
        wind_u_complex = complex(u10_interpolation([time,lat,lon]),-v10_interpolation([time,lat,lon]))
        ud_complex = water_u_complex + alpha*wind_u_complex
        return np.array([ud_complex.real,ud_complex.imag])
    
    return u_drift


def u_drift_linear_matrix_randomized(alpha, u10_interpolation, v10_interpolation,water_u_interpolation,water_v_interpolation, DUWATER, DUWIND):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)
        wind_u = np.zeros(2)

        dwater_u = np.random.normal(loc=0.0, scale=DUWATER/2)
        dwater_v = np.random.normal(loc=0.0, scale=DUWATER/2)
        dwind_u = np.random.normal(loc=0.0, scale=DUWIND/2)
        dwind_v = np.random.normal(loc=0.0, scale=DUWIND/2)

        water_u[0] = water_u_interpolation([time,0,lat,lon]) + dwater_u
        water_u[1] = water_v_interpolation([time,0,lat,lon]) + dwater_v
        wind_u[0] = u10_interpolation([time,lat,lon]) + dwind_u
        wind_u[1] = v10_interpolation([time,lat,lon]) + dwind_v

        ud = water_u + alpha@wind_u

        return ud
    
    return u_drift


#======================= Components alone ===================================

def u_drift_ocean_only(water_u_interpolation,water_v_interpolation):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        water_u = np.zeros(2)

        water_u[0] = water_u_interpolation([time,0,lat,lon])
        water_u[1] = water_v_interpolation([time,0,lat,lon])

        ud = water_u

        return ud
    
    return u_drift

def u_drift_wind_only(alpha, u10_interpolation, v10_interpolation):
    #now alpha is a 2x2 matrix

    def u_drift(time,lon,lat):
        wind_u = np.zeros(2)

        wind_u[0] = u10_interpolation([time,lat,lon])
        wind_u[1] = v10_interpolation([time,lat,lon])

        ud = alpha@wind_u

        return ud
    
    return u_drift