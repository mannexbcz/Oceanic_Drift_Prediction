import math

REARTH = 6378137 # radius of the earth in m 

# Computes new longitude and new latitudes based on a coordinate and a displacement in meters
# Assumes the displacemnt to be small compared to the radius of the earth 
# Assumes we are not too close to the poles
# from https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters


def new_latitude(latitude,longitude, dlat):
    return latitude + (dlat/REARTH)*(180/math.pi)

def new_longitude(latitude,longitude,dlon):
    return longitude + (dlon/REARTH)*(180/math.pi)/math.cos(latitude*math.pi/180)


def dist_latitudes(lat1,lat2):
    return (lat2-lat1)*(math.pi/180)*REARTH

def dist_longitudes(lon1,lon2,lat1,lat2):
    mean_lat = (lat1+lat2)/2
    return (lon2-lon1)*(math.pi/180)*REARTH*math.cos(mean_lat*math.pi/180)

