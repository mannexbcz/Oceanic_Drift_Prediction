from utils.convert_lats import *

def shrink_boundary(min_lat, max_lat, min_lon, max_lon, nkm):
    min_lat_boundary = new_latitude(min_lat, min_lon, nkm*1000)
    max_lat_boundary = new_latitude(max_lat, min_lon, -nkm*1000)
    min_lon_boundary = max(new_longitude(min_lat, min_lon, nkm*1000), new_longitude(max_lat, min_lon, nkm*1000))
    max_lon_boundary = min(new_longitude(min_lat, max_lon, nkm*1000), new_longitude(max_lat, max_lon, nkm*1000))

    return min_lat_boundary, max_lat_boundary, min_lon_boundary, max_lon_boundary

def is_inside_boundary(lat, lon, min_lat, max_lat, min_lon, max_lon):
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def find_closest_point(lat, lon, min_lat, max_lat, min_lon, max_lon):
    closest_lat = clamp(lat, min_lat, max_lat)
    closest_lon = clamp(lon, min_lon, max_lon)
    return closest_lat, closest_lon                  


def check_point_boundaries(lat, lon, min_lat, max_lat, min_lon, max_lon, nkm):
    min_lat_sh, max_lat_sh, min_lon_sh, max_lon_sh = shrink_boundary(min_lat, max_lat, min_lon, max_lon,nkm)

    if is_inside_boundary(lat, lon,min_lat_sh, max_lat_sh, min_lon_sh, max_lon_sh):
        return lat, lon
    else:
        newpoint = find_closest_point(lat, lon, min_lat_sh, max_lat_sh, min_lon_sh, max_lon_sh)
        return newpoint


