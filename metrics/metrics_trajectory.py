from utils.read_data import get_closest_position
from haversine import haversine


'''def normalized_cumulative_Lagrangian_dist(path_drift, pred_longitudes,pred_latitudes):
    sum_ds = 0
    len_true_traj = 0
    sum_dl = 0
    for i in range(len(pred_latitudes)):
        if i == 0:
            true_lon_prec, true_lat_prec = get_closest_position(path_drift, i)
            true_pos_prec = (true_lat_prec, true_lon_prec)
            pass
        true_lon, true_lat = get_closest_position(path_drift, i)
        #true_lon_prec, true_lat_prec = get_closest_position(path_drift, i-1) #
        true_pos = (true_lat,true_lon)
        #true_pos_prec = (true_lat_prec, true_lon_prec)

        pred_pos = (pred_latitudes[i], pred_longitudes[i])
        sum_ds += haversine(true_pos,pred_pos) # distance in km 
        len_true_traj += haversine(true_pos, true_pos_prec)
        sum_dl += len_true_traj
        true_pos_prec = true_pos
    
    sc = sum_ds/sum_dl
    return sc'''

def normalized_cumulative_Lagrangian_dist(true_longitudes_extrapolated, true_latitiudes_extrapolated, pred_longitudes,pred_latitudes):
    sum_ds = 0
    len_true_traj = 0
    sum_dl = 0
    for i in range(len(pred_latitudes)):
        if i == 0:
            true_lon_prec, true_lat_prec = true_longitudes_extrapolated[i],true_latitiudes_extrapolated[i]
            true_pos_prec = (true_lat_prec, true_lon_prec)
            pass
        true_lon, true_lat = true_longitudes_extrapolated[i],true_latitiudes_extrapolated[i]
        true_pos = (true_lat,true_lon)

        pred_pos = (pred_latitudes[i], pred_longitudes[i])
        sum_ds += haversine(true_pos,pred_pos) # distance in km 

        len_true_traj += haversine(true_pos, true_pos_prec)
        sum_dl += len_true_traj
        true_pos_prec = true_pos
    
    sc = sum_ds/sum_dl
    return sc

def separation_after_N_hours(true_longitudes_extrapolated, true_latitiudes_extrapolated, pred_longitudes,pred_latitudes, nhours):
    true_position = (true_latitiudes_extrapolated[nhours], true_longitudes_extrapolated[nhours])
    pred_position = (pred_latitudes[nhours], pred_longitudes[nhours])
    return haversine(true_position,pred_position) #km


def skill_score(true_longitudes_extrapolated, true_latitiudes_extrapolated, pred_longitudes,pred_latitudes):
    sc = normalized_cumulative_Lagrangian_dist(true_longitudes_extrapolated, true_latitiudes_extrapolated, pred_longitudes,pred_latitudes)
    if sc <= 1:
        ssc = 1 - sc
    else:
        ssc = 0
    return ssc

def time_averaged_distance(true_longitudes_extrapolated, true_latitiudes_extrapolated, pred_longitudes,pred_latitudes):

    sum_dist = 0

    for i in range(1,len(pred_latitudes)):
        true_pos = (true_latitiudes_extrapolated[i],true_longitudes_extrapolated[i])
        pred_pos = (pred_latitudes[i], pred_longitudes[i])
        sum_dist += haversine(true_pos,pred_pos) # distance in km 

    return sum_dist/(len(pred_latitudes)-1)



