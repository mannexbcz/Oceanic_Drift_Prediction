import os
import pandas as pd
from utils.read_data import read_drift_positions

# This is the path where all the initial files are stored.
FOLDER_PATH = '../data/Dumont-etal_2019/datasets'
SAVING_PATH_24 = '../data/Dumont-etal_2019/dataset6h'
SAVING_PATH_UNDER_24 = '../data/Dumont-etal_2019/datasetunder6h'

def get_time_continuous_path(drift_tab):

    list_df_ok = []
    idx_start = 0
    for i in range(1,len(drift_tab)):
        hours_diff = drift_tab['hours'].iloc[i]-drift_tab['hours'].iloc[i-1]
        if hours_diff > 6:
            list_df_ok.append(drift_tab.iloc[idx_start:i])
            idx_start = i
           
    return list_df_ok

def get_length_trace(drift_tab):
    return drift_tab['hours'].iloc[-1]-drift_tab['hours'].iloc[0]

def get_24h_chunks(df):
    list_24h = []
    list_under_24h = []
    h = 6
    df['hours_since_start'] = df['hours']- df['hours'].iloc[0]
    nchunks = df['hours_since_start'].iloc[-1] // h
    for i in range(int(nchunks)):
        cuthour = (i+1)*h
        prev_cuthour = i*h
        df_chunk = df.loc[(df['hours_since_start'] >= prev_cuthour) & (df['hours_since_start'] < cuthour)]
        df_chunk.drop(['hours_since_start'], axis = 1, inplace = True)
        list_24h.append(df_chunk)
    last_cuthour = nchunks*h
    df_chunk = df.loc[df['hours_since_start'] >= last_cuthour]
    df_chunk.drop(['hours_since_start'], axis = 1, inplace = True)
    list_under_24h.append(df_chunk)
    return list_24h, list_under_24h


if __name__ == "__main__": 

    list24 = []
    listunder24 = []

    for data_file in os.listdir(FOLDER_PATH):
        # Open data file
        data_path = os.path.join(FOLDER_PATH, data_file)
        drift_tab = read_drift_positions(data_path)

        # Get time-continuous trajectories
        list_df_ok = get_time_continuous_path(drift_tab)

        for df in list_df_ok:
            l24h, lu24h = get_24h_chunks(df)
            list24 = list24 + l24h
            listunder24 = listunder24 + lu24h

    i = 0

    for df in list24:
        dataname = 'Trajectory_' + str(i) + '.csv'
        data_path = os.path.join(SAVING_PATH_24, dataname)
        df.to_csv(data_path)
        i = i+1
    
    for df in listunder24:
        dataname = 'Trajectory_' + str(i) + '.csv'
        if len(df) >1 and get_length_trace(df)>=6:
            data_path = os.path.join(SAVING_PATH_UNDER_24, dataname)
            df.to_csv(data_path)
            i = i+1



