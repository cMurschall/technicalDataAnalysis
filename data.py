# Built-in/Generic Imports (os, sys, ...)
import os
import sys

# Common Libs (numpy, pandas, ...)
import numpy as np
import pandas as pd
import h5py

file = "./../data/data_geo_2020012414325300287_20230210_115111.h5"


def count_journeys():
    with h5py.File(file, 'r') as h:
        return len(h.keys())


def get_sensors():
    return ['ADC1', 'ADC2', 'IMU']


def read_data(jrn, source):
    """
    Parameters
    ----------
    file : str
        Name of h5 file to be read.
    jrn : int
        Journey number.
    source : str
        Data source whose data should be read. Either AD converter name of axle-
        box acceleration sensor ('ADC1', 'ADC2') or 'IMU'.

    Returns
    -------
    pandas.DataFrame
        DF containing the data of the journey and sensor.

    """

    jrn_gr_name = 'journey_' + str(jrn).zfill(2)

    with h5py.File(file, 'r') as h:
        journey_grs = list(h.keys())
        if jrn_gr_name not in journey_grs:
            print('No journey ' + str(jrn) + ' in input file. Available '
                  + 'journeys: \n')
            print(journey_grs)
            return pd.DataFrame()
        jrn_gr = h[jrn_gr_name]
        sensor_grs = list(jrn_gr.keys())
        if source not in sensor_grs:
            print('This sensor is not available. Available sources:\n')
            print(sensor_grs)
            return pd.DataFrame()
        sensor_data = jrn_gr[source]['data_' + source]
        df = pd.DataFrame()
        for col in sensor_data.dtype.names:
            df[col] = sensor_data[col]

        if source in ('ADC1', 'ADC2'):
            # timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # integrate speed
            df["distance"] = (np.abs(df["speed"]) * df['time'].diff().dt.total_seconds()).cumsum().fillna(0)

            # integrate speed
            df["ch0_local_velocity"] = (np.abs(df["ch0"]) * df['time'].diff().dt.total_seconds()).fillna(0)
            # integrate distance
            df["ch0_local_distance"] = (np.abs(df["ch0_local_velocity"]) * df['time'].diff().dt.total_seconds()).fillna(0)
        return df


def read_as_chuncked(source, chunk_length=0.3):
    file_name = f"{source}_{chunk_length}.h5"
    if os.path.exists(file_name):
        print(f"Reading from {file_name}")
        df = pd.read_pickle(file_name)
        return df

    print(f"Creating {file_name}")
    data = []
    # count_journeys()
    for j in range(3):
        print(f"Journey {j}")
        df_journey = read_data(j, source)

        # remove zero speed
        df_journey = df_journey[np.abs(df_journey['speed']) > 0.05]

        df_journey["Journey"] = j
        df_journey["Bin"] = 0

        bins = np.arange(0, df_journey['distance'].max(), chunk_length)

        for i in range(0, len(bins) - 1):
            # get chunk
            df_journey.loc[((df_journey['distance'] < bins[i + 1]) & (df_journey['distance'] >= bins[i]), "Bin")] = i

        data.append(df_journey)

    df = pd.concat(data)

    pd.to_pickle(df, file_name)

    return df