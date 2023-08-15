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
            df["ch0_local_distance"] = (np.abs(df["ch0_local_velocity"]) * df['time'].diff().dt.total_seconds()).fillna(
                0)
        return df


def read_as_chuncks(source, chunk_length=1):
    file_name = f"chunks_{source}_{chunk_length}.pkl"

    # if 'chunks.pkl' file exists, read it
    if os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
        return df

    chunks = []
    #  )
    for j in range(0, count_journeys()):
        print(f"Journey {j}")
        df = read_data(j, source)

        bins = np.arange(0, df['distance'].max(), chunk_length)
        chunks = []

        for i in range(0, len(bins) - 1):
            # get chunk
            chunk = df[(df['distance'] < bins[i + 1]) & (df['distance'] >= bins[i])].copy()
            chunk['Journey'] = j
            chunk['Bin'] = i
            chunks.append(chunk)

    df = pd.concat(chunks)
    # store
    pd.to_pickle(df, file_name)
    return df
