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
        return df
