#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats

import sys
sys.path.append('fluidity-master')
import vtktools

""" FILE NAME: 'vtuToCsv.py'
    DESCRIPTION: This file is loading and preparing the data from the raw fluidity
    files to panda dataframes that are then saved in csv files. Right now, the average
    value of every field over time is taken for each of the 100,080 positions.
    INPUT: LSBU_200/ - LSBU_988/
    OUTPUT: average_over_time.csv
"""

def getDF(pos, tracer, average=True):
    """ Function that combines the numpy arrays with the position and tracer values
        in a pandas dataframe that will later be saved as a csv file. If the parameter
        average is set to true, the average of the tracer value over all timestep is
        calculated.
        Inputs:
        - pos: 3D-Numpy Array having all positions.
        - tracer: array with all tracer values
        - average: boolean specifying if the average over all timesteps should be taken.
    """
    df = pd.DataFrame({'X': pos[:, 0],
                       'Y': pos[:, 1],
                       'Z': pos[:, 2],
                       'tracer': output})
    if average:
        for i in range(201, 537):
            last_index = i+1
            print('LSBU_'+str(i)+'.vtu')
            ug = vtktools.vtu('data/LSBU_32_RAW/LSBU_'+str(i)+'/LSBU_16.vtu')
            ug.GetFieldNames()
            pos = ug.GetLocations()

            tracer += ug.GetScalarField('TracerGeorge')

        tracer /= last_index

    return df

def remove_outlier(df_in, col_name):
    """ This funtion drops all outliers in a pandas dataframe according to the
        specified column with the IQR method.
        Inputs:
        - df_in: pandas dataframe that the outliers will be removed from.
        - col_name: name of the column that the IQR will be calculated on.
    """
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def removeUnderFiftyMeters(df):
    """ This function removes all tracer observation for which Z > 50m.
        Inputs:
        - df: dataframe that function should be applied at.
    """
    return df.query('Z <= 50.01')

def normalize(df, column):
    """ This function normalized the dataframe according to the specified column.
        Inputs:
        - df: dataframe that is to be normalized
        - column: name of the column that the normalization will be based on.
    """
    return (df[column]-df[column].min()) / (df[column].max()-df[column].min())

def standardize(df, column):
    """ This function standardize the dataframe according to the specified column.
        Inputs:
        - df: dataframe that is to be normalized
        - column: name of the column that the standardization will be based on.
    """
    return (df[column]-df[column].mean()) / df[column].std()


ug = vtktools.vtu('data/LSBU_32_RAW/LSBU_200/LSBU_16.vtu')
ug.GetFieldNames()
pos = ug.GetLocations()
tracer = ug.GetScalarField('TracerGeorge')
df = getDF(pos, tracer)
df = remove_outlier(df, 'tracer')
df['tracer'] = normalize(df, 'tracer')

""" Saving the dataframe in a csv file """
print('Saving...')
print('DF Shape: ', df.shape)
df.to_csv('data/csv_data/normalized/LSBU_32/average_over_time_16.csv', index=False)
