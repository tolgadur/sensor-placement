#!/usr/bin/python
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from scipy import stats

import sys
sys.path.append('fluidity-master')
import vtktools

""" FILE NAME: 'data_preperation.py'
    DESCRIPTION: This file is loading and preparing the data from the raw fluidity
    files to panda dataframes that are then saved in csv files.
"""

def positionsCSV(pos):
    """ This function saves a CSV file with all positions.
        Input:
        - pos: 3D-Numpy Array having all positions.
    """
    df = pd.DataFrame({'X': pos[:, 0],
                       'Y': pos[:, 1],
                       'Z': pos[:, 2]})

    print('Saving position CSV-File')
    print('DF Shape: ', df.shape)
    df.to_csv('data/csv_data/positions.csv', index=False)

def tracerMatrixCSV(tracer, j):
    """ This function saves a CSV file with tracer values accross all timesteps.
        Input:
        - tracer:
    """
    df = pd.DataFrame({'t1': tracer})

    for i in range(1, 537):
        print('LSBU_'+str(i))
        ug = vtktools.vtu('../LSBU32/LSBU_'+str(i)+'_'+str(j)+'.vtu')
        ug.GetFieldNames()

        tracer = ug.GetScalarField('TracerFront')
        tracer += ug.GetScalarField('TracerGeorge')
        df['t'+str(i)] = tracer

    print('Saving Tracer CSV-File')
    print('DF Shape: ', df.shape)
    df.to_csv('data/csv_data/tracer.csv', index=False)

def removeOutlier(df_in, col_name):
    """ This funtion drops all outliers in a pandas dataframe according to the
        specified column with the IQR method.
        Input:
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
        Input:
        - df: dataframe that function should be applied at.
    """
    # indices = df.index[df['Z'] > 50.01].tolist()
    return df.query('Z <= 50.01')

def normalize(df, column):
    """ This function normalized the dataframe according to the specified column.
        Input:
        - df: dataframe that is to be normalized
        - column: name of the column that the normalization will be based on.
    """
    return (df[column]-df[column].min()) / (df[column].max()-df[column].min())

def standardize(df, column):
    """ This function standardize the dataframe according to the specified column.
        Input:
        - df: dataframe that is to be normalized
        - column: name of the column that the standardization will be based on.
    """
    return (df[column]-df[column].mean()) / df[column].std()

ug = vtktools.vtu('../LSBU32/LSBU_0_'+str(sys.argv[1])+'.vtu')
ug.GetFieldNames()

# pos = ug.GetLocations()
tracer = ug.GetScalarField('TracerFront')
tracer += ug.GetScalarField('TracerBlackfriars')

# positionAndTracerCSV(pos, tracer)
# positionsCSV(pos)
tracerMatrixCSV(tracer, sys.argv[1])
