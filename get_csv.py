#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats

import sys
sys.path.append('fluidity-master')
import vtktools

################################################################################
#        FILE NAME: 'get_csv.py'                                               #
#        DESCRIPTION: This file is loading and preparing the data from the raw #
#        fluidity files to panda dataframes that are then saved in csv files.  #
#        Right now, the average value of every field over time is taken for    #
#        each of the 100,080 positions.                                        #
#        INPUT: LSBU_200/ - LSBU_988/                                          #
#        OUTPUT: average_over_time.csv                                         #
################################################################################

ug = vtktools.vtu('data/LSBU_raw/LSBU_200/LSBU_200_7.vtu')
ug.GetFieldNames()
pos = ug.GetLocations()
tracer = ug.GetScalarField('TracerGeorge')

##################### FUNCTION TO COMBINE DATA IN DATAFRAME ####################

def getDF(pos, tracer_george):
    df = pd.DataFrame({'X': pos[:, 0],
                       'Y': pos[:, 1],
                       'Z': pos[:, 2],
                       'tracer': tracer})
    return df

######################## GET AVERAGE OVER TIMESTEPS ############################

for i in range(201, 507):
    last_index = i+1
    print('LSBU_'+str(i)+'.vtu')
    ug = vtktools.vtu('data/LSBU_raw/LSBU_'+str(i)+'/LSBU_'+str(i)+'_7.vtu')
    ug.GetFieldNames()
    pos = ug.GetLocations()

    tracer += ug.GetScalarField('TracerGeorge')

tracer /= last_index

df = getDF(pos, tracer)

######################## DROP OUTLIERS WITH IQR METHOD #########################

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = remove_outlier(df, 'tracer')

################ NORMALIZE/STANDARDIZE POTENTIAL OUTPUT FIELDS #################

def normalize(df, column):
    return (df[column]-df[column].min()) / (df[column].max()-df[column].min())

# def standardize(df, column):
#     return (df[column]-df[column].mean()) / df[column].std()

# df['tracer'] = standardize(df, 'tracer')
df['tracer'] = normalize(df, 'tracer')


######################### COMBINE AND SAVE DATA ################################

print('Saving...')
df.to_csv('data/csv_data/normalized/tracer_george/average_over_time_7.csv', index=False)
