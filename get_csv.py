#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn import preprocessing

import sys
sys.path.append('fluidity-master')
import vtktools

################################################################################
#        FILE NAME: 'get_csv.py'                                               #
#        DESCRIPTION: This file is loading and preparing the data from the raw #
#        fluidity files to panda dataframes that are then saved in csv files.  #
#        Right now, the average value of every field over time is taken for    #
#        each of the 100,080 positions.
#        INPUT: LSBU_0.vtu - LSBU_988.vtu                                      #
#        OUTPUT: average_over_time.csv                                         #
################################################################################

ug = vtktools.vtu('data/raw_data/LSBU_200.vtu')
ug.GetFieldNames()
pos = ug.GetLocations()
tracer = ug.GetScalarField('Tracer')
tracer_background = ug.GetScalarField('TracerBackground')
pressure = ug.GetScalarField('Pressure')
velocity = ug.GetVectorNorm('Velocity')

##################### FUNCTION TO COMBINE DATA IN DATAFRAME ####################

def getDF(pos, velocity, pressure, tracer_background, tracer):
    df = pd.DataFrame({'X': pos[:, 0],
                       'Y': pos[:, 1],
                       'Z': pos[:, 2],
                       'velocity_norm': velocity,
                       'pressure': pressure,
                       'tracer_background': tracer_background,
                       'tracer': tracer})
    return df

######################## GET AVERAGE OVER TIMESTEPS ############################

for i in range(201, 989):
    last_index = i+1
    print('LSBU_'+str(i)+'.vtu')
    ug = vtktools.vtu('data/raw_data/LSBU_'+str(i)+'.vtu')
    ug.GetFieldNames()
    pos = ug.GetLocations()

    tracer += ug.GetScalarField('Tracer')
    tracer_background += ug.GetScalarField('TracerBackground')
    pressure += ug.GetScalarField('Pressure')
    velocity += ug.GetVectorNorm('Velocity')

tracer /= last_index
tracer_background /= last_index
pressure /= last_index
# velocity /= last_index

df = getDF(pos, velocity, pressure, tracer_background, tracer)

################ NORMALIZE/STANDARDIZE POTENTIAL OUTPUT FIELDS #################

def normalize(df, column):
    return (df[column]-df[column].min()) / (df[column].max()-df[column].min())

def standardize(df, column):
    print(df[column].mean())
    return (df[column]-df[column].mean()) / df[column].std()

df['tracer'] = standardize(df, 'tracer')
df['tracer_background'] = standardize(df, 'tracer_background')
df['pressure'] = standardize(df, 'pressure')
df['velocity_norm'] = standardize(df, 'velocity_norm')

######################### COMBINE AND SAVE DATA ################################

print('Saving...')
df.to_csv('timestep_500.csv', index=False)
