#!/usr/bin/python
import numpy as np
import pandas as pd

import sys
sys.path.append('fluidity-master')
import vtktools

################################################################################
#        FILE NAME: "get_csv.py"                                               #
#        DESCRIPTION: This file is loading and preparing the data from the raw #
#        fluidity files to panda dataframes that are then saved in csv files.  #
#        INPUT: LSBU_0.vtu - LSBU_988.vtu                                      #
#        OUTPUT: data1.csv, data2.csv, data3.csv                               #
################################################################################

ug = vtktools.vtu('raw_data/LSBU_0.vtu')
ug.GetFieldNames()
pos = ug.GetLocations()

X = pos[:, 0]
Y = pos[:, 1]
Z = pos[:, 2]
time = ug.GetScalarField('Time')
tracer = ug.GetScalarField('Tracer')

for i in range(0, 988):
    print('LSBU_'+str(i)+'.vtu')
    ug = vtktools.vtu('raw_data/LSBU_'+str(i)+'.vtu')
    ug.GetFieldNames()
    pos = ug.GetLocations()

    X = np.hstack((X, pos[:, 0]))
    Y = np.hstack((Y, pos[:, 1]))
    Z = np.hstack((Z, pos[:, 2]))
    time = np.hstack((time, ug.GetScalarField('Time')))
    tracer = np.hstack((tracer, ug.GetScalarField('Tracer')))

######################## TRUNCATED SINGULAR VALUE DECOMPOSITION ################

# svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
#
# svd.fit(principalComponents)
# print(principalComponents)

######################### COMBINE AND SAVE DATA ################################

df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'time': time, 'tracer': tracer})
print(df)

df.to_csv('sample.csv', index=False)
