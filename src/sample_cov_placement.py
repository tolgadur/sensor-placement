#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn import decomposition
from sensor_placement import SensorPlacement

""" FILE NAME: 'sample_cov_placement.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
    This implementation uses a sample calculated that is calculated after applying PCA
    on the timesteps. 95% of the variance is hereby preserved.
"""

""" Data is loaded and prepared for the placement algorithm. """
V_df = pd.read_csv('data/csv_data/positions.csv')
V = V_df[['X', 'Y', 'Z']].copy().values
V = V[:100]
index_sets = SensorPlacement.positionIndices(V)

tracer_df = pd.read_csv('data/csv_data/tracerOverTime.csv')
tracer = tracer_df.values
tracer = tracer[:100]

""" The covariance matrix needed for the placement algorithm is calculated. """
pca = decomposition.PCA(0.95)
tracer = pca.fit_transform(tracer)

cov = np.cov(tracer)

""" Placement algorithm is called and the results are printed in the terminal. """
# A = SensorPlacement.naiveSensorPlacement(cov, 5, V, index_sets)
A = SensorPlacement.lazySensorPlacement(cov, 5, V, index_sets)
print(A)
