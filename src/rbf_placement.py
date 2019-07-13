#!/usr/bin/python
import numpy as np
import pandas as pd
from SensorPlacement import SensorPlacement


""" FILE NAME: 'rbf_placement.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
    This is a simple implementation that uses the RBF kernel instead of a non-stationary
    estimated kernel as will be done in the final model.
"""

""" Data is loaded and prepared for the placement algorithm. """
data = pd.read_csv('data/csv_data/normalized/LSBU_32/average_over_time_16.csv')
V = data[['X', 'Y', 'Z']].copy().values
V = V[::3][:100]
index_sets = SensorPlacement.positionIndices(V)

tracer = data['tracer'].values # used as output to calculate cov matrix.
tracer = np.reshape(tracer, (tracer.shape[0], 1))
tracer = tracer[::3][:100]

""" The covariance matrix needed for the placement algorithm is calculated. """
k = GPy.kern.RBF(input_dim=3, name='rbf')
m = GPy.models.GPRegression(V, tracer, k) # Is this the prior or posterior
m.optimize()
m['Gaussian_noise.variance'] = 0

cov = k.K(V, V)

""" Placement algorithm is called and the results are printed in the terminal. """
# A = SensorPlacement.naiveSensorPlacement(cov, 5, V, index_sets)
A = SensorPlacement.lazySensorPlacement(cov, 5, V, index_sets)
print(A)
