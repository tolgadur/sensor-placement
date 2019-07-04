#!/usr/bin/python
import numpy as np
import math
import pandas as pd
import operator
from matplotlib import pyplot as plt
import GPy

################################################################################
#        FILE NAME: 'simple_gp.py'                                             #
#        DESCRIPTION: This file is implementing the sensor positioning solu-   #
#        -tion proposed by Krause, Singh and Guestrin (2008) on the MAGIC test #
#        side in Elephant and Castle. It should be noted that this is a simple #
#        implementation that uses the RBF kernel instead of non-stationary es- #
#        -timated kernel as will be done in the final model. Furthermore, this #
#        model is not parrallelised.                                           #
################################################################################

############################# Function Definitions #############################

def subtract_pos(A, B): # subtracts multidimensional numpy array according to set theory
    if B.ndim > 1 and B.shape[0] > 0:
        cumdims = (np.maximum(A.max(),B.max())+1)**np.arange(B.shape[1])
        return A[~np.in1d(A.dot(cumdims),B.dot(cumdims))]
    elif B.shape[0] != 0:
        index = getIndex(A, B)
        return np.delete(A, index, 0)
    else:
        return A

def getPossibleImpossible(pos):
    possible = pos[(pos[:,2]<=29.0)]
    impossible = subtract_pos(pos, possible)
    return possible, impossible

def getIndex(positions, y):
    if np.any((positions[:] == y).all(1)):
        return np.where((positions == y).all(axis=1))[0][0]
    else:
        print('Could not get index. Row is not in the matrix.')
        exit(0)

def getCovarianceValue(V, cov, v1, v2):
    i1 = getIndex(V, v1)
    i2 = getIndex(V, v2)
    return cov[i1][i2]

def getCovarianceMatrix(V, cov, v1, v2): # TO-DO: improve time complexity
    dim1 = 1 if (v1.ndim == 1) else v1.shape[0]
    dim2 = 1 if (v2.ndim == 1) else v2.shape[0]
    matrix = np.zeros((dim1, dim2))

    if dim1 == 1:
        for i, v in enumerate(v2):
            matrix[0][i] = getCovarianceValue(V, cov, v1, v)
    elif dim2 == 1:
        for i, v in enumerate(v1):
            matrix[i][0] = getCovarianceValue(V, cov, v, v2)
    else:
        for i, vv1 in enumerate(v2):
            for j, vv2 in enumerate(v2):
                matrix[i][j] = getCovarianceValue(V, cov, vv1, vv2)
    return matrix

def getMatrixDecomposition(V, cov, y, A):
    if len(A) != 0:
        cov_y_A = getCovarianceMatrix(V, cov, y, A)
        cov_A_A = np.linalg.inv(getCovarianceMatrix(V, cov, A, A))
        cov_A_y = getCovarianceMatrix(V, cov, A, y)
        return cov_y_A @ cov_A_A @ cov_A_y
    return 0

def getMutualInformation(cov, V, y, A, AHat):
    var_y = getCovarianceValue(V, cov, y, y)
    decomp1 = getMatrixDecomposition(V, cov, y, A)
    decomp2 = getMatrixDecomposition(V, cov, y, AHat)

    print('decomp2', decomp2)
    print('var_y', var_y)
    print('delta', delta)

    return (var_y - decomp1)/(var_y - decomp2)

def approximation1(cov, k, V, S, U): # parameters and variables were named like in pseudo-code of paper
    A = np.empty((0, 3))
    for j in range(0, k):
        remaining = subtract_pos(S, A)
        AHat = subtract_pos(V, A)
        delta_max = 0
        y_max = np.zeros((1, 3))

        for y in remaining:
            AHat = subtract_pos(AHat, y)

            delta = getMutualInformation(cov, V, y, A, AHat)

            if delta_max < delta[0][0]:
                delta_max = delta
                y_max[0][0] = y[0]; y_max[0][1] = y[1]; y_max[0][2] = y[2]

        A = np.concatenate((A, y_max), axis=0)
    return A

def approximation2(cov, k, V, S, U):
    A = np.empty((0, 3))
    delta = []
    current = []
    for i, y in enumerate(S):
        delta.append((math.inf, i))
    for j in range(0, k):
        remaining = subtract_pos(S, A)
        for y in remaining:
            current.append(False);
        while True:
            delta_index, delta_value = max(enumerate(delta), key=operator.itemgetter(1))
            y_index = delta[delta_i][1]
            if current[y_index]:
                break
            delta[delta_index] = getMutualInformation(cov, V, y, A, AHat)
            current[y_index] = True;
        A = np.concatenate((A, V[y_index]), axis=0) # won't work
    return A

# def approximation3(cov, k, V, S, U):
#     A = np.empty((0, 3))
#     delta = []
#     AHat = subtract_pos(V, A)
#     for y in S:
#         AHat = subtract_pos(AHat, y)
#         delta.append(getMutualInformation(cov, V, y, A, AHat))
#     for j in range(0, k):
#         # yStar = argmaxyDeltay
#         A = np.concatenate((A, y_star), axis=0)
#         for  y in :
#
#
#     return A

########################### Preparing data for GP ##############################

# loading and preparing data from csv file
data = pd.read_csv('data/csv_data/standardized/timestep_500.csv')

obs_pos = data[['X', 'Y', 'Z']].copy().values
obs_pos = obs_pos[:100]

tracer = data['tracer_background'].values # used as output to calculate cov matrix. (change dimension)
tracer = np.reshape(tracer, (100040, 1))
tracer = tracer[:100]

########################### Simple GP implementation ###########################

# define covariance kernel. Mean is assumed to be 0.
k = GPy.kern.RBF(3, name='rbf')
m = GPy.models.GPRegression(obs_pos, tracer, k)
m.optimize()

# new test points to sample function from
X_sample, Y_sample, Z_sample = np.mgrid[-360:360:100, -340:340:100, 0:250:100] # check if it is correct
sample_pos = np.array([X_sample.flatten(), Y_sample.flatten(), Z_sample.flatten()]).T
possible, impossible = getPossibleImpossible(sample_pos)

# calculate the posterior covariance matrix
post_mean, post_cov = m.predict(sample_pos, full_cov=True)

# find optimal sensor A with the first approximation function
A = approximation1(post_cov, 9, sample_pos, possible, impossible)
# A = approximation2(post_cov, 9, sample_pos, possible, impossible)
