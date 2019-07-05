#!/usr/bin/python
import numpy as np
import math
import pandas as pd
import operator
from matplotlib import pyplot as plt
import operator
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
    possible = pos[(pos[:,2]<=33.0)]
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
        decomp = cov_y_A @ cov_A_A @ cov_A_y
        return decomp[0][0]
    return 0

def getMutualInformation(cov, V, y, A, AHat):
    var_y = getCovarianceValue(V, cov, y, y)
    decomp1 = getMatrixDecomposition(V, cov, y, A)
    decomp2 = getMatrixDecomposition(V, cov, y, AHat)

    return (var_y - decomp1)/(var_y - decomp2)

def approximation1(cov, k, V, S, U): # parameters and variables were named like in pseudo-code of paper
    A = np.empty((0, 3))
    for j in range(0, k):
        remaining = subtract_pos(S, A)
        AHat = subtract_pos(V, A)
        delta = []

        for y in remaining:
            AHat = subtract_pos(AHat, y)
            delta.append(getMutualInformation(cov, V, y, A, AHat))

        i, v = max(enumerate(delta), key=operator.itemgetter(1))
        A = np.concatenate((A, remaining[i]), axis=0)
    return A

def approximation2(cov, k, V, S, U): # parameters and variables were named like in pseudo-code of paper
    A = np.empty((0, 3))
    delta = []
    current = []
    for y in S:
        delta.append(math.inf)

    for j in range(0, k):
        remaining = subtract_pos(S, A)
        for y in remaining:
            current.append(False);
        while True:
            i, v = max(enumerate(delta), key=operator.itemgetter(1)) # S\A instead of S
            if current[i]:
                break
            delta[i] = getMutualInformation(cov, V, y, A, AHat)
            current[i] = True;
        A = np.concatenate((A, remaining[i]), axis=0)
    return A

# def approximation3(cov, k, V, S, U): # parameters and variables were named like in pseudo-code of paper
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
data = pd.read_csv('data/csv_data/normalized/tracer_george/LSBU_500_3.csv')

obs_pos = data[['X', 'Y', 'Z']].copy().values
obs_pos = obs_pos[:100]

tracer = data['tracer'].values # used as output to calculate cov matrix. (change dimension)
tracer = np.reshape(tracer, (tracer.shape[0], 1))
tracer = tracer[:100]

########################### Simple GP implementation ###########################

# define covariance kernel. Mean is assumed to be 0.
k = GPy.kern.RBF(3, name='rbf')
m = GPy.models.GPRegression(obs_pos, tracer, k)
m.optimize()

# new test points to sample function from
X_3_sample, Y_3_sample, Z_3_sample = np.mgrid[-1:508:100, -315:326:100, 0.2:250:100]
sample_pos_3 = np.array([X_3_sample.flatten(), Y_3_sample.flatten(), Z_3_sample.flatten()]).T
possible_3, impossible_3 = getPossibleImpossible(sample_pos_3)

# X_7_sample, Y_7_sample, Z_7_sample = np.mgrid[121:683:100j, -228:219:100j, 0.2:250:100j]
# sample_pos_7 = np.array([X_3_sample.flatten(), Y_7_sample.flatten(), Z_7_sample.flatten()]).T
# possible_7, impossible_7 = getPossibleImpossible(sample_pos_7)

# calculate the posterior covariance matrix
post_mean, post_cov = m.predict(sample_pos_3, full_cov=True)

# find optimal sensor A with the first approximation function
A = approximation1(post_cov, 9, sample_pos_3, possible_3, impossible_3)
# A = approximation2(post_cov, 9, sample_pos, possible, impossible)
