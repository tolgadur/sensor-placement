#!/usr/bin/python
import numpy as np
import pandas as pd
import GPy
import heapq

""" FILE NAME: 'simplePlacement.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
    It should be noted that this is a simple implementation that uses the RBF kernel
    instead of a non-stationary estimated kernel as will be done in the final model.
    Furthermore, thismodel is not parrallelised.

"""

def positionIndices(V):
    V_i = np.array(range(len(V)))
    S_i = np.argwhere(V[:,2]<=33.0).flatten()
    U_i = np.setdiff1d(V_i, S_i, assume_unique=True)
    return V_i, S_i, U_i

def conditionalEntropy(cov, y, A):
    """ This function calculates the conditional entropy of y given A. """
    conditionalVariance = cov[y, y] - (cov[np.ix_([y], A)] @ np.linalg.inv(cov[np.ix_(A, A)]) @ cov[np.ix_(A, [y])])
    return 0.5*np.log(conditionalVariance[0][0])+0.5*np.log(2*np.pi)+1

def naiveSensorPlacement(cov, k, V, index_sets):
    """ This is an implementation of the first approximation function suggested in
        the 'Near-Optimal Sensor Placement' paper.
        Inputs:
        - cov: covariance matrix
        - k: number of Sensors to be placed
        - V: all possible positions.
        - index_sets: set with array that have the index positions of V, S and U.
    """

    V_i, S_i, U_i = index_sets
    A = np.empty((0, 3))
    A_i = []

    for j in range(0, k):
        S_A = np.setdiff1d(S_i, A_i).astype(int)
        delta = []
        for i, y in enumerate(S_A):
            AHat_i = np.setdiff1d(V_i, np.append(A_i, [y]))
            delta.append(conditionalEntropy(cov, y, A_i) - conditionalEntropy(cov, y, AHat_i))

        y_i = S_A[np.argmax(delta)]
        A_i = np.append(A_i, y_i).astype(int)

        # Put position into the result set
        y_star = V[y_i]
        y_star.shape = (1, 3)
        A = np.concatenate((A, y_star), axis=0)
    return A

def lazySensorPlacement(cov, k, V, index_sets):
    """ This is an implementation of the second approximation function suggested in
        the 'Near-Optimal Sensor Placement' paper. It uses a priority queue in order
        to reduce the time complexity from O(k*n^4) to O(k*n^3).
        Inputs:
        - cov: covariance matrix
        - k: number of Sensors to be placed
        - V: all possible positions.
        - index_sets: set with array that have the index positions of V, S and U.
    """

    V_i, S_i, U_i = index_sets
    A = np.empty((0, 3))
    A_i = []

    delta = -1 * np.inf * np.ones((len(S_i), 1))
    current = np.ones((len(S_i), 1))
    heap = list(zip(delta, S_i, current))
    heapq.heapify(heap)

    for j in range(k):
        S_A = np.setdiff1d(S_i, A_i).astype(int)
        current = []
        while True:
            delta, y_i, current = heapq.heappop(heap)
            if current == j:
                break
            AHat_i = np.setdiff1d(V_i, np.append(A_i, [y_i]))
            criterion = conditionalEntropy(cov, y_i, A_i) - conditionalEntropy(cov, y_i, AHat_i)
            heapq.heappush(heap, (-1 * criterion, y_i, j))

        A_i = np.append(A_i, y_i).astype(int)

        # Put position into the result set
        y_star = V[y_i]
        y_star.shape = (1, 3)
        A = np.concatenate((A, y_star), axis=0)
    return A

""" Data is loaded and prepared for the placement algorithm. """
data = pd.read_csv('data/csv_data/normalized/LSBU_32/average_over_time_16.csv')
V = data[['X', 'Y', 'Z']].copy().values
V = V[::3][:100]
index_sets = positionIndices(V)

tracer = data['tracer'].values # used as output to calculate cov matrix.
tracer = np.reshape(tracer, (tracer.shape[0], 1))
tracer = tracer[::3][:100]

""" The covariance matrix needed for the placement algorithm is calculated. """
k = GPy.kern.RBF(3, name='rbf')
m = GPy.models.GPRegression(V, tracer, k)
m.optimize()

cov = k.K(V, V)

""" Placement algorithm is called and the results are printed in the terminal. """
A = naiveSensorPlacement(cov, 5, V, index_sets)
# A = lazySensorPlacement(cov, 5, V, index_sets)
print(A)
