#!/usr/bin/python
import numpy as np
import GPy
import heapq

""" FILE NAME: 'sensor_placement.py'
    DESCRIPTION: This file is implementing the class that will be used for sensor
    positioning according to solution proposed by Krause, Singh and Guestrin (2008)
    on the MAGIC testside in Elephant and Castle.
"""

class SensorPlacement:
    @staticmethod
    def positionIndices(V):
        V_i = np.array(range(len(V)))
        S_i = np.argwhere(V[:,2]<=33.0).flatten()
        U_i = np.setdiff1d(V_i, S_i, assume_unique=True)
        return V_i, S_i, U_i

    @staticmethod
    def conditionalEntropy(cov, y, A):
        """ This function calculates the conditional entropy of y given A. """
        conditionalVariance = cov[y, y] - (cov[np.ix_([y], A)] @ np.linalg.inv(cov[np.ix_(A, A)]) @ cov[np.ix_(A, [y])])
        return 0.5*np.log(conditionalVariance[0][0])+0.5*np.log(2*np.pi)+1

    @staticmethod
    def naiveSensorPlacement(cov, k, V, index_sets):
        """ This is an implementation of the first approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper.
            Input:
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
                delta.append(SensorPlacement.conditionalEntropy(cov, y, A_i) - SensorPlacement.conditionalEntropy(cov, y, AHat_i))

            y_i = S_A[np.argmax(delta)]
            A_i = np.append(A_i, y_i).astype(int)

            # Put position into the result set
            y_star = V[y_i]
            y_star.shape = (1, 3)
            A = np.concatenate((A, y_star), axis=0)
        return A

    @staticmethod
    def lazySensorPlacement(cov, k, V, index_sets):
        """ This is an implementation of the second approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper. It uses a priority queue in order
            to reduce the time complexity from O(k*n^4) to O(k*n^3).
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: all possible positions.
            - index_sets: set with array that have the index positions of V, S and U.
        """
        print("started")
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
                criterion = SensorPlacement.conditionalEntropy(cov, y_i, A_i) - SensorPlacement.conditionalEntropy(cov, y_i, AHat_i)
                heapq.heappush(heap, (-1 * criterion, y_i, j))

            A_i = np.append(A_i, y_i).astype(int)

            # Put position into the result set
            y_star = V[y_i]
            y_star.shape = (1, 3)
            A = np.concatenate((A, y_star), axis=0)
        return A

    @staticmethod
    def 
