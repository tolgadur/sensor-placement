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
    def __conditionalVariance(cov, y, A):
        """ This function calculates the conditional variance of y given A. """
        var = cov[y, y] - (cov[np.ix_([y], A)] @ np.linalg.inv(cov[np.ix_(A, A)]) @ cov[np.ix_(A, [y])])
        return var[0][0]

    @staticmethod
    def __conditionalEntropy(cov, y, A):
        """ This function calculates the conditional entropy of y given A. """
        conditionalVariance = SensorPlacement.__conditionalVariance(cov, y, A)
        return 0.5*np.log(conditionalVariance)+0.5*np.log(2*np.pi)+1

    @staticmethod
    def __localSet(cov, S_i, y, epsilon):
        """ This function returns the set of points Xin S for which K(y*, x) > epsilon.
            Input:
            - cov: covariance matrix
            - S_i: array with all indices of i
            - epsilon: hyperparamter
        """
        X = []
        for x in S_i:
            if cov[y, x] > epsilon:
                X.append()
        return X

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
            for y in S_A:
                AHat_i = np.setdiff1d(V_i, np.append(A_i, [y]))
                delta.append(SensorPlacement.__conditionalVariance(cov, y, A_i) / SensorPlacement.__conditionalVariance(cov, y, AHat_i))

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
                criterion = SensorPlacement.__conditionalVariance(cov, y_i, A_i) / SensorPlacement.__conditionalVariance(cov, y_i, AHat_i)
                heapq.heappush(heap, (-1 * criterion, y_i, j))

            A_i = np.append(A_i, y_i).astype(int)

            # Put position into the result set
            y_star = V[y_i]
            y_star.shape = (1, 3)
            A = np.concatenate((A, y_star), axis=0)
        return A

    @staticmethod
    def localKernelPlacement(cov, k, V, index_sets): # Fix!
        """ This is an implementation of the third approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper. It only considers local kernels
            in order to reduce the time complexity O(k*n).
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: all possible positions.
            - index_sets: set with array that have the index positions of V, S and U.
        """
        V_i, S_i, U_i = index_sets
        A = np.empty((0, 3))
        A_i = []
        delta = []

        for y in S_i:
            # Simple and stupid implementation: Fix and implement truncated!
            V_y = np.setdiff1d(V_i, y).astype(int)
            delta.append(cov[y, y] / SensorPlacement.__conditionalVariance(cov, y, V_y))
        for j in range(0, k):
            y_i = S_i[np.argmax(delta)]
            A_i = np.append(A_i, y_i).astype(int)

            y_star = V[y_i]
            y_star.shape = (1, 3)
            A = np.concatenate((A, y_star), axis=0)

            N = SensorPlacement.__localSet(cov, S_i, y_i, 0.1) # What is a good value for epsilon?
            for y in N:
                delta[y] = SensorPlacement.__conditionalVariance(cov, y_i, A_i) / SensorPlacement.__conditionalVariance(cov, y_i, AHat_i)
        return A

    @staticmethod
    def localKernelLazyPlacement(cov, k, V, index_sets): # Fix
        """ This is an implementation of the third approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper. It only considers local kernels
            in order to reduce the time complexity O(k*n).
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: all possible positions.
            - index_sets: set with array that have the index positions of V, S and U.
        """
        V_i, S_i, U_i = index_sets
        A = np.empty((0, 3))
        A_i = []
        delta = -1 * np.inf * np.ones((len(S_i), 1))

        for i, y in enumerate(S_i):
            V_y = np.setdiff1d(V_i, y).astype(int)
            delta[i] = cov[y, y] / SensorPlacement.__conditionalVariance(cov, y, V_y) #wrong H(Yleverthing)

        current = -1 * np.ones((len(S_i), 1))
        heap = list(zip(delta, S_i, current))
        heapq.heapify(heap)

        for j in range(0, k):
            delta, y_i, current = heapq.heappop(heap)
            if current == j:
                break
            A_i = np.append(A_i, y_i).astype(int)
            # Append to solution set
            y_star = V[y_i]
            y_star.shape = (1, 3)
            A = np.concatenate((A, y_star), axis=0)

            AHat_i = np.setdiff1d(V_i, np.append(A_i, [y_i]))
            N = SensorPlacement.__localSet(cov, S_i, y_i, 0.1) # What is a good value for epsilon?
            for y in N:
                delta[y] = SensorPlacement.__conditionalVariance(cov, y_i, A_i) / SensorPlacement.__conditionalVariance(cov, y_i, AHat_i)
        return A
