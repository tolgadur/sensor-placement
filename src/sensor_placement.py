#!/usr/bin/python
import numpy as np
import heapq
import pandas as pd
import GPy
import queue

from pprint import pprint

""" FILE NAME: 'sensor_placement.py'
    DESCRIPTION: This file is implementing the class that will be used for sensor
    positioning according to solution proposed by Krause, Singh and Guestrin (2008).
"""

class SensorPlacement:
    @staticmethod
    def isMonotonic(cov, k, V, S, U):
        """ This function checks if values in the dataset are monotonic or not. For
            datasets > 2000 observations, non-monotonicity might lead to suboptimal
            results.
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        A = np.array([])
        for j in range(k):
            S_A = np.setdiff1d(S, A).astype(int)
            for y in S_A:
                AHat = np.setdiff1d(V, np.append(A, [y]))
                condition = SensorPlacement.__conditionalEntropy(cov, y, A) - SensorPlacement.__conditionalEntropy(cov, y, AHat)
                if condition < 0:
                    print(condition)
                    return False
        return True

    @staticmethod
    def __conditionalVariance(cov, y, A):
        """ This function calculates the conditional variance of y given A. """
        var = np.absolute(cov[y, y] - (cov[np.ix_([y], A)] @ np.linalg.inv(cov[np.ix_(A, A)]) @ cov[np.ix_(A, [y])]))
        return var[0][0]

    @staticmethod
    def __conditionalEntropy(cov, y, A):
        """ This function calculates the conditional entropy of y given A. """
        conditionalVariance = SensorPlacement.__conditionalVariance(cov, y, A)
        return 0.5 * np.log(2*np.pi*conditionalVariance)

    @staticmethod
    def __localConditionalEntropy(cov, y, A, epsilon):
        """ This function calculates the conditional entropy of y given A for
            all values where cov[y, A] > epsilon. """
        A_ = SensorPlacement.__localSet(cov, y, A, epsilon)
        return SensorPlacement.__conditionalEntropy(cov, y, A_)

    @staticmethod
    def __localConditionalVariance(cov, y, A, epsilon):
        """ This function calculates the conditional variance of y given A for
            all values where cov[y, A] > epsilon. """
        A_ = SensorPlacement.__localSet(cov, y, A, epsilon)
        return SensorPlacement.__conditionalVariance(cov, y, A_)

    @staticmethod
    def __localSet(cov, y, A, epsilon):
        """ This function returns the set of points X in S for which K(y*, x) > epsilon.
            Input:
            - cov: covariance matrix
            - S_i: array with all indices of i
            - epsilon: hyperparamter
        """
        return [x for x in A if cov[y, x] > epsilon]

    @staticmethod
    def naiveSensorPlacement(cov, k, V, S, U, A, subdomain=None, output=None):
        """ This is an implementation of the first approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper.
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        print('Algorithm is starting for subdomain', subdomain, flush=True)
        A = A

        for j in range(k):
            S_A = np.setdiff1d(S, A).astype(int)
            delta = np.array([])
            for y in S_A:
                AHat = np.setdiff1d(V, np.append(A, [y]))
                delta = np.append(delta, SensorPlacement.__conditionalVariance(cov, y, A) / \
                                         SensorPlacement.__conditionalVariance(cov, y, AHat))
            y_star = S_A[np.argmax(delta)]
            A = np.append(A, y_star).astype(int)
            print('subdomain ', subdomain, ': ', A, flush=True)
        if subdomain != None:
            output.put((subdomain, 2*A))
        return 2*A

    @staticmethod
    def lazySensorPlacement(cov, k, V, S, U, A, subdomain=None, output=None):
        """ This is an implementation of the second approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper. It uses a priority queue in order
            to reduce the time complexity from O(k*n^4) to O(k*n^3).
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        print('Algorithm is starting for subdomain', subdomain, flush=True)
        A = A

        delta = -1 * np.inf * np.ones((len(S), 1))
        heap = [(delta[i], S[i], -1) for i in range(len(delta))]
        heapq.heapify(heap)

        for j in range(k):
            while True:
                delta_star, y_star, current = heapq.heappop(heap)
                if current == j:
                    break
                AHat = np.setdiff1d(V, np.append(A, [y_star]))
                criterion = SensorPlacement.__conditionalVariance(cov, y_star, A) / \
                            SensorPlacement.__conditionalVariance(cov, y_star, AHat)
                heapq.heappush(heap, (-1 * criterion, y_star, j))

            A = np.append(A, y_star).astype(int)
            print('subdomain ', subdomain, ': ', 2*A, flush=True)
        if subdomain != None:
            output.put((subdomain, 2*A))
        return 2*A

    @staticmethod
    def localKernelPlacement(cov, k, V, S, U, A, subdomain=None, output=None):
        """ This is an implementation of the third approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper. It only considers local kernels
            in order to reduce the time complexity O(k*n).
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        print('Algorithm is starting for subdomain', subdomain, flush=True)
        A = A
        epsilon = 1e-10

        delta = np.array([]); N = S
        for y in S:
            V_y = np.setdiff1d(V, y).astype(int)
            delta = np.append(delta, cov[y, y] / SensorPlacement.__localConditionalVariance(cov, y, V_y, epsilon))

        for j in range(k):
            y_star = N[np.argmax(delta)]
            A = np.append(A, y_star).astype(int)
            print('subdomain ', subdomain, ': ', A, flush=True)

            N = SensorPlacement.__localSet(cov, y_star, S, epsilon)
            N = np.setdiff1d(S, A).astype(int)
            delta = np.array([])
            for y in N:
                AHat = np.setdiff1d(V, np.append(A, [y]))
                delta = np.append(delta, SensorPlacement.__localConditionalVariance(cov, y, A, epsilon) / \
                                         SensorPlacement.__localConditionalVariance(cov, y, AHat, epsilon))

        if subdomain != None:
            output.put((subdomain, 2*A))
        return 2*A

    @staticmethod
    def lazyLocalKernelPlacement(cov, k, V, S, U, A, subdomain=None, output=None):
        """ This is a mix between the lazySensorPlacement method and the localKernelPlacement
            method.
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        print('Algorithm is starting for subdomain', subdomain, flush=True)
        A = A
        epsilon = 1e-10

        delta = -1 * np.inf * np.ones((len(S), 1))
        heap = [(delta[i], S[i], -1) for i in range(len(delta))]
        heapq.heapify(heap)

        for j in range(k):
            while True:
                delta_star, y_star, current = heapq.heappop(heap)
                if current == j:
                    break
                AHat = np.setdiff1d(V, np.append(A, [y_star]))
                criterion = SensorPlacement.__localConditionalVariance(cov, y_star, A, epsilon) / \
                            SensorPlacement.__localConditionalVariance(cov, y_star, AHat, epsilon)
                heapq.heappush(heap, (-1 * criterion, y_star, j))

            A = np.append(A, y_star).astype(int)
            print('subdomain ', subdomain, ': ', A, flush=True)
        if subdomain != None:
            output.put((subdomain, 2*A))
        return 2*A
