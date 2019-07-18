#!/usr/bin/python
import numpy as np
import heapq
import multiprocessing as mp
import pandas as pd

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
    def __localConditionalVariance(cov, y, A, epsilon):
        A_ = SensorPlacement.__localSet(cov, y, A, epsilon )
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
    def naiveSensorPlacement(cov, k, V, S, U, area, output):
        """ This is an implementation of the first approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper.
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        A = []

        for j in range(0, k):
            S_A = np.setdiff1d(S, A).astype(int)
            delta = []
            for y in S_A:
                AHat = np.setdiff1d(V, np.append(A, [y]))
                delta.append(SensorPlacement.__conditionalVariance(cov, y, A) / \
                             SensorPlacement.__conditionalVariance(cov, y, AHat))

            y = S_A[np.argmax(delta)]
            A = np.append(A, y).astype(int)
        output.put((area, A))
        return A

    @staticmethod
    def lazySensorPlacement(cov, k, V, S, U, area, output):
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
        A = []

        delta = -1 * np.inf * np.ones((len(S), 1))
        current = np.ones((len(S), 1))
        heap = list(zip(delta, S, current))
        heapq.heapify(heap)

        for j in range(k):
            while True:
                delta, y, current = heapq.heappop(heap)
                if current == j:
                    break
                AHat = np.setdiff1d(V, np.append(A, [y]))
                criterion = SensorPlacement.__conditionalVariance(cov, y, A) / \
                            SensorPlacement.__conditionalVariance(cov, y, AHat)
                heapq.heappush(heap, (-1 * criterion, y, j))

            A = np.append(A, y).astype(int)
        output.put((area, A))
        return A

    @staticmethod
    def localKernelPlacement(cov, k, V, S, U, area, output):
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
        A = []
        epsilon = 1.e-100

        delta = []
        for y in S:
            V_y = np.setdiff1d(V, y).astype(int)
            delta.append(-1 * (cov[y, y] / SensorPlacement.__localConditionalVariance(cov, y, V_y, epsilon)))
        heap = list(zip(delta, S))
        heapq.heapify(heap)

        for j in range(0, k):
            delta, y = heapq.heappop(heap)
            A = np.append(A, y).astype(int)

            N = SensorPlacement.__localSet(cov, y, S, epsilon)
            for y in N:
                AHat = np.setdiff1d(V, np.append(A, [y]))
                delta = SensorPlacement.__localConditionalVariance(cov, y, A, epsilon) / \
                        SensorPlacement.__localConditionalVariance(cov, y, AHat, epsilon)
                heapq.heappush(heap, (-1 * delta, y))
        output.put((area, A))
        return A

    @staticmethod
    def parallelPlacement(position_files, tracer_files):
        """ This function is used to compute the sensor placement on multiple areas
            concurrently using MPI.
        """
        V_i, S_i, U_i = [], [], []
        for i in position_files:
            V_df = pd.read_csv(i)
            V = V_df[['X', 'Y', 'Z']].copy().values
            V = V[:100]
            V_, S_, U_ = SensorPlacement.positionIndices(V)
            V_i.append(V_)
            S_i.append(S_)
            U_i.append(U_)

        cov = []
        for j in tracer_files:
            tracer_df = pd.read_csv(j)
            tracer = tracer_df.values
            tracer = tracer[:100]
            cov.append(np.cov(tracer))

        output = mp.Queue()
        processes = [mp.Process(target=SensorPlacement.localKernelPlacement,
                                args=(cov[x], 4, V_i[x], S_i[x], U_i[x], x, output)) for x in range(len(cov))]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [output.get() for p in processes]
