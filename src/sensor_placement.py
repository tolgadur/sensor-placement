#!/usr/bin/python
import numpy as np
import heapq
import multiprocessing as mp
import pandas as pd
import GPy
import queue

from pprint import pprint

""" FILE NAME: 'sensor_placement.py'
    DESCRIPTION: This file is implementing the class that will be used for sensor
    positioning according to solution proposed by Krause, Singh and Guestrin (2008)
    on the MAGIC testside in Elephant and Castle.
"""

class SensorPlacement:
    @staticmethod
    def __isMonotonic(cov, k, V, S, U):
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
    def __positionIndices(V):
        V_i = np.array(range(len(V)))
        S_i = np.argwhere(V[:,2]<=30.0).flatten()
        U_i = np.setdiff1d(V_i, S_i, assume_unique=True)
        return V_i, S_i, U_i

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
        A_ = SensorPlacement.__localSet(cov, y, A, epsilon)
        return SensorPlacement.__conditionalEntropy(cov, y, A_)

    @staticmethod
    def __localConditionalVariance(cov, y, A, epsilon):
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
    def naiveSensorPlacement(cov, k, V, S, U, A, area=None, output=None):
        """ This is an implementation of the first approximation function suggested in
            the 'Near-Optimal Sensor Placement' paper.
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        print('Algorithm is starting for area', area, flush=True)
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
            print('Area ', area, ': ', A, flush=True)
        if area != None:
            output.put((area, A))
        return A

    @staticmethod
    def lazySensorPlacement(cov, k, V, S, U, A, area=None, output=None):
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
        print('Algorithm is starting for area', area, flush=True)
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
            print('Area ', area, ': ', A, flush=True)
        if area != None:
            output.put((area, A))
        return A

    @staticmethod
    def localKernelPlacement(cov, k, V, S, U, A, area=None, output=None):
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
        print('Algorithm is starting for area', area, flush=True)
        A = A
        epsilon = 1e-10

        delta = np.array([]); N = S
        for y in S:
            V_y = np.setdiff1d(V, y).astype(int)
            delta = np.append(delta, cov[y, y] / SensorPlacement.__localConditionalVariance(cov, y, V_y, epsilon))

        for j in range(k):
            y_star = N[np.argmax(delta)]
            A = np.append(A, y_star).astype(int)
            print('Area ', area, ': ', A, flush=True)

            N = SensorPlacement.__localSet(cov, y_star, S, epsilon)
            N = np.setdiff1d(S, A).astype(int)
            delta = np.array([])
            for y in N:
                AHat = np.setdiff1d(V, np.append(A, [y]))
                delta = np.append(delta, SensorPlacement.__localConditionalVariance(cov, y, A, epsilon) / \
                                         SensorPlacement.__localConditionalVariance(cov, y, AHat, epsilon))

        if area != None:
            output.put((area, A))
        return A

    @staticmethod
    def lazyLocalKernelPlacement(cov, k, V, S, U, A, area=None, output=None):
        """ This is a mix between the lazySensorPlacement method and the localKernelPlacement
            method.
            Input:
            - cov: covariance matrix
            - k: number of Sensors to be placed
            - V: indices of all position
            - S: indices of all possible sensor positions
            - U: indices of all impossible sensor positions
        """
        print('Algorithm is starting for area', area, flush=True)
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
            print('Area ', area, ': ', A, flush=True)
        if area != None:
            output.put((area, A))
        return A

    @staticmethod
    def simplePlacement(position_file, tracer_file, k, algorithm_choice=None, already_placed=np.array([])):
        """ This function computes the optimal sensor placement on one area.
            Input:
            - position_file: Filepath to the position file (has to be a csv-file)
            - tracer_file: Filepath to the tracer file (has to be a csv-file)
            - algorithm_choice: Integer specifying which approximation algorithm the sensor
              positions are calculated with. '1' == naive, '2' == priority queue,
              'default' == local kernel.
        """
        """ Preparing the index arrays """
        V_df = pd.read_csv(position_file)
        V = V_df[['X', 'Y', 'Z']].copy().values
        V = V[:100:2]
        V_i, S_i, U_i = SensorPlacement.__positionIndices(V)

        """ Preparing the sample covariance matrix """
        tracer_df = pd.read_csv(tracer_file)
        tracer = tracer_df.values
        tracer = tracer[:100:2]

        cov = np.cov(tracer)

        """ Executing algorithm """
        if algorithm_choice==1:
            return SensorPlacement.naiveSensorPlacement(cov, k, V_i, S_i, U_i, already_placed)
        elif algorithm_choice==2:
            return SensorPlacement.lazySensorPlacement(cov, k, V_i, S_i, U_i, already_placed)
        elif algorithm_choice==3:
            return SensorPlacement.localKernelPlacement(cov, k, V_i, S_i, U_i, already_placed)
        else:
            return SensorPlacement.lazyLocalKernelPlacement(cov, k, V_i, S_i, U_i, already_placed)

    @staticmethod
    def parallelPlacement(position_files, tracer_files, k, algorithm_choice=None, already_placed=None):
        """ This function is used to compute the sensor placement on multiple areas
            concurrently using the multiprocessing library of python. NOTE: The tracer
            files and position files have to correspont to each other. This means
            that the tracer file at position 1, for instance, has to describe the
            same area as the position file at position 1.
            Input:
            - position_files: Array with the filepaths to the position files (have to be csv-files)
            - tracer_files: Array with the filepaths to the tracer files (have to be csv-files)
        """
        print('Starting parallel placement...', flush=True)
        already_placed = [np.array([])]*len(position_files) if already_placed==None else already_placed
        V_i, S_i, U_i, cov = [], [], [], []
        for i in range(0, len(position_files)):
            """ Preparing the index arrays """
            V_df = pd.read_csv(position_files[i])
            V = V_df[['X', 'Y', 'Z']].copy().values
            V = V[:100:2]
            V_, S_, U_ = SensorPlacement.__positionIndices(V)
            V_i.append(V_); S_i.append(S_); U_i.append(U_)

            """ Preparing the sample covariance matrix """
            tracer_df = pd.read_csv(tracer_files[i])
            tracer = tracer_df.values
            tracer = tracer[:100:2]

            cov.append(np.cov(tracer))

        """ Choosing Algorithm """
        output = mp.Queue()

        if algorithm_choice==1:
            processes = [mp.Process(target=SensorPlacement.naiveSensorPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], already_placed[x], x, output)) for x in range(len(cov))]
        elif algorithm_choice==2:
            processes = [mp.Process(target=SensorPlacement.lazySensorPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], already_placed[x], x, output)) for x in range(len(cov))]
        elif algorithm_choice==3:
            processes = [mp.Process(target=SensorPlacement.localKernelPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], already_placed[x], x, output)) for x in range(len(cov))]
        else:
            processes = [mp.Process(target=SensorPlacement.lazyLocalKernelPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], already_placed[x], x, output)) for x in range(len(cov))]

        """ Algorithm starts computing for all areas in parallel """
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [output.get() for p in processes]
