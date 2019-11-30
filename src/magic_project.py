#!/usr/bin/python
import numpy as np
import pandas as pd
import multiprocessing as mp
from matplotlib import pyplot as plt
from sensor_placement import SensorPlacement

""" FILE NAME: 'magic_project.py'
    DESCRIPTION: This file is implementing the class that serves as the API for the
    sensor placement in the Magic test-side in Elephant and Castle. It assumes that
    the position and tracer values are saved in csv-files in the subfolder
    '/data/csv_data/subdomain_i/'. If this is not the case then 'dataPreparation.py'
    should be run before with the vtu files.
"""

class MagicProject:
    def describeData(subdomain):
        """ This method describes the data in the specified subdomain.
            Input:
            - tracer: tracer variable we are interested in
            - subdomain: integers in range [0,31] indicating the subdomain we are interested in
        """
        df = pd.read_csv('data/csv_data/subdomain_'+str(subdomain)+'/tracer.csv')
        df = df.apply(pd.DataFrame.describe, axis=1)
        print(df.sort_values('max', ascending=False))
        return None

    def plotResiuals(subdomain, number_bins=600):
        df = pd.read_csv('data/csv_data/subdomain_'+str(subdomain)+'/tracer.csv')
        df = df.T
        residuals = df - df.mean()
        data = residuals[450].values

        plt.figure(figsize=(10, 4))
        counts, bins = np.histogram(data, bins=number_bins)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()
        return None

    def plotHistogram(subdomain, number_bins=600):
        """ This method prints calculates and displays an histogram based on the
            inputted data.
            Input:
            - tracer: tracer variable we are interested in
            - subdomain: integers in range [0,31] indicating the subdomain to plot the histogram of
            - number_bins: specifies how many bins the histogram should have
        """
        df = pd.read_csv('data/csv_data/subdomain_'+str(subdomain)+'/tracer.csv')
        data = df.mean(axis=1)
        plt.figure(figsize=(10, 4))
        counts, bins = np.histogram(data, bins=number_bins)
        plt.title('LSBU32_'+str(subdomain)+'. Number of bins: '+str(number_bins))
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()

        """ Printing histogram information on shell """
        print('LSBU32_'+str(subdomain)+':\n')
        print('Bins:\n', bins, '\n')
        print('counts:\n', counts, '\n')

    @staticmethod
    def __positionIndices(V):
        """ This helper method separates all positions into placeable and
            and unplaceable positions. The condition is hardcoded: all positions
            higher than 30m are unplacable. The rest is placeable.
            Input:
            - V: array with all positions
        """
        V_i = np.array(range(len(V)))
        S_i = np.argwhere(V[:,2]<=30.0).flatten()
        U_i = np.setdiff1d(V_i, S_i, assume_unique=True)
        return V_i, S_i, U_i

    @staticmethod
    def __allSubDomains():
        """ This method loads and concatenates the csv-files of all subdomains into
            two large dataframes.
        """
        V_df = pd.read_csv('data/csv_data/subdomain_0/positions.csv')
        tracer_df = pd.read_csv('data/csv_data/subdomain_0/tracer.csv')
        for i in range(31):
            position_file = 'data/csv_data/subdomain_'+str(i)+'/positions.csv'
            tracer_file = 'data/csv_data/subdomain_'+str(i)+'/tracer.csv'

            V_df = pd.concat([V_df, pd.read_csv(position_file)], sort=False)
            tracer_df = pd.concat([tracer_df, pd.read_csv(tracer_file)], sort=False)

        return V_df, tracer_df

    @staticmethod
    def __dataPreperation(subdomain):
        """ This helper method prepares the data in the format that is required for
            the sensor placement algorithms.
            Input:
            - subdomain: integers in range [0,31] indicating the subdomain we are interested in
        """
        if subdomain==-1:
            V_df, tracer_df = MagicProject.__allSubDomains()
        else:
            position_file = 'data/csv_data/subdomain_'+str(subdomain)+'/positions.csv'
            tracer_file = 'data/csv_data/subdomain_'+str(subdomain)+'/tracer.csv'
            V_df = pd.read_csv(position_file)
            tracer_df = pd.read_csv(tracer_file)

        """ Preparing index and tracer arrays """
        V = V_df[['X', 'Y', 'Z']].copy().values
#         V = V[::2]
        V_i, S_i, U_i = MagicProject.__positionIndices(V)

        tracer = tracer_df.values
#         tracer = tracer[::2]

        return tracer, V_i, S_i, U_i

    @staticmethod
    def validation_random(subdomain, k=4):
        pos = pd.read_csv('data/csv_data/subdomain_'+str(subdomain)+'/positions.csv')

        random_mean = 0
        for i in range(10):
            print(i)
            indLoc = np.random.randint(pos.shape[0]/2, size=(k))
            real, prediction = MagicProject.validation(subdomain, indLoc)
            random_mean += prediction

        return (random_mean/10)

    @staticmethod
    def validation(subdomain, A):
        """ This method validates our sensor placed by predicting all positions
            with the trained GP and then computing the difference to the real values.
            This error value can then be compared to random placements.
            Inputs:
            - subdomain: subdomain of interest
            - A: indices of sensors to place.
        """
#         A = A * 2
        tracer, V_i, S_i, U_i = MagicProject.__dataPreperation(subdomain)
        mean_tracer = tracer.mean(axis=1)
        cov = np.cov(tracer)
        X = np.arange(0, cov.shape[0])
        Y = tracer[A].mean(axis=1)

        # post_cov = np.absolute(cov[y, y] - (cov[np.ix_(y, A)] @ np.linalg.inv(cov[np.ix_(A, A)]) @ cov[np.ix_(A, y)]))
        post_mean = (cov[np.ix_(X, A)] @ np.linalg.inv(cov[np.ix_(A, A)])) @ Y

        return mean_tracer.mean(), post_mean.mean()

    @staticmethod
    def simplePlacement(subdomain=0, k=4, algorithm=None, placed=np.array([], dtype=int)):
        """ This method computes the optimal sensor placement on one subdomain.
            Input:
            - subdomain: integers in range [0,31] indicating the subdomain to place sensors in
            - k: number of sensors to be placed
            - algorithm: integer specifying which approximation algorithm the sensor
              positions are calculated with. '1' == naive, '2' == priority queue,
              'default' == local kernel.
            - placed: array with already placed sensors
        """
        print('Starting sensor placement...', flush=True)
        tracer, V_i, S_i, U_i = MagicProject.__dataPreperation(subdomain)
        cov = np.cov(tracer)
        cov += 1e-8*np.eye(cov.shape[0])
        placed = np.divide(placed, 2).astype(int)

        """ Choosing and executing algorithm """
        if algorithm==1:
            return SensorPlacement.naiveSensorPlacement(cov, k, V_i, S_i, U_i, placed)
        elif algorithm==2:
            return SensorPlacement.lazySensorPlacement(cov, k, V_i, S_i, U_i, placed)
        elif algorithm==3:
            return SensorPlacement.localKernelPlacement(cov, k, V_i, S_i, U_i, placed)
        else:
            return SensorPlacement.lazyLocalKernelPlacement(cov, k, V_i, S_i, U_i, placed)

    @staticmethod
    def parallelPlacement(subdomains, k, algorithm=None, placed=None):
        """ This method is used to compute the sensor placement on multiple subdomains
            concurrently using the multiprocessing library of python. NOTE: The tracer
            files and position files have to correspont to each other. This means
            that the tracer file at position 1, for instance, has to describe the
            same subdomain as the position file at position 1.
            Input:
            - subdomains: array of integers in range [0,31] indicating the subdomains to place sensors in
            - k: number of sensors to be placed
            - algorithm: integer specifying which approximation algorithm the sensor
              positions are calculated with. '1' == naive, '2' == priority queue,
              'default' == local kernel.
            - placed: array with already placed sensors
        """
        print('Starting parallel placement...', flush=True)
        placed = [np.array([], dtype=int)]*len(subdomains) if placed==None else np.divide(placed, 2).astype(int)

        V_i, S_i, U_i, cov = [], [], [], []
        for i in subdomains:
            tracer, V_, S_, U_ = MagicProject.__dataPreperation(i)
            V_i.append(V_); S_i.append(S_); U_i.append(U_)
            c = np.cov(tracer)
            c += 1e-8*np.eye(c.shape[0])
            cov.append(c)

        """ Choosing Algorithm """
        output = mp.Queue()

        if algorithm==1:
            processes = [mp.Process(target=SensorPlacement.naiveSensorPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], placed[x], x, output)) for x in range(len(cov))]
        elif algorithm==2:
            processes = [mp.Process(target=SensorPlacement.lazySensorPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], placed[x], x, output)) for x in range(len(cov))]
        elif algorithm==3:
            processes = [mp.Process(target=SensorPlacement.localKernelPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], placed[x], x, output)) for x in range(len(cov))]
        else:
            processes = [mp.Process(target=SensorPlacement.lazyLocalKernelPlacement,
                                    args=(cov[x], k, V_i[x], S_i[x], U_i[x], placed[x], x, output)) for x in range(len(cov))]

        """ Algorithm starts executing for all subdomains in parallel """
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [output.get() for p in processes]
