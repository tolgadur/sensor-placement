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
        """ This function describes the data in the specified subdomain according to
            the pandas describe function.
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
        data = residuals[500].values

        plt.figure(figsize=(10, 4))
        counts, bins = np.histogram(data, bins=number_bins)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()
        return None

    def plotHistogram(subdomain, number_bins=600):
        """ This function prints calculates and displays an histogram based on the
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
        """ This helper function separates all positions into placeable and
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
        """ This function loads and concatenates the csv-files of all subdomains into
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
        """ This helper function prepares the data in the format that is required for
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

        """ Preparing index arrays """
        V = V_df[['X', 'Y', 'Z']].copy().values
        V = V[:100:2]
        V_i, S_i, U_i = MagicProject.__positionIndices(V)

        """ Preparing tracer matrix """
        tracer = tracer_df.values
        tracer = tracer[:100:2]

        return tracer, V_i, S_i, U_i

    @staticmethod
    def simplePlacement(subdomain, k=4, algorithm=None, placed=np.array([], dtype=int)):
        """ This function computes the optimal sensor placement on one subdomain.
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
        """ This function is used to compute the sensor placement on multiple subdomains
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
            cov.append(np.cov(tracer))

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
