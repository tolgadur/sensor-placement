#!/usr/bin/python
import numpy as np
import pandas as pd
import multiprocessing as mp
from matplotlib import pyplot as plt
from sensor_placement import SensorPlacement

""" FILE NAME: 'magic_project.py'
    DESCRIPTION: This file is implementing the class that will be used for sensor
    positioning
"""

class MagicProject:


    @staticmethod
    def __positionIndices(V):
        V_i = np.array(range(len(V)))
        S_i = np.argwhere(V[:,2]<=30.0).flatten()
        U_i = np.setdiff1d(V_i, S_i, assume_unique=True)
        return V_i, S_i, U_i

    @staticmethod
    def __all_areas():
        V_df = pd.read_csv('data/csv_data/area_0/positions.csv')
        tracer_df = pd.read_csv('data/csv_data/area_0/tracer.csv')
        for i in range(31):
            position_file = 'data/csv_data/area_'+str(i)+'/positions.csv'
            tracer_file = 'data/csv_data/area_'+str(i)+'/tracer.csv'

            V_df = pd.concat([V_df, pd.read_csv(position_file)])
            tracer_df = pd.concat([tracer_df, pd.read_csv(tracer_file)])

        """ Preparing index arrays """
        V = V_df[['X', 'Y', 'Z']].copy().values
        V = V[:100:2]
        V_i, S_i, U_i = MagicProject.__positionIndices(V)

        """ Preparing tracer matrix """
        tracer = tracer_df.values
        tracer = tracer[:100:2]

        return tracer, V_i, S_i, U_i

    @staticmethod
    def __dataPreperation(area):
        if area==-1:
            return MagicProject.__all_areas()

        position_file = 'data/csv_data/area_'+str(area)+'/positions.csv'
        tracer_file = 'data/csv_data/area_'+str(area)+'/tracer.csv'

        """ Preparing index arrays """
        V_df = pd.read_csv(position_file)
        V = V_df[['X', 'Y', 'Z']].copy().values
        V = V[:100:2]
        V_i, S_i, U_i = MagicProject.__positionIndices(V)

        """ Preparing tracer matrix """
        tracer_df = pd.read_csv(tracer_file)
        tracer = tracer_df.values
        tracer = tracer[:100:2]

        return tracer, V_i, S_i, U_i

    def showHistogram(area, number_bins=600):
        """ This function prints calculates and displays an histogram based on the
            inputted data.
            Input:
            - data: data on which the histogram is calculated on (1D - Array)
            - title: specifies the title of the histogram.
            - number_bins: specifies how many bins the histogram should have
        """
        df = pd.read_csv('data/csv_data/area_'+str(area)+'/tracer.csv')
        data = df.mean(axis=1)
        plt.figure(figsize=(10, 4))
        counts, bins = np.histogram(data, bins=number_bins)
        plt.title('LSBU32_'+str(area)+'. Number of bins: '+str(number_bins))
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()

        """ Printing histogram information on shell """
        print('LSBU32_'+str(area)+':\n')
        print('Bins:\n', bins, '\n')
        print('counts:\n', counts, '\n')

    @staticmethod
    def simplePlacement(area, k=4, algorithm_choice=None, already_placed=np.array([])):
        """ This function computes the optimal sensor placement on one area.
            Input:
            - position_file: Filepath to the position file (has to be a csv-file)
            - tracer_file: Filepath to the tracer file (has to be a csv-file)
            - algorithm_choice: Integer specifying which approximation algorithm the sensor
              positions are calculated with. '1' == naive, '2' == priority queue,
              'default' == local kernel.
        """
        print('Starting sensor placement...', flush=True)
        tracer, V_i, S_i, U_i = MagicProject.__data_preperation(area)
        cov = np.cov(tracer)

        """ Choosing and executing algorithm """
        if algorithm_choice==1:
            return SensorPlacement.naiveSensorPlacement(cov, k, V_i, S_i, U_i, already_placed)
        elif algorithm_choice==2:
            return SensorPlacement.lazySensorPlacement(cov, k, V_i, S_i, U_i, already_placed)
        elif algorithm_choice==3:
            return SensorPlacement.localKernelPlacement(cov, k, V_i, S_i, U_i, already_placed)
        else:
            return SensorPlacement.lazyLocalKernelPlacement(cov, k, V_i, S_i, U_i, already_placed)

    @staticmethod
    def parallelPlacement(areas, k, algorithm_choice=None, already_placed=None):
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
        already_placed = [np.array([])]*len(areas) if already_placed==None else already_placed

        V_i, S_i, U_i, cov = [], [], [], []
        for i in areas:
            tracer, V_, S_, U_ = MagicProject.__data_preperation(i)
            V_i.append(V_); S_i.append(S_); U_i.append(U_)
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

        """ Algorithm starts executing for all areas in parallel """
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [output.get() for p in processes]
