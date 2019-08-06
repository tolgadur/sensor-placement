#!/usr/bin/python
import numpy as np
import pandas as pd
from sensor_placement import SensorPlacement
from time import time

""" FILE NAME: '__main__.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
    This implementation uses a sample calculated that is calculated after applying PCA
    on the timesteps. 95% of the variance is hereby preserved.
"""

def main():
    """ Defining file paths and calling functions to optimize sensor placement """
    position_files, tracer_files = np.array([]), np.array([])
    position_files = np.append(position_files, 'data/csv_data/area_6/positions.csv')
    position_files = np.append(position_files, 'data/csv_data/area_8/positions.csv')
    tracer_files = np.append(tracer_files, 'data/csv_data/area_6/tracer.csv')
    tracer_files = np.append(tracer_files, 'data/csv_data/area_8/tracer.csv')

    # t0 = time()
    # A = SensorPlacement.parallelPlacement(position_files, tracer_files, 1)
    # t1 = time()
    # print('The parallel placement algorithm takes ', (t1-t0), 'seconds')

    t0 = time()
    A = SensorPlacement.simplePlacement('data/csv_data/area_8/positions.csv',
                                        'data/csv_data/area_8/tracer.csv', 3)
    t1 = time()
    print('The non-parallel placement algorithm takes ', (t1-t0), 'seconds')

    """ Printing the optimizing sensor placements """
    V_df = pd.read_csv(position_files[0])
    V_6 = V_df[['X', 'Y', 'Z']].copy().values
    V_df = pd.read_csv(position_files[1])
    V_8 = V_df[['X', 'Y', 'Z']].copy().values

    print('-------------------------')
    # print('Area 6: \n', V_6[A[0][1]])
    # print('Area 8: \n', V_8[A[1][1]])
    # print(V_6[A])
    print(V_8[A])

if __name__== "__main__":
  main()
