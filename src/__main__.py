#!/usr/bin/python
import numpy as np
import pandas as pd
from sensor_placement import SensorPlacement
from magic_project import MagicProject
from time import time

""" FILE NAME: '__main__.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
    This implementation uses a sample calculated that is calculated after applying PCA
    on the timesteps. 95% of the variance is hereby preserved.
"""

def main():
    """ Defining file paths and calling functions to optimize sensor placement """
    # MagicProject.showHistogram(area=1)

    t0 = time()
    A = MagicProject.parallelPlacement(areas=[1, 2], k=4, algorithm_choice=4)
    t1 = time()
    print('The parallel placement algorithm takes ', (t1-t0), 'seconds')

    # t0 = time()
    # # A = MagicProject.simplePlacement(area=1, k=4, algorithm_choice=4)
    # t1 = time()
    # print('The non-parallel placement algorithm takes ', (t1-t0), 'seconds')

    """ Printing the optimizing sensor placements """
    V1_df = pd.read_csv('data/csv_data/area_1/positions.csv')
    V2_df = pd.read_csv('data/csv_data/area_2/positions.csv')
    V1= V1_df[['X', 'Y', 'Z']].copy().values
    V2= V2_df[['X', 'Y', 'Z']].copy().values

    print('-------------------------')
    print('Area 6: \n', V1[A[0][1]])
    print('Area 8: \n', V2[A[1][1]])
    # print(V1[A])

if __name__== "__main__":
  main()
