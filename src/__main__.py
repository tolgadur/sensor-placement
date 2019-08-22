#!/usr/bin/python
import numpy as np
import pandas as pd
from sensor_placement import SensorPlacement
from magic_project import MagicProject
from time import time

""" FILE NAME: '__main__.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
"""

def main():
    """ Defining file paths and calling functions to optimize sensor placement """
    MagicProject.plotHistogram(subdomain=8)
    exit(0)
    # MagicProject.describeData(subdomain=15, tracer='tracer_blackfriars')
    # MagicProject.plotResiuals(subdomain=15, tracer='tracer_blackfriars')

    # t0 = time()
    # A = MagicProject.parallelPlacement(subdomains=[1, 2], k=4, algorithm=4)
    # t1 = time()
    # print('The parallel placement algorithm takes ', (t1-t0), 'seconds')

    t0 = time()
    A = MagicProject.simplePlacement(subdomain=6, k=4, algorithm=2)
    t1 = time()
    print('The non-parallel placement algorithm takes ', (t1-t0), 'seconds')

    """ Printing the optimizing sensor placements """
    V1_df = pd.read_csv('data/csv_data/subdomain_19/positions.csv')
    V1 = V1_df[['X', 'Y', 'Z']].copy().values
    # V2_df = pd.read_csv('data/csv_data/subdomain_8/positions.csv')
    # V2 = V2_df[['X', 'Y', 'Z']].copy().values

    print('-------------------------')
    # print('subdomain 6: \n', V1[A[0][1]])
    # print('subdomain 8: \n', V2[A[1][1]])
    print(V1[A])

if __name__== "__main__":
  main()
