#!/usr/bin/python
import numpy as np
import pandas as pd
from sensor_placement import SensorPlacement

""" FILE NAME: '__main__.py'
    DESCRIPTION: This file is implementing the sensor positioning solution proposed
    by Krause, Singh and Guestrin (2008) on the MAGIC testside in Elephant and Castle.
    This implementation uses a sample calculated that is calculated after applying PCA
    on the timesteps. 95% of the variance is hereby preserved.
"""

def main():
    """ Defining file paths and calling functions to optimize sensor placement """
    position_files, tracer_files = [], []
    position_files.append('data/csv_data/area_6/positions.csv')
    position_files.append('data/csv_data/area_8/positions.csv')
    tracer_files.append('data/csv_data/area_6/positions.csv')
    tracer_files.append('data/csv_data/area_8/positions.csv')

    A = SensorPlacement.parallelPlacement(position_files, tracer_files)

    """ Prining the optimizing sensor placements """
    V_df = pd.read_csv(position_files[0])
    V_6 = V_df[['X', 'Y', 'Z']].copy().values
    print(V_6[A[0][1]])

    V_df = pd.read_csv(position_files[0])
    V_8 = V_df[['X', 'Y', 'Z']].copy().values
    print(V_6[A[1][1]])

if __name__== "__main__":
  main()
