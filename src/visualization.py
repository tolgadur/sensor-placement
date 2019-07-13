#!/usr/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import GPy

""" FILE NAME: 'visualization.py'
    DESCRIPTION: This file is providing visualizations for the data used in this
    project.

"""

def printHistogramInformation(title, bins, counts):
    """ This function prints the title, bins and counts of the histogram that is displayed.
        Input:
        - title: specifies the title of the histogram.
        - bins: specifies the bins.
        - counts: specifies how many observations are in each bin.
    """
    print(title, ':\n')
    print('Bins:\n', bins, '\n')
    print('counts:\n', counts, '\n')

def showHistogram(data, title, number_bins=400):
    """ This function prints calculates and displays an histogram based on the
        inputted data.
        Input:
        - data: data on which the histogram is calculated on.
        - title: specifies the title of the histogram.
        - number_bins: specifies how many bins the histogram should have
    """
    plt.figure(figsize=(10, 4))
    counts, bins = np.histogram(data, bins=number_bins)
    plt.title(title + '. Number of bins: ' + str(number_bins))
    plt.hist(bins[:-1], bins, weights=counts)
    printHistogramInformation(title, bins, counts)
    plt.show()

df16 = pd.read_csv('data/csv_data/normalized/LSBU_32/LSBU_500_16.csv')
df17 = pd.read_csv('data/csv_data/normalized/LSBU_32/LSBU_500_17.csv')
showHistogram(df16, 'LSBU_500_16')
showHistogram(df17, 'LSBU_500_17')
