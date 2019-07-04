#!/usr/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import GPy

################################################################################
#        FILE NAME: 'visualization.py'                                         #
#        DESCRIPTION: This file is providing visualizations for the data used  #
#        in this project.                                                      #
################################################################################

# average_not_stand = pd.read_csv('data/csv_data/not_standardized/average_time.csv')
# time_500_not_stand = pd.read_csv('data/csv_data/not_standardized/timestep_500.csv')
# time_500_stand = pd.read_csv('data/csv_data/standardized/timestep_500.csv')
pvtu_500_0 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_0.csv')
pvtu_500_1 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_1.csv')
pvtu_500_2 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_2.csv')
pvtu_500_3 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_3.csv')
pvtu_500_4 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_4.csv')
pvtu_500_5 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_5.csv')
pvtu_500_6 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_6.csv')
pvtu_500_7 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_7.csv')
pvtu_500_8 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_8.csv')
pvtu_500_9 = pd.read_csv('data/csv_data/pvtu_500_not_standardized/PVTU_500_9.csv')


################################# HISTOGRAMS ###################################

def printHistogramInformation(title, bins, counts):
    print(title, ':\n')
    print('Bins:\n', bins, '\n')
    print('counts:\n', counts, '\n')

def showHistogram(data, title, number_bins=400):
    plt.figure(figsize=(10, 4))
    counts, bins = np.histogram(data, bins=number_bins)
    plt.title(title + '. Number of bins: ' + str(number_bins))
    plt.hist(bins[:-1], bins, weights=counts)
    printHistogramInformation(title, bins, counts)
    plt.show()

# TracerGeorge
showHistogram(PVTU_500_0['tracer'], 'PVTU_500_0: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_1['tracer'], 'PVTU_500_1: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_2['tracer'], 'PVTU_500_2: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_3['tracer'], 'PVTU_500_3: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_4['tracer'], 'PVTU_500_4: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_5['tracer'], 'PVTU_500_5: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_6['tracer'], 'PVTU_500_6: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_7['tracer'], 'PVTU_500_7: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_8['tracer'], 'PVTU_500_8: TracerGeorge-Histogram (Not Standardized)')
showHistogram(PVTU_500_9['tracer'], 'PVTU_500_9: TracerGeorge-Histogram (Not Standardized)')

# # Tracer
# showHistogram(average_not_stand['tracer'], 'Avg. Time: Tracer-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['tracer'], 'Timestep 500: Tracer-Histogram (Not Standardized)')
# showHistogram(time_500_stand['tracer'], 'Timestep 500: Tracer-Histogram (Standardized)')
#
# # Tracer Background
# showHistogram(average_not_stand['tracer_background'], 'Avg. Time: TracerBackground-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['tracer_background'], 'Timestep 500: TracerBackground-Histogram (Not Standardized)')
# showHistogram(time_500_stand['tracer_background'], 'Timestep 500: TracerBackground-Histogram (Standardized)')
#
# # Pressure
# showHistogram(average_not_stand['pressure'], 'Avg. Time: Pressure-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['pressure'], 'Timestep 500: Pressure-Histogram (Not Standardized)')
# showHistogram(time_500_stand['pressure'], 'Timestep 500: Pressure-Histogram (Standardized)')
#
# # Velocity Norm
# showHistogram(average_not_stand['velocity_norm'], 'Avg. Time: VelocityNorm-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['velocity_norm'], 'Timestep 500: VelocityNorm-Histogram (Not Standardized)')
# showHistogram(time_500_stand['velocity_norm'], 'Timestep 500: VelocityNorm-Histogram (Standardized)')
