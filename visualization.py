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


# # Test dataset
# average_not_stand = pd.read_csv('data/csv_data/not_standardized/average_time.csv')
# time_500_not_stand = pd.read_csv('data/csv_data/not_standardized/timestep_500.csv')
# time_500_stand = pd.read_csv('data/csv_data/standardized/timestep_500.csv')

# # tracer_george 500 (outliers not dropped - all regions)
# tracer_george_500_0 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_0.csv')
# tracer_george_500_1 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_1.csv')
# tracer_george_500_2 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_2.csv')
# tracer_george_500_3= pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_3.csv')
# tracer_george_500_4 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_4.csv')
# tracer_george_500_5 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_5.csv')
# tracer_george_500_6 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_6.csv')
# tracer_george_500_7 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_7.csv')
# tracer_george_500_8 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_8.csv')
# tracer_george_500_9 = pd.read_csv('data/csv_data/not_standardized/tracer_george/outliers_not_dropped/LSBU_500_9.csv')

# # tracer_george (outliers dropped - region 3 and 7)
# tracer_george_500_3 = pd.read_csv('data/csv_data/not_standardized/tracer_george/LSBU_500_3.csv')
# tracer_george_500_7 = pd.read_csv('data/csv_data/not_standardized/tracer_george/LSBU_500_7.csv')

# Normalization vs standardization on the example of LSBU_500_3
# LSBU_500_3_s = pd.read_csv('data/csv_data/standardized/tracer_george/LSBU_500_3.csv')
# LSBU_500_3_n = pd.read_csv('data/csv_data/normalized/tracer_george/LSBU_500_3.csv')


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

# # Normalization vs standardization on the example of LSBU_500_3
# showHistogram(LSBU_500_3_s['tracer'], 'LSBU_500_3: TracerGeorge-Histogram (Standardized)')
# showHistogram(LSBU_500_3_n['tracer'], 'LSBU_500_3: TracerGeorge-Histogram (Normalized)')

# # TracerGeorge (3 & 7)
# showHistogram(tracer_george_500_3['tracer'], 'LSBU_500_3: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_7['tracer'], 'LSBU_500_7: TracerGeorge-Histogram (Not Standardized)')

# # TracerGeorge (all regions)
# showHistogram(tracer_george_500_0['tracer'], 'LSBU_500_0: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_1['tracer'], 'LSBU_500_1: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_2['tracer'], 'LSBU_500_2: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_3['tracer'], 'LSBU_500_3: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_4['tracer'], 'LSBU_500_4: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_5['tracer'], 'LSBU_500_5: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_6['tracer'], 'LSBU_500_6: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_7['tracer'], 'LSBU_500_7: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_8['tracer'], 'LSBU_500_8: TracerGeorge-Histogram (Not Standardized)')
# showHistogram(tracer_george_500_9['tracer'], 'LSBU_500_9: TracerGeorge-Histogram (Not Standardized)')

# # Tracer (test dataset)
# showHistogram(average_not_stand['tracer'], 'Avg. Time: Tracer-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['tracer'], 'Timestep 500: Tracer-Histogram (Not Standardized)')
# showHistogram(time_500_stand['tracer'], 'Timestep 500: Tracer-Histogram (Standardized)')
#
# # Tracer Background (test dataset)
# showHistogram(average_not_stand['tracer_background'], 'Avg. Time: TracerBackground-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['tracer_background'], 'Timestep 500: TracerBackground-Histogram (Not Standardized)')
# showHistogram(time_500_stand['tracer_background'], 'Timestep 500: TracerBackground-Histogram (Standardized)')
#
# # Pressure (test dataset)
# showHistogram(average_not_stand['pressure'], 'Avg. Time: Pressure-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['pressure'], 'Timestep 500: Pressure-Histogram (Not Standardized)')
# showHistogram(time_500_stand['pressure'], 'Timestep 500: Pressure-Histogram (Standardized)')
#
# # Velocity Norm (test dataset)
# showHistogram(average_not_stand['velocity_norm'], 'Avg. Time: VelocityNorm-Histogram (Not Standardized)')
# showHistogram(time_500_not_stand['velocity_norm'], 'Timestep 500: VelocityNorm-Histogram (Not Standardized)')
# showHistogram(time_500_stand['velocity_norm'], 'Timestep 500: VelocityNorm-Histogram (Standardized)')
