#!/usr/bin/python
import numpy as np
import pandas as pd

data = pd.read_csv('data/csv_data/subdomain_6/tracer.csv')
mean_total = data.describe().values

indLoc = np.loadtxt('./solutions/LSBU32/subdomain_6_8/validation_format/output_6_4sens.txt', dtype=int) # indexes of nodes
indLoc = indLoc.tolist()
mean_loc = data.ix[indLoc].describe().values

# mean = []
# for i in range(0, 536):
#     loc = mean_loc[1][i]
#     tot = mean_total[1][i]
#     mean.append(loc/tot)
#
# print(np.mean(mean))

print(data.describe())
print(data.ix[indLoc].describe())
