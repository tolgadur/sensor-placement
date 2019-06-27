#!/usr/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import GPy

################################################################################
#        FILE NAME: "simple_gp.py"                                             #
#        DESCRIPTION: This file is implementing the sensor positioning solu-   #
#        -tion proposed by Krause, Singh and Guestrin (2008) on the MAGIC test #
#        side in Elephant and Castle. It should be noted that this is a simple #
#        implementation that uses the RBF kernel instead of non-stationary es- #
#        -timated kernel as will be done in the final model. Furthermore, this #
#        model is not parrallelised.                                           #
#        INPUT: data1.csv, data2.csv, data3.csv                                #
################################################################################

############################# Function Definitions #############################

def plot_gp(X, m, C, training_points=None): # Copied from Jupyter Notebook by Marc Deisenroth
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    plt.fill_between(X[:],
                     m[:,0] - 1.96*np.sqrt(np.diag(C)),
                     m[:,0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    plt.legend(labels=["GP fit"])

    plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])

def subtract_pos_pd(df1, df2): # subtracts pandas dataframe according to set theory
    df_all = df1.merge(df2.drop_duplicates(), on=['X','Y', 'Z'], how='left', indicator=True)
    df_all = df_all[df_all['_merge'] == 'left_only']
    return df_all.drop(['_merge'], axis=1)

def subtract_pos_np(A, B): # subtracts multidimensional numpy array according to set theory
    cumdims = (np.maximum(A.max(),B.max())+1)**np.arange(B.shape[1])
    return A[~np.in1d(A.dot(cumdims),B.dot(cumdims))]

def print_full(x): # Prints whole pandas dataframe
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def getPossibleImpossibleA(pos): # seperates between possible and impossible sensor positions
    possible = pos.query('Z<=0.21')
    impossible = subtract_pos_pd(pos, possible)
    return possible.values, impossible.values

def approximation1(cov, num_sensors, all, possible, impossible):
    results = np.zeros((num_sensors, 3))
    for j in range(0, num_sensors):
        remaining = subtract_pos_np(possible, results)
        unselected = subtract_pos_np(all, results) # redo
        delta = np.zeros((remaining.shape[0], 3))
        for y in range(0, remaining.shape[0]):
            inv = np.linalg.inv(cov)
            var_y = cov[y][y]
            # decomp1 = cov[y][results]*inv[results][results]*cov[results][y]
            # decomp2 = cov[y][results]*inv[unselected][unselected]*cov[unselected][y]
            # delta[y] = (var_y - decomp1)/(var_y - decomp2)

        results[j] = np.amax(delta) # probably isn't this easy
    return results

# def approximation2(cov, num_sensors, possible, impossible):
#     A = pd.DataFrame(columns=['X', 'Y', 'Z'])
#     return A
#
# def approximation3(cov, num_sensors, possible, impossible):
#     A = pd.DataFrame(columns=['X', 'Y', 'Z'])
#     for y in possible:
#     return A

########################### Preparing data for GP ##############################

# loading the data into panda dataframes
# data1 = pd.read_csv("data1.csv")
# data2 = pd.read_csv("data2.csv")
data3 = pd.read_csv("data3.csv")
print(data3.sort_values('time'))
exit(0)
sample = pd.read_csv("sample.csv") # new

# data1Pos = data1[['X', 'Y', 'Z']].copy()
# data2Pos = data2[['X', 'Y', 'Z']].copy()
# data3Pos = data3[['X', 'Y', 'Z']].copy()
# pos = data1Pos.append(data2, ignore_y=True)
# pos = pos.append(data3, ignore_y=True)
all_locations = sample[['X', 'Y', 'Z']].copy() # used as input to calculate cov matrix.
all_locations = all_locations[:1000] # shuffle the dataset
possible, impossible = getPossibleImpossibleA(all_locations)
all_locations = all_locations.values

tracer = sample['tracer'].values # used as output to calculate cov matrix. (change dimension)
tracer = np.reshape(tracer, (100040, 1))
tracer = tracer[:1000]

########################### Simple GP implementation ###########################

# define covariance kernel. Mean is assumed to be 0.
k = GPy.kern.RBF(3, name="rbf")
m = GPy.models.GPRegression(all_locations, tracer, k)
m.optimize()

# calculate the posterior covariance matrix
mean, cov = m.predict(all_locations, full_cov=True)

# plotting posterior (makes no sense)
X = sample['X'].values[:1000]
plot_gp(X, mean, cov, training_points=(X, tracer))
plt.show()

# find optimal sensor A with the first approximation function
A = approximation1(cov, 9, all_locations, possible, impossible)
# print(A)
