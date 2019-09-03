# Parallel Gaussian Processes for Optimal Sensor Placement
Master's thesis of Tolga H. Dur that was carried out in the summer of 2019. Academic work cited by this thesis can be found in the documentation branch. 

## Abstract
Sensor placement within the MAGIC project is currently done in a seemingly arbitrary manner. The ESPRC-funded project attempts to develop a computational system that can be used to replace heating, ventilation and air conditioning systems with environmentally friendly alternatives such as natural ventilation systems. In order to work optimally, this system relies on input data that is measured by specifically designed sensors. However, thus far, there is no methodology to optimally place sensors. This thesis aims to use large scale simulation data to place sensors such pollution across the space is best possibly observed. Exemplary, the sensor placement task is solved in the LSBU test-side. In this context, LSBU is referring to London South Bank University, which was used as an initial test-side of the MAGIC project. 

Our solution is based on the Gaussian process (GP) based sensor placement algorithm developed by Krause, Singh and Guestrin [1](http://www.cs.cmu.edu/~guestrin/Publications/JMLR08SensorPlace/jmlr08-sensor-place.pdf). Due to the practical limitation of Gaussian processes, our solution only executes on relevant sub-domains of the LSBU test-side that were obtained by Fluidity's native domain decomposition. In them, it uses various parallel execution methodologies to fully utilise all available resources. Validation is done by comparing the real pollution levels to the ones predicted by a GP fitted with our sensor placements. We find that our predictions nearly perfect, especially compared to random placement. Furthermore, we calculate performance of our sensor placements for the data assimilation with TSVD application. Here, again, we find that our sensor placements perform much better than random placements.

## User Guide
For sensor placement outside the MAGIC project, the API of the SensorPlacement class can simply be called with the necessary parameters, such as the prior covariance matrix and the number sensors to be placed. For sensor placement within the MAGIC project, however, an API was written to further ease this process. Furthermore, with a few alternations in the MagicProject class, this can also be used for sensor placement outside the MAGIC project. The process of using this API is outlined in the following.  

The simulation data has to be converted into CSV-files and placed correctly in the file structure that is described in chapter 5. For VTU-files, the conversion is taken care of by the *data_preparation.py* script, which merely needs to be called with the file-path. Optionally, this script also normalizes and standardizes observations. After conversion, the CSV-Files have to be placed into the relevant folder specifying the sub-domain that is of interest. For work outside the LSBU32 test-side, this can be placed into the folder called *subdomain_None*. 

After these preparations, the MagicProject API can be called for description, validation and sensor placements. For sensor placement, for example, the API only needs a few parameters specifying, for example, the sub-domains of interest. If the placement is outside the LSBU32 test-side, this can be substituted with a *None*. Furthermore, there is the option of specifying already placed sensors with an array of indices. An exemplary usage of the API can be seen in the following: 

``` python
""" Description """
MagicProject.plotHistogram(subdomain=8, number_bins=5)
MagicProject.describeData(subdomain=8)
MagicProject.plotResiuals(subdomain=8, number_bins=800)

""" Validation """
print(MagicProject.validation_random(subdomain=6, k=1))

indLoc = np.loadtxt('./solutions/subdomain_8/validation_format/num_sens/7sens.txt', dtype=int)
print(MagicProject.validation(subdomain=8, A=indLoc))

""" Calling functions to optimize sensor placement """
t0 = time()
A = MagicProject.parallelPlacement(subdomains=[1, 2], k=4, algorithm=4)
t1 = time()
print('The parallel placement algorithm takes ', (t1-t0), 'seconds')

t0 = time()
A = MagicProject.simplePlacement(subdomain=6, k=4, algorithm=3)
t1 = time()
print('The non-parallel placement algorithm takes ', (t1-t0), 'seconds')
```

## Data
Simulation data that models air pollution around the London South Bank University. 

## Dependencies
NumPy, Matplotlib, Multiprocessing, Vtk, Vtktools, GP, Pandas, Heapq, Time
