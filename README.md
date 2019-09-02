# Parallel Gaussian Processes for Optimal Sensor Placement


## User Guide
For sensor placement outside the MAGIC project, the API of the SensorPlacement class can simply be called with the necessary parameters, such as the prior covariance matrix and the number sensors to be placed. For sensor placement within the MAGIC project, however, an API was written to further ease this process. The process of using this API is outlined in the following. Furthermore, with a few alternations in the MagicProject class, this can also be used for sensor placement outside the MAGIC project.  

The simulation data has to be converted into CSV-files and placed correctly in the file structure that is described in chapter 5. For VTU-files, the conversion is taken care of by the \textit{data\_preparation.py} script, which merely needs to be called with the file-path. Optionally, this script also normalizes and standardizes observations. After conversion, the CSV-Files have to be placed into the relevant folder specifying the sub-domain that is of interest. For work outside the LSBU32 test-side, this can be placed into the folder called \textit{subdomain\_0}. 

After these preparations, the MagicProject API can be called for description, validation and sensor placements. For sensor placement, for example, the API only needs a few parameters specifying, for example, the sub-domains of interest. If the placement is outside the LSBU32 test-side, this can be substituted with a $0$. Furthermore, there is the option of specifying already placed sensors with an array of indices. An exemplary usage of the API can be seen in the following: 

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

## Dependencies
NumPy, Matplotlib, Multiprocessing, Vtk, Vtktools, GP, Pandas, Heapq, Time

## Contributing
We welcome pull request with improvements and suggestions.

## Licence 
[MIT](https://github.com/tolgadur/sensor-placement/blob/master/LICENSE)
