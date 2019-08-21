import math
import pandas as pd
import numpy as np

data = pd.read_csv('./solutions/LSBU32/subdomain_6_8/output_absolute_4sensoren.csv', dtype=np.float64)
data = data.values

average_8 = 0
for i in range(3):
    average_8 += math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[i], data[i+1])]))
print(average_8/4)

average_6 = 0
for i in range(4, 7):
    average_6 += math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[i], data[i+1])]))
print(average_6/4)
