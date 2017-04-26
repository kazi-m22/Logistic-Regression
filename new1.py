from pandas import read_csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas import set_option

print(os.getcwd())
data = read_csv('databae.csv')

# ar = np.array([2231,43])
# ar = np.array(data).reshape(2231,43)
# print(ar[1,:])
# set_option('display.width', 100)
# set_option('precision', 3)
# correlations = data.corr(method = 'pearson')
# print(correlations)



