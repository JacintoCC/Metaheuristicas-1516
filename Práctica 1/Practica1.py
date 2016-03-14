from scipy.io import arff
from algorithms import *
import numpy as np

datalist = arff.loadarff('Datos/wdbc.arff')[0]

categories = np.array([row[0] for row in datalist])
data = np.array([[row[j] for j in range(1, len(datalist[0]))] for row in datalist])

print(runSFS(data, categories))
