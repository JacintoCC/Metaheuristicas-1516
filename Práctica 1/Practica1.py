from scipy.io import arff
from algorithms import *

datos = arff.loadarff('Datos/wdbc.arff')[0]

print(runSFS(datos))
