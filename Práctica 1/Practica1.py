from scipy.io import arff
from greedySFS import *
from localSearch import *
from simulatedAnnealing import *
from tabuSearch import *
from BasicFunctions import *
import time
import numpy as np
import argparse
import sys

"""
    Importación del generador de números aleatorios en C
"""
try:
    from ctypes import *
except ImportError:
    print('ERROR! La biblioteca *ctypes* para Python no esta disponible.')
    sys.exit(-1)

random_ppio = cdll.LoadLibrary('./Random_ppio/random_ppio.so')

"""
    Lectura de la semilla pasada como argumento por línea de comandos,
    base de datos a utilizar y algoritmo a utilizar
"""
database_name = 'Datos/'
db_options = {'W': 'wdbc', 'L': 'movement_libras', 'A':'arrhythmia'}
alg_options = {'K': kNNSolution, 'G': greedySFS, 'L': localSearch, 'S':simAnnealing, 'T': tabuSearch}
class_row = {'W': 0, 'L': 90, 'A':278}

parser = argparse.ArgumentParser(description='')
parser.add_argument('DB', choices=['W','L','A'],
                   help='DB to use. W -> WDBC;   L -> Libras;   A -> Arrythmia')
parser.add_argument('-a', choices=['K','G','L','S','T','A'],
                  help='Algorithm to use. K -> kNN; G -> Greedy; L -> Local Search; S -> Simulated annealing; T -> Tabu Search', default='K')
parser.add_argument('-seed', type=int,
                   help='Seed to random generator. Default=314159', default=314159)


args = parser.parse_args()
random_ppio.Set_random(args.seed)

opt = args.DB
database_name += db_options[args.DB] + '.arff'
database = arff.loadarff(database_name)[0]
categories = np.array([row[class_row[args.DB]].decode("utf-8") for row in database])
data = np.array( [[row[j]   for j in range(len(database[0])) if j!=class_row[args.DB]]
                            for row in database])

print("Seed = " + str(random_ppio.Get_random()))

# Función para ejecutar y probar los algoritmos de búsqueda de soluciones
def runAlgorithm(data, categories, function, iterations = 5, num_partitions = 2):
    results_table = np.empty([iterations*num_partitions,3], dtype=float)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    for i in range(iterations):
        print("Iteration ", i)
        partition  = makePartitions(data, categories, random_ppio)
        for j in range(num_partitions):
            print("Sub iteration ", j)
            start = time.time()

            training_data = partition[0][j]
            training_categ = partition[1][j]

            test_data = np.array([partition[0][k][l] for k in range(num_partitions) if k!=j for l in range(len(partition[0][k]))], float)
            test_categ = np.array([partition[1][k][l] for k in range(num_partitions) if k!=j for l in range(len(partition[1][k]))])

            solution = function(training_data, training_categ)

            nbrs =  neighbors.KNeighborsClassifier(3)
            nbrs.fit(training_data[:,solution],training_categ)
            rate = 100*nbrs.score(test_data[:,solution], test_categ)

            end = time.time()
            results_table[i*num_partitions+j,0] = rate
            results_table[i*num_partitions+j,1] = (1 - sum(solution)/len(training_data[0]))*100
            results_table[i*num_partitions+j,2] = end-start

            print("Rate = " + str(rate) + "\nTime = " + str(end-start) + " s")

    return results_table

print(runAlgorithm(data, categories, alg_options[args.a]))
