import sys
import argparse
import time
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from BasicFunctions import *
from greedySFS import *
from localSearch import *
from simulatedAnnealing import *
from tabuSearch import *
from extendedTabuSearch import *

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
alg_options = {'K': kNNSolution, 'G': greedySFS, 'L': localSearch, 'S':simAnnealing, 'T': tabuSearch, 'E': extendedTabuSearch}
alg_names = {'K': "KNN", 'G': "SFS", 'L': "LS", 'S':"SA", 'T': "TS", 'E': "ETS"}
class_row = {'W': 0, 'L': 90, 'A':278}

parser = argparse.ArgumentParser(description='')
parser.add_argument('DB', choices=['W','L','A'],
                   help='DB to use. W -> WDBC;   L -> Libras;   A -> Arrythmia')
parser.add_argument('-a', choices=['K','G','L','S','T','E'],
                  help='Algorithm to use. K -> kNN; G -> Greedy; L -> Local Search; S -> Simulated annealing; T -> Tabu Search; E -> Extended Tabu Search', default='K')
parser.add_argument('-write', type=bool,
                   help='True to format the output and save it in a .csv file', default=False)
parser.add_argument('-seed', type=int,
                   help='Seed to random generator. Default=314159', default=314159)


args = parser.parse_args()
random_ppio.Set_random(args.seed)
np.random.seed(args.seed)

opt = args.DB
database_name += db_options[args.DB] + '.arff'
database = arff.loadarff(database_name)[0]
categories = np.array([row[class_row[args.DB]].decode("utf-8") for row in database])
data = np.array( [[row[j]   for j in range(len(database[0])) if j!=class_row[args.DB]]
                            for row in database])

print("Seed = " + str(random_ppio.Get_random()))

# Función para ejecutar y probar los algoritmos de búsqueda de soluciones
def runAlgorithm(data, categories, function, iterations = 5, num_partitions = 2):
    results_table = np.empty([iterations*num_partitions,4], dtype=float)
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

            solution, train_rate = function(training_data, training_categ)

            end = time.time()

            nbrs =  neighbors.KNeighborsClassifier(3)
            nbrs.fit(training_data[:,solution],training_categ)
            rate = 100*nbrs.score(test_data[:,solution], test_categ)

            results_table[i*num_partitions+j,0] = train_rate/len(training_data)*100
            results_table[i*num_partitions+j,1] = rate
            results_table[i*num_partitions+j,2] = (1 - sum(solution)/len(training_data[0]))*100
            results_table[i*num_partitions+j,3] = end-start

            print("Rate = " + str(rate) + "\nTime = " + str(end-start) + " s")

    return results_table

def  resultsToCSV(name_alg, name_db, results):
    f = open('Resultados/'+name_db+name_alg+'.csv','a')
    f.write("partition,in,out,red,T\n")

    for i in range(len(results)):
        row = 'Particion ' + str(i//2+1) + '-' + str(i%2+1)
        for num in results[i]:
            row += ', ' + str(num)
        f.write(row +  '\n')

    mean_results = np.mean(results, axis=0)
    f.write('Media, ' + str(mean_results[0]) + ', ' + str(mean_results[1]) + ', ' + str(mean_results[2]) + ', ' + str(mean_results[3]) + '\n')
    f.close()


results = runAlgorithm(data, categories, alg_options[args.a])

if(args.write):
    resultsToCSV(alg_names[args.a], args.DB.lower(),results)
else:
    print(results)
