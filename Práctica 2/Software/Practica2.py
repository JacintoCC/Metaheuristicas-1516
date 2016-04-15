import sys
import argparse
import time
import numpy as np
import random

    # Importación de las utilidades necesarias
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from BasicFunctions import *

    # Importación de los algoritmos utilizados en la práctica
from algorithms.kNNSolution import kNNSolution
from algorithms.greedySFS import greedySFS
from algorithms.basicMultibootSearch import basicMultibootSearch
from algorithms.grasp import grasp
from algorithms.iteratedLocalSearch import iteratedLocalSearch

    # Importación de la clase para realizar el score de una solución
# from knnGPU.knnLooGPU import knnLooGPU

    #Importación del generador de números aleatorios en C
try:
    from ctypes import *
except ImportError:
    print('ERROR! La biblioteca *ctypes* para Python no esta disponible.')
    sys.exit(-1)


def runAlgorithm(data, categories, function, random_generator,
                 iterations = 5, num_partitions = 2):
    """ Método para realizar las ejecuciones de un algoritmo
        Argumentos:
            data: Conjunto de características de los datos.
            categories: Categorías correspondientes a los datos.
            function: algoritmo de obtención de la solución.
            random_generator: Generador de números aleatorios.
            iterations: Número de iteraciones total a ejecutar.
            num_partitions: Número de particiones por iteración.
        Devuelve una matriz con los datos requeridos.
    """

    # Creación de la tabla de resultados.
    results_table = np.empty([iterations*num_partitions,4], dtype=float)

    # Escalado de los datos.
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    for i in range(iterations):
        # Se realiza una partición aleatoria para cada iteración
        print("Iteration ", i)
        partition  = makePartitions(data, categories, random_generator)

        # Validación cruzada
        for j in range(num_partitions):
            # Una parte como test, el resto es entrenamiento
            print("Sub iteration ", j)

            training_data = np.array([partition[0][k][l] for k in range(num_partitions)
                                  if k!=j for l in range(len(partition[0][k]))], float)
            training_categ = np.array([partition[1][k][l] for k in range(num_partitions)
                                          if k!=j for l in range(len(partition[1][k]))])

            test_data = partition[0][j]
            test_categ = partition[1][j]

            # Objeto Scorer que usa la GPU para reducir tiempos
            #scorerGPU = knnLooGPU(len(training_data), len(training_data[0]), 3)

            # Inicio del contador
            start = time.time()

            # Obtención de la solución
            solution, train_rate = function(training_data, training_categ,
                                            scorerGPU.scoreSolution)

            # Fin del contador
            end = time.time()

            # Evaluación en el conjunto de test.
            nbrs =  neighbors.KNeighborsClassifier(3)
            nbrs.fit(training_data[:,solution],training_categ)
            rate = 100*nbrs.score(test_data[:,solution], test_categ)

            # Rellenamos la tabla con los resultados obtenidos
            row = i*num_partitions+j
            results_table[row,0] = train_rate
            results_table[row,1] = rate
            results_table[row,2] = (1 - sum(solution)/len(training_data[0]))*100
            results_table[row,3] = end-start

            # Mensaje para conocer el tiempo y acierto en ejecución
            print("Rate = " + str(rate) + "\nTime = " + str(end-start) + " s")

    return results_table


def main(args):
    """ Método principal
        Argumentos posicionales:
            DB: Base de datos a utilizar
        Argumentos optativos:
            -a: Algoritmo a utilizar. Por defecto, kNN.
            -seed: Semilla aleatoria implantada
            -write: Controla dónde se escriben los resultados
        Devuelve una matriz con los datos requeridos o los imprime en un fichero
    """

    # Se cargan los ficheros de C para el generador.
    random_ppio = cdll.LoadLibrary('./Random_ppio/random_ppio.so')

    #   Lectura de la semilla pasada como argumento por línea de comandos,
    #   base de datos a utilizar y algoritmo a utilizar
    database_name = 'Datos/'
    db_options = {'W': 'wdbc', 'L': 'movement_libras', 'A':'arrhythmia'}
    alg_options = {'K': kNNSolution, 'S': greedySFS, 'B': basicMultibootSearch,
                   'G':grasp, 'I': iteratedLocalSearch}
    alg_names = {'K': "KNN", 'S': "SFS", 'B': "BMB", 'G':"GRASP", 'I': "ILS"}
    class_row = {'W': 0, 'L': 90, 'A':278}
    bytes_to_int = {bB = 0, bM = 1}

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('DB', choices=['W','L','A'],
                       help='DB to use. W -> WDBC;   L -> Libras;   A -> Arrhythmia')
    parser.add_argument('-a', choices=['K','S','B','G','I'],
                        help='Algorithm to use. K -> kNN; S -> greedy SFS; B -> BMB; G -> GRASP; I -> ILS. Default=K',
                        default='K')
    parser.add_argument('-write', type=bool,
                        help='True to format the output and save it in a .csv file. Default=False',
                        default=False)
    parser.add_argument('-seed', type=int,
                        help='Seed to random generator. Default=314159',
                        default=314159)


    args = parser.parse_args()
    random_ppio.Set_random(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    opt = args.DB
    database_name += db_options[args.DB] + '.arff'
    database = arff.loadarff(database_name)[0]

    if( args.DB = 'W'){
        categories = np.array([bytes_to_int[row[class_row[args.DB]]] for row in database],
                              dtype=int32)
    }
    else{
        categories = np.array([row[class_row[args.DB]] for row in database],
                              dtype=int32)
    }


    data = np.array( [[row[j]   for j in range(len(database[0])) if j!=class_row[args.DB]]
                                for row in database], dtype=float32)

    print("Seed = " + str(random_ppio.Get_random()))

    # Ejecutamos el algoritmo seleccionado sobre la base de datos
    results = runAlgorithm(data, categories, alg_options[args.a], random_ppio)

    # Mostramos por pantalla o guardamos en un fichero
    if(args.write):
        resultsToCSV(alg_names[args.a], args.DB.lower(),results)
    else:
        print(results)

if __name__=="__main__":
    main(sys.argv)
