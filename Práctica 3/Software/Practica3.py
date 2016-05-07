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
from algorithms.generationalGA import generationalGA, generationalGA_hux
from algorithms.stationaryGA import stationaryGA, stationaryGA_hux

    # Importación de la clase para realizar el score de una solución
from knnGPU.knnLooGPU import knnLooGPU

    #Importación del generador de números aleatorios en C
try:
    from ctypes import *
except ImportError:
    print('ERROR! La biblioteca *ctypes* para Python no esta disponible.')
    sys.exit(-1)


def runAlgorithm(data, categories, function, random_generator,
                 alg_name, db_name, save,
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
    results_table = np.zeros((iterations*num_partitions,4), dtype=np.float32)

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
                                  if k!=j for l in range(len(partition[0][k]))], np.float32)
            training_categ = np.array([partition[1][k][l] for k in range(num_partitions)
                                          if k!=j for l in range(len(partition[1][k]))], np.int32)

            test_data = np.array(partition[0][j], np.float32)
            test_categ = np.array(partition[1][j], np.int32)

            # Objeto Scorer que usa la GPU para reducir tiempos
            scorerGPU = knnLooGPU(len(training_data), len(test_data), len(training_data[0]), 3)

            # Inicio del contador
            start = time.time()

            # Obtención de la solución
            solution, train_rate = function(training_data, training_categ,
                                            scorerGPU.scoreSolution)

            # Fin del contador
            end = time.time()

            # Evaluación en el conjunto de test.
            rate = scorerGPU.scoreOut(training_data[:,solution], test_data[:,solution],
                                      training_categ, test_categ)

            # Rellenamos la tabla con los resultados obtenidos
            row = i*num_partitions+j
            results_table[row,0] = train_rate
            results_table[row,1] = rate
            results_table[row,2] = (1 - sum(solution)/len(training_data[0]))*100
            results_table[row,3] = end-start

            # Mensaje para conocer el tiempo y acierto en ejecución
            print("BD = " + db_name  + "\nAlgorithm =" + alg_name +
                  "\nRate = " + str(rate) + "\nTime = " + str(end-start) + " s")

            if save:
                resultsToCSV(alg_name, db_name, results_table)

    print("BD = " + db_name  + "\nAlgorithm =" + alg_name)
    print(results_table)


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
    alg_options = {'K': kNNSolution, 'S': greedySFS, 'E': stationaryGA,
                   'G':generationalGA, 'EH':stationaryGA_hux, 'GH':generationalGA_hux}
    alg_names = {'K': "KNN", 'S': "SFS", 'E': "AGE", 'G':"AGG", 'EH':"AGEH", 'GH': "AGGH"}
    class_row = {'W': 0, 'L': 90, 'A':253}
    bytes_to_int = {b'B': 0, b'M': 1}

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('DB', choices=['W','L','A'],
                       help='DB to use. W -> WDBC;   L -> Libras;   A -> Arrhythmia')
    parser.add_argument('-a', choices=['K','S','E','G','EH','GH'],
                        help='Algorithm to use. K -> kNN; S -> greedy SFS; E -> AGE; G -> AGG',
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

    if( args.DB == 'W'):
        categories = np.array([bytes_to_int[row[class_row[args.DB]]] for row in database],
                              dtype=np.int32)
    else:
        categories = np.array([row[class_row[args.DB]] for row in database],
                              dtype=np.int32)


    data = np.array( [[row[j]   for j in range(len(database[0])) if j!=class_row[args.DB]]
                                for row in database], dtype=np.float32)

    print("Seed = " + str(random_ppio.Get_random()))

    # Ejecutamos el algoritmo seleccionado sobre la base de datos
    runAlgorithm(data, categories, alg_options[args.a], random_ppio,
                 alg_name = alg_names[args.a], db_name = args.DB.lower(),
                 save = args.write)

if __name__=="__main__":
    main(sys.argv)
