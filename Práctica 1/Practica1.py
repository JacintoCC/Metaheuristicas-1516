from scipy.io import arff
from algorithms import *
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
    Lectura de la semilla pasada como argumento por línea de comandos
    y base de datos a utilizar
"""
database_name = 'Datos/'
db_options = {'W': 'wdbc', 'L': 'movement_libras', 'A':'arrhythmia'}
class_row = {'W': 0, 'L': 90, 'A':278}

parser = argparse.ArgumentParser(description='')
parser.add_argument('DB', choices=['W','L','A'],
                   help='DB to use. W -> WDBC;   L -> Libras;   A -> Arrythmia')
parser.add_argument('-seed', type=int,
                   help='Seed to random generator. Default=314159', default=314159)

args = parser.parse_args()
random_ppio.Set_random(args.seed)

opt = args.DB
database_name += db_options[args.DB] + '.arff'
database = arff.loadarff(database_name)[0]
categories = np.array([row[class_row[args.DB]] for row in database])
data = np.array( [[row[j]   for j in range(len(database[0])) if j!=class_row[args.DB]]
                            for row in database])

print("Seed = " + str(random_ppio.Get_random()))

"""
    SFS
"""

print(runSFS(data, categories))
