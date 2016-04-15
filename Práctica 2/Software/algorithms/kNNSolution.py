import numpy as np
from knnGPU.knnLooGPU import *
from BasicFunctions import *

# Selección de todas las características para algoritmo a comparar
def kNNSolution(train_data, train_categ, scorer):
    solution = np.repeat(True,len(train_data[0]))
    rate_in = scorer.scoreSolution(train_data,train_categ)
    return [solution,rate_in]
