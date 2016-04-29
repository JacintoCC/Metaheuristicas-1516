import numpy as np
from algorithms.localSearch import *
from knnGPU.knnLooGPU import *
from BasicFunctions import *


def generationalGA(train_data, train_categ, scorer):
    num_features = len(train_data[0])
    solution = np.random.random(size = num_features) < 0.5
    best_solution = np.copy(solution)
    best_value = 0


    return [best_solution, best_value]
