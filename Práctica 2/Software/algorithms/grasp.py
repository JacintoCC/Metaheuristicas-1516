import numpy as np
from algorithms.localSearch import *
from algorithms.greedySFS import *
from knnGPU.knnLooGPU import *
from BasicFunctions import *

def grasp(train_data, train_categ, scorer):
    num_features = len(train_data[0])
    best_solution = np.zeros(num_features)
    best_value = 0

    for i in range(25):
        print(i)
        # Partimos de un vector solución obtenido mediante greedy
        solution = greedySFS(train_data, train_categ)[0]

        solution, value_solution = localSearch(train_data,train_categ,
                                               scorer, solution)

        if value_solution > best_value or (value_solution == best_value and
                                           sum(solution)<sum(best_solution)):
           best_value = value_solution
           best_solution = np.copy(solution)

    return [best_solution, best_value]
