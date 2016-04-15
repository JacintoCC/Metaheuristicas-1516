import numpy as np
from algorithms.localSearch import *
from knnGPU.knnLooGPU import *
from BasicFunctions import *

def mutateSolution(solution, mutant_factor=0.1):
    num_features = len(solution)
    mut = np.copy(solution)
    num_features_to_mutate = int(round(mutant_factor*num_features))
    change_index = np.random.randint(num_features,size=num_features_to_mutate)

    mut[change_index] = np.logical_not(mut[change_index])
    return mut

def iteratedLocalSearch(train_data, train_categ, scorer):
    num_features = len(train_data[0])
    solution = np.random.random(size = num_features) < 0.5
    best_solution = np.copy(solution)
    best_value = 0

    for i in range(25):
        solution, value_solution = localSearch(train_data, train_categ,
                                               scorer, solution)

        if value_solution > best_value or (value_solution == best_value and
                                           sum(solution)<sum(best_solution)):
           best_value = value_solution
           best_solution = np.copy(solution)

        solution = mutateSolution(best_solution)

    return [best_solution, best_value]
