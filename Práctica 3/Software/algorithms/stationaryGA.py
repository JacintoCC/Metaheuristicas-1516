import numpy as np
from algorithms.geneticAlgorithm import *
# from knnGPU.knnLooGPU import *
from BasicFunctions import *

def selectionOp_Stationary(population):
    sel_items = np.random.randint(len(population), size = 4)
    p_1 = tournament(sel_items[:2])
    p_2 = tournament(sel_items[2:])

    return np.array([population[p_1], population[p_2]], dtype = genes_type)

def replaceOp_Stationary(population, descendants):
    num_descendants = len(descendants)
    replacement = np.concatenate((population[-num_descendants:],descendants))

    replacement.sort(order = 'value')
    population[:num_descendants] = replacement[-num_descendants:]


def stationaryGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Stationary, twoPointsCrossOperator,
                            mutate, replaceOp_Stationary)
