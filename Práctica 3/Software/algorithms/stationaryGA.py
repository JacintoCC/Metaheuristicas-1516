import numpy as np
from algorithms.geneticAlgorithm import *
from knnGPU.knnLooGPU import *
from BasicFunctions import *

def selectionOp_Stationary(population, pop_scores):
    selected_chrom = np.random.randint(len(pop_scores), size = 4)
    p_1 = tournament(selected_chrom[:2], pop_scores)
    p_2 = tournament(selected_chrom[2:], pop_scores)

    return np.array([population[p_1], population[p_2])

def replaceOp_Stationary(population, pop_scores, descendants, desc_scores):
    num_descendants = len(descendants)
    pop = np.concatenate(population[-num_descendants:],descendants)
    scores = np.concatenate(pop_scores[-num_descendants:],desc_scores)

    sort_population(pop, scores)
    population[-num_descendants:] = pop[:num_descendants]
    pop_scores[-num_descendants:] = scores[:num_descendants]

def stationaryGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Stationary, twoPointsCrossOperator,
                            mutate, replaceOp_Stationary)
