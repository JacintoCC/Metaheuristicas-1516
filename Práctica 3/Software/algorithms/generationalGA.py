import numpy as np
from algorithms.geneticAlgorithm import *
from knnGPU.knnLooGPU import *
from BasicFunctions import *

def selectionOp_Generational(population, pop_scores):
    num_parents = len(pop_scores)
    selected_chrom = np.random.randint(num_parents, size = 2*num_parents)
    parents = np.zeros((num_parents,len(population[0])),np.bool)

    for i in range(num_parents):
        p = tournament(selected_chrom[(2*i):(2*(i+1))], pop_scores)
        parents[i] = population[p]

    return parents

def replaceOp_Generational(population, pop_scores, descendants, desc_scores):
    num_descendants = len(descendants)
    best_ancestor = population[0]

    sort_population(descendants, desc_scores)

    if best_ancestor not in descendants:
        num_descendants[-1] = best_ancestor
        desc_scores[-1] = pop_scores[0]

    population = descendants
    pop_scores = desc_scores
    

def stationaryGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational, twoPointsCrossOperator,
                            mutate, replaceOp_Generational)
