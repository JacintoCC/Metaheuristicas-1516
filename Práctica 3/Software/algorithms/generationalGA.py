import numpy as np
from algorithms.geneticAlgorithm import *
# from knnGPU.knnLooGPU import *
from BasicFunctions import *

def selectionOp_Generational(population):
    num_parents = len(population)
    sel_items = np.random.randint(num_parents, size = 2*num_parents)
    tournament_parents = population[sel_items]

    parents = [tournament(population[sel_items[(2*i):(2*(i+1))]])
               for i in range(num_parents)]

    return np.array(parents)

def replaceOp_Generational(population, descendants):
    num_descendants = len(descendants)
    best_ancestor = population[-1]

    descendants.sort(order = 'score')

    if best_ancestor not in descendants:
        descendants[0] = best_ancestor

    population = descendants


def generationalGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational, twoPointsCrossOperator,
                            mutate, replaceOp_Generational)
