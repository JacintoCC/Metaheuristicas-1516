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

def crossOp_Generational(algorithm):
    def crossOp(parents):
        num_features = len(parents[0]['chromosome'])
        num_parents = len(parents)
        genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]
        descendants = np.zeros(num_parents, dtype=genes_type)

        for i in range(round(0.35*num_parents)):
            parent1 = parents[2*i]
            parent2 = parents[2*i+1]

            descendants[2*i], descendants[2*i+1] = algorithm(parent1, parent2)

        for i in range(round(0.35*num_parents), int(num_parents//2)):
            descendants[2*i] = parents[2*i]
            descendants[2*i+1] = parents[2*i+1]

        return descendants

    return crossOp

def replaceOp_Generational(population, descendants):
    num_descendants = len(descendants)
    best_ancestor = population[-1]

    if best_ancestor not in descendants:
        descendants[descendants['score'].argmin()] = best_ancestor

    return descendants


def generationalGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational)
