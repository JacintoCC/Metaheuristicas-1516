import numpy as np
from algorithms.antAlgorithm import *
from knnGPU.knnLooGPU import *

def updatePheromoneAMM(pheromone, best_score, rho):
    tau_max = best_score/rho
    tau_min = tau_max/500

    for i in pheromone:
        i = tau_max if i>tau_max else tau_min if i<tau_min else i

    return pheromone

def getInitialPheromoneAMM(num_features, score, rho):
    pheromone = [('car',    str(num_features)+'float'),
                 ('num_car',str(num_features)+'float')]

    for i in pheromone['car']:
        i= 10**-6

    for i in pheromone['num_car']:
        i= 1/num_features

    return pheromone

# Algoritmo de Colonia de Hormigas Max-Min
def antMaxMin(data, categories, scorer):

    return antAlgorithm(data, categories, scorer, getInitialPheromoneAMM,
                        updatePheromoneAMM)
