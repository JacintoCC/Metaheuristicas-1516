import numpy as np
from algorithms.antAlgorithm import *
from knnGPU.knnLooGPU import *

def updatePheromoneACS(pheromone, rho, best_solution, best_score):
    num_features = len(best_solution)

    for i in range(num_features):
        if best_solution[i]:
            pheromone['char'][i] = (1-rho) * pheromone['char'][i] + rho * best_score


def getInitialPheromoneACS(num_features, score, rho):
    tau_max = score/rho

    pheromone = [('car',    str(num_features)+'float'),
                 ('num_car',str(num_features)+'float')]

    for i in pheromone['car']:
        i= tau_max

    for i in pheromone['num_car']:
        i= 1/num_features

    return pheromone

def getAddFeatureACSMethod(q0, phi):
    def addFeatureACS(solution, pheromone, heuristic, alpha, beta, pheromone_0):
        rand = np.random.random()

        if rand < q0:
            num_features = len(solution)
            prob = np.zeros(num_features, dtype=np.float32)
            for i in range(num_features):
                if not solution[i]:
                    value = pheromone[i]**alpha * heuristic[i]**beta
                    prob[i] = value

            feature_added = np.argmax(prob)
            flip(solution, feature_added)
        else:
            feature_added = getNewFeatureGeneral(solution, pheromone, heuristic,
                                                 alpha, beta, pheromone_0)

        pheromone[feature_added] = (1-phi)*pheromone[feature_added] + phi*pheromone_0[i]
        return feature_added

    return addFeatureACS

# Algoritmo de Colonia de Hormigas
def antColony(data, categ, scorer):

    return antAlgorithm(data, categ, scorer, getInitialPheromoneACS,
                        getAddFeatureACSMethod(q0 = 0.8, phi = 0.2),
                        updatePheromoneGlobalACS)
