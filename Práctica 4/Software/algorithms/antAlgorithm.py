import numpy as np
from knnGPU.knnLooGPU import *
from BasicFunctions import *
from localSearch import *
from math import log

def getHeuristicInformation(data, categ):
    h=10
    num_features = len(data[0])
    num_data = len(data)

    set_cat = set(categ)
    index_d = {}
    for i,j in zip(set_cat,range(len(set_cat))):
        index_d[i]=j

    prob_class_fij = np.zeros((len(prob_class),num_features,h),
                              dtype=np.float32))
    prob_fij = np.zeros((num_features,h), dtype=np.float32))
    prob_class = np.zeroz(len(set_cat), dtype=np.float32)
    heuristic = np.zeros(size = num_features)

    # Formamos la matriz fij y la matriz f
    for item in range(num_data):
        for i in range(num_features):
            element = data[item,i]
            j = floor(round(element/0.1,4)) if element < 1 else 9
            prob_fij[i,j] += 1

            c = index_d[categ[item]]
            prob_class_fij[c,i,j] += 1
            prob_class[c] += 1

    # Formamos ahora el vector que nos da la heurística
    for c in prob_class:
        for i in range(num_features):
            for j in range(h):
                heuristic[i] += prob_class_fij[c,i,j]/num_data * log(prob_class_fij[c,i,j]*num_data/(prob_class[c]*prob_fij[i,j]), 2)

    return heuristic

def getNewFeatureGeneral(solution, pheromone, heuristic, alpha, beta):
    num_features = len(solution)
    prob = np.zeros(num_features, dtype=np.float32)
    sum_prob = 0

    for i in range(num_features):
        if not solution[i]:
            value = pheromone[i]**alpha * heuristic[i]**beta
            prob[i] = value
            sum_prob += value

    feature_added = roulette(prob/sum_prob)
    flip(solution, feature_added)

    return feature_added

# Estructura genérica para un algoritmo basado en hormigas.
def antColony(data, categories, scorer, getInitialPheromone, addFeature,
              updatePheromone):
    # Establecemos los parámetros del modelo
    num_ants = 10
    max_checks = 15000
    alpha = 1
    beta = 2
    rho = 0.2

    num_features = len(data[0])

    # Partimos de una solución aleatoria.
    best_solution = np.random.random(size = num_features) < 0.5
    best_score = score(data[:,best_solution], categories)

    # Calculamos la información heurística
    heuristic = getHeuristicInformation(data, categories)

    # Obtenemos la feromona inicial
    pheromone = getInitialPheromone(num_features, best_score, rho)

    #
    num_checks = 1

    while num_checks < max_checks:
        for ant in range(num_ants):
            # Partimos de una solución vacía
            solution = np.zeros(num_features, dtype = bool)

            # Seleccionamos el número de características a seleccionar
            sol_num_features = roulette(pheromone['num_car'])+1

            # Completamos hasta obtener este número de características
            current_num_features = 0
            while current_num_features < sol_num_features:
                addFeature(solution, pheromone['car'], heuristic, alpha, beta)
                current_num_features += 1

            # Realizamos la optimización local
            solution, score, new_checks = localSearchOneStep(data, categories,
                                                             scorer, solution)

            # Actualizamos la información disponible
            if best_score < score:
                best_score = score
                best_solution = np.copy( solution )

            num_checks  += new_checks

            # Actualización de la feromona
            updatePheromone(pheromone, rho, best_solution, best_score)




    return [best_solution, best_score]
