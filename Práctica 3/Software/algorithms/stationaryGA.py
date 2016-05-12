import numpy as np
from algorithms.geneticAlgorithm import *
# from knnGPU.knnLooGPU import *
from BasicFunctions import *

# Operador de selección para esquema estacionario
def selectionOp_Stationary(population):
    sel_items = np.random.randint(len(population), size = 4)
    p_1 = tournament(population[sel_items[:2]])
    p_2 = tournament(population[sel_items[2:]])

    return np.array([p_1, p_2])

# Método para obtener el operador de cruce en el esquema estacionario
# a partir del algoritmo de cruce dado.
def crossOp_Stationary(algorithm):
    def crossOp(parents):
        num_features = len(parents[0]['chromosome'])
        num_parents = len(parents)
        genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]
        descendants = np.zeros(num_parents, dtype=genes_type)

        # Se seleccionan todos los padres por parejas para cruzarlos
        for i in range(int(num_parents//2)):
            parent1 = parents[2*i]
            parent2 = parents[2*i+1]

            descendants[2*i], descendants[2*i+1] = algorithm(parent1, parent2)

        return descendants

    return crossOp

# Operador de reemplazamiento para el esquema estacionario
def replaceOp_Stationary(population, descendants):
    num_descendants = len(descendants)
    replacement = np.concatenate((population[:num_descendants],descendants))

    replacement.sort(order = 'score')
    population[:num_descendants] = replacement[-num_descendants:]

    return population

# Algoritmo genético estacionario con cruce por dos puntos
def stationaryGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Stationary,
                            crossOp_Stationary(twoPointsCrossOperator),
                            mutate, replaceOp_Stationary)

# Algoritmo genético estacionario con cruce uniforme
def stationaryGA_hux(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Stationary,
                            crossOp_Stationary(huxCrossOperator),
                            mutate, replaceOp_Stationary)
