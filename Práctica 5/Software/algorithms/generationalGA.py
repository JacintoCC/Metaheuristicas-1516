import numpy as np
from algorithms.geneticAlgorithm import *
# from knnGPU.knnLooGPU import *
from BasicFunctions import *

# Operador de selección para esquema generacional
def selectionOp_Generational(population):
    num_parents = len(population)
    sel_items = np.random.randint(num_parents, size = 2*num_parents)
    tournament_parents = population[sel_items]

    parents = [tournament(population[sel_items[(2*i):(2*(i+1))]])
               for i in range(num_parents)]

    return np.array(parents)

# Método para obtener el operador de cruce en el esquema generacional
# a partir del algoritmo de cruce dado.
def crossOp_Generational(algorithm):
    def crossOp(parents):
        num_features = len(parents[0]['chromosome'])
        num_parents = len(parents)
        genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]
        descendants = np.zeros(num_parents, dtype=genes_type)

        # Se cruzan el 70% de los padres
        for i in range(round(0.35*num_parents)):
            parent1 = parents[2*i]
            parent2 = parents[2*i+1]

            descendants[2*i], descendants[2*i+1] = algorithm(parent1, parent2)

        # El 30% restante pasa a la siguiente generación
        for i in range(round(0.35*num_parents), int(num_parents//2)):
            descendants[2*i] = parents[2*i]
            descendants[2*i+1] = parents[2*i+1]

        return descendants

    return crossOp

# Operador de reemplazamiento para el esquema generacional
def replaceOp_Generational(population, descendants):
    num_descendants = len(descendants)
    best_ancestor = population[-1]

    if best_ancestor['score'] > max(descendants['score']):
        descendants[descendants['score'].argmin()] = best_ancestor

    return descendants

# Algoritmo genético generacional con cruce por dos puntos
def generationalGA(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(twoPointsCrossOperator),
                            mutate, replaceOp_Generational)

# Algoritmo genético generacional con cruce uniforme
def generationalGA_hux(train_data, train_categ, scorer):
    return geneticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational)
