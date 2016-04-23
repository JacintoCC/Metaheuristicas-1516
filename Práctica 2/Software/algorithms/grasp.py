import numpy as np
from algorithms.localSearch import *
from knnGPU.knnLooGPU import *
from BasicFunctions import *

# Selección de una característica aleatoria con un umbral
def getFeatureThreshold(train_data, train_categ, score, solution, tolerance):
    num_features = len(train_data[0])

    # Tomamos las características no seleccionadas
    features = np.arange(num_features)
    features = features[solution == False]
    profit_v = np.zeros(len(features), np.int32)

    for i in range(len(features)):
        # Activamos  y proyectamos por cada una de estas características
        solution[features[i]] = True

        profit_v[i] = score(train_data[:,solution], train_categ)

        # Desactivamos la característica activada
        solution[features[i]] = False

    # Filtramos aquellas que tengan un valor menor que el umbral
    mu = profit_v.max() - tolerance*(profit_v.max()-profit_v.min())
    lrc = [[features[i],profit_v[i]] for i in range(len(features)) if profit_v[i]>=mu]
    selected = np.random.randint(len(lrc))
    return lrc[selected][0],lrc[selected][1]



# Algoritmo greedy SFS que devuelve una selección de características
def greedyRandom(train_data, train_categ, score):
    num_features = len(train_data[0])

    # tolerance para la selección de características
    tolerance = 0.3

    # Partimos de un vector que no selecciona ninguna característica
    solution = np.empty(num_features, bool)
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        #Comparamos el número de aciertos actual y el obtenido
        max_position, current_profit = getFeatureThreshold(train_data,
                                       train_categ, score, solution,tolerance)
        exists_profit = current_profit > previous_profit

        if(exists_profit):
            solution[max_position] = True
            previous_profit = current_profit

    return solution, previous_profit


def grasp(train_data, train_categ, score):
    num_features = len(train_data[0])
    best_solution = np.empty(num_features, np.bool)
    best_value = 0

    for i in range(25):
        print("Vuelta " + str(i))
        # Partimos de un vector solución obtenido mediante greedy
        solution = greedyRandom(train_data, train_categ, score)[0]

        solution, value_solution = localSearch(train_data,train_categ,
                                               score, solution)

        if value_solution > best_value or (value_solution == best_value and
                                           sum(solution)<sum(best_solution)):
           best_value = value_solution
           best_solution = np.copy(solution)

    return [best_solution, best_value]
