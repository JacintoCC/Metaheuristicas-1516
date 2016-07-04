import numpy as np
from knnGPU.knnLooGPU import *

# Selección de la mejor característica
def getBetterFeature(train_data, train_categ, scorer, solution):
    num_features = len(train_data[0])

    # Tomamos las características no seleccionadas
    features = np.array(range(num_features))
    features = features[solution == False]
    profit_v = np.zeros(num_features, int)

    for feat in features:
        # Activamos  y proyectamos por cada una de estas características
        solution[feat] = True

        profit_v[feat] = scorer(train_data[:,solution], train_categ)

        # Desactivamos la característica activada
        solution[feat] = False

    return [profit_v.max(),profit_v.argmax()]

# Algoritmo greedy SFS que devuelve una selección de características
def greedySFS(train_data, train_categ, scorer):
    num_features = len(train_data[0])

    # Partimos de un vector que no selecciona ninguna característica
    solution = np.zeros(num_features, bool)
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        #Comparamos el número de aciertos actual y el máximo hasta ahora.
        current_profit, max_position = getBetterFeature(train_data, train_categ, scorer, solution)
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[max_position] = True
            previous_profit = current_profit

    return [solution, previous_profit]
