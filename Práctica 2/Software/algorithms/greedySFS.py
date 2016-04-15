import numpy as np
from knnGPU.knnLooGPU import *

# Selección de una característica aleatoria con un umbral
def getFeatureThreshold(train_data, train_categ, scorer, solution, tolerance):
    num_features = len(train_data[0])

    # Tomamos las características no seleccionadas
    features = np.arange(num_features)
    features = features[solution == False]
    profit_v = np.zeros(num_features, int)

    for feat in features:
        # Activamos  y proyectamos por cada una de estas características
        solution[feat] = True

        profit_v[feat] = scorer.scoreSolution(train_data[:,solution], train_categ)

        # Desactivamos la característica activada
        solution[feat] = False

    # Filtramos aquellas que tengan un valor menor que el umbral
    mu = profit_v.max() - tolerance*(profit_v.max()-profit_v.min())
    lrc = [[features[i],profit_v[i]] for i in range(num_features) if profit_v[i]>=mu]
    selected = np.random.randint(len(lrc))
    return lrc[selected][0],lrc[selected][1]



# Algoritmo greedy SFS que devuelve una selección de características
def greedySFS(train_data, train_categ, scorer):
    num_features = len(train_data[0])

    # tolerance para la selección de características
    tolerance = 0.3

    # Partimos de un vector que no selecciona ninguna característica
    solution = np.zeros(num_features, bool)
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        #Comparamos el número de aciertos actual y el máximo hasta ahora.
        max_position, current_profit = getFeatureThreshold(train_data,
                                       train_categ, scorer, solution,tolerance)
        exists_profit = current_profit > previous_profit

        if(exists_profit):
            solution[max_position] = True
            previous_profit = current_profit

    return solution, previous_profit
