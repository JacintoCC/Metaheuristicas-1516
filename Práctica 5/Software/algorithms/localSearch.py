import numpy as np
from BasicFunctions import *

# Algoritmo de búsqueda local que devuelve una selección de características
def localSearch(train_data, train_categ, score, solution):
    # Obtención del número de características
    num_features = len(train_data[0])

    # Partimos de la medida actual de acierto
    exists_profit = True
    last_score = score(train_data[:,solution],train_categ)

    # Número de comprobaciones
    max_checks = 15000
    num_checks = 0

    while exists_profit and num_checks < max_checks:
        solution, current_score, inner_checks = localSearch1iteration(train_data, train_categ,
                                                                      score, solution)
        exists_profit = current_score > last_score
        num_checks += inner_checks

        if exists_profit:
            last_score = current_score

    return solution, previous_profit, num_checks


# Iteración de búsqueda local
def localSearch1iteration(train_data, train_categ, score, solution):
    # Obtención del número de características
    num_features = len(train_data[0])

    # Partimos de la medida actual de acierto
    last_score = score(train_data[:,solution],train_categ)
    exists_profit = False

    # Determinamos la característica por la que empezamos la búsqueda
    first_neigh = np.random.randint(0,num_features-1)
    i = first_neigh
    num_checks = 0

    while not exists_profit and num_checks < num_features:
        # Cambiamos cada una de estas características y proyectamos
        flip(solution,i)

        # Medimos la tasa de acierto con este cambio
        current_score = score(train_data[:,solution], train_categ)
        num_checks += 1

        # Si mejora la solución nos quedamos con este cambio
        exists_profit = current_score > last_score

        # Descambiamos la característica cambiada
        if not exists_profit:
            flip(solution,i)
        else:
            last_score = current_score

        # Avanzamos el contador
        i = (i+1)%num_features


    return solution, last_score, num_checks
