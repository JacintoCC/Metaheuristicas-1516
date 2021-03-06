import numpy as np
from BasicFunctions import *

# Algoritmo de búsqueda local que devuelve una selección de características
def localSearch(train_data, train_categ):
    num_features = len(train_data[0])

    # Partimos de un vector solución aleatorio
    solution = np.random.random(size=num_features)<0.5

    # Partimos de la medida actual de acierto
    exists_profit = True
    previous_profit = getRateL1O(train_data[:,solution],train_categ)

    # Ponemos el contador de comprobaciones a 0
    num_checks = 0

    while exists_profit and num_checks<15000:
        exists_profit = False
        first_neigh = np.random.randint(0,num_features-1)
        last_neigh = (first_neigh-1)%num_features
        i = first_neigh

        while not exists_profit and i != last_neigh and num_checks<15000:
            # Cambiamos cada una de estas características y proyectamos
            flip(solution,i)
            data_w_features = train_data[:,solution]

            # Medimos la tasa de acierto con este cambio
            current_profit = getRateL1O(data_w_features,train_categ)
            num_checks += 1

            # Si mejora la solución nos quedamos con este cambio
            exists_profit = current_profit > previous_profit

            # Descambiamos la característica cambiada
            if not exists_profit:
                flip(solution,i)
            else:
                previous_profit = current_profit

            # Avanzamos el contador
            i = (i+1)%num_features


    return [solution, previous_profit]
