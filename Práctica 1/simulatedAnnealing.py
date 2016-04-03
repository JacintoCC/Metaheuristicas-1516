import numpy as np
from BasicFunctions import getRateL1O, flip

# Función para actualizar el valor de la temperatura
def updateTemp(t_k, beta):
    return t_k/(1+beta*t_k)

# Enfriamiento Simulado
def simAnnealing(train_data, train_categ):
    num_features = len(train_data[0])
    max_neighbours = 10 * num_features

    # Partimos de un vector solución aleatorio
    solution = np.random.random(size=num_features)<0.5
    cost_current_sol = getRateL1O(train_data[:,solution],train_categ)

    # Mejor solución encontrada
    best_solution = solution
    cost_best_sol = cost_current_sol

    # Establecemos la temperatura inicial y final
    temp_0 = 0.3 * cost_best_sol / -np.log(0.3)
    temp_final = 0.001 if temp_0 > 0.001 else temp_0*0.01

    # Inicializamos las variables que controlan el bucle
    temp = temp_0
    num_checks = 1
    max_checks = 15000
    num_successes = 1
    max_successes = 0.1 * max_neighbours

    beta = (temp_0 - temp_final) / (max_checks/max_neighbours)*temp_0*temp_final

    while num_successes > 0 and num_checks < max_checks and temp > temp_final:
        num_successes = 0
        i = 0
        while num_successes < max_successes and i < max_neighbours and num_checks < max_checks:
            j = np.random.randint(0, num_features)
            flip(solution, j)

            cost_last_sol =  getRateL1O(train_data[:,solution], train_categ)
            improvement = cost_last_sol - cost_current_sol

            if improvement != 0 and (improvement > 0 or np.random.random() < np.exp(improvement/temp)):
                cost_current_sol = cost_last_sol
                num_successes += 1
                if cost_last_sol > cost_best_sol:
                    best_solution = np.copy(solution)
                    cost_best_sol = cost_current_sol
            else:
                flip(solution, j)

            num_checks += 1
            i += 1

        temp = updateTemp(temp, beta)

    return [best_solution,cost_best_sol]
