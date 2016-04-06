import numpy as np
from numpy.random import random
from BasicFunctions import getRateL1O, flip

# Función para reinicializar solución según la frecuencia
def restartSolutionFreq(frequencies):
    num_solutions = sum(frequencies)
    return np.array([random()< 1-freq_i/num_solutions for freq_i in frequencies],bool)

# Función para modificar el tamaño de la lista tabú
def changeTabuList(tabu_list):
    if random() < 0.5:
        tabu_list = [-1 for i in range(int(np.ceil(len(tabu_list)/2)))]
    else:
        tabu_list = [-1 for i in range(3*len(tabu_list)//2)]

    return tabu_list

# Función para reinicializar solución:
def restartSolution(train_data, train_categ, best_solution, cost_best_sol, frequencies):
    restart_type = random()
    if restart_type < 0.25:
        solution = np.random.random(size=len(train_data[0]))<0.5
        cost_solution = getRateL1O(train_data[:,solution],train_categ)
    elif restart_type < 0.5:
        solution = np.copy(best_solution)
        cost_solution = cost_best_sol
    else:
        solution = restartSolutionFreq(frequencies)
        cost_solution = getRateL1O(train_data[:,solution],train_categ)

    return [solution, cost_solution]


# Búsqueda Tabú
def extendedTabuSearch(train_data, train_categ):

    num_features = len(train_data[0])
    max_neighbours = 30

    # Partimos de un vector solución aleatorio
    solution = np.random.random(size=num_features)<0.5
    cost_current_sol = getRateL1O(train_data[:,solution],train_categ)

    # Mejor solución encontrada
    best_solution = np.copy(solution)
    cost_best_sol = cost_current_sol

    # Determinamos el número de iteraciones
    max_iter = 5000//max_neighbours

    # Lista tabú
    tabu_list = [-1 for i in range(num_features//3)]
    tabu_index = 0
    choosen = False

    # Memoria a largo plazo
    long_term = np.zeros(num_features, int)

    # Número máximo de iteraciones sin mejorar la mejor solución
    max_iter_wo_improvement = 10
    last_improvement = 0

    for i in range(max_iter):
        neighbors = np.random.random_integers(0, num_features-1, max_neighbours)
        rates = np.array([[j,0] for j in neighbors], object)
        for j in range(max_neighbours):
            flip(solution, neighbors[j])
            rates[j, 1] = getRateL1O(train_data[:,solution],train_categ)
            flip(solution, neighbors[j])

        rates = rates[rates[:,1].argsort()][::-1]

        for j in range(max_neighbours):
            feature_selected = rates[j,0]

            if rates[j,1] > cost_best_sol:
                cost_best_sol =  rates[j,1]
                flip(solution, feature_selected)
                best_solution = np.copy(solution)
                last_improvement = i
                choosen = True
            elif i - last_improvement > max_iter_wo_improvement:
                tabu_index = 0
                tabu_list = changeTabuList(tabu_list)
                solution, cost_current_sol = restartSolution(train_data, train_categ, best_solution, cost_best_sol, long_term)
                break
            elif feature_selected not in tabu_list:
                flip(solution, feature_selected)
                choosen = True
            else:
                choosen = False

            if(choosen):
                tabu_list[tabu_index] = feature_selected
                tabu_index = (tabu_index+1)%len(tabu_list)
                long_term[feature_selected] +=1
                break


    return [best_solution, cost_best_sol]
