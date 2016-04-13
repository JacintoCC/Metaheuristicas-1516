import numpy as np
from localSearch import *
from BasicFunctions import *

def basicMultibootSearch(train_data, train_categ):
    num_features = len(train_data[0])
    best_solution = np.zeros(num_features)
    best_value = 0

    for i in range(25):
        # Partimos de un vector solución aleatorio
        solution = np.random.random(size=num_features)<0.5

        solution, value_solution = localSearch(train_data,train_categ,solution)

        if value_solution > best_value or (value_solution == best_value and sum(solution)<sum(best_solution)):
           best_value = value_solution
           best_solution = np.copy(solution)

    return [best_solution, best_value]
