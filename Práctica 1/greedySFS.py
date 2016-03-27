import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation

# Selección de todas las características para algoritmo a comparar
def kNNSolution(train_data, train_categ):
    return np.repeat(1,len(train_data[0]))

# Selección de la mejor característica
def getBetterFeature(train_data, train_categ, solution):
    num_features = len(train_data[0])

    # Tomamos las características no seleccionadas
    features = np.array(range(num_features))
    features = features[solution == False]
    profit_v = np.zeros(num_features, int)

    # Creación del clasificador
    nbrs =  neighbors.KNeighborsClassifier(3)

    for feat in features:
        # Activamos cada una de estas características
        solution[feat] = True

        leave_1_out_iterators = cross_validation.LeaveOneOut(len(train_data))

        # Proyectamos por las columnas seleccionadas
        data_w_fratures = train_data[:,solution]

        # Para cada dato, hacemos kNN con las características activas y el conjunto de entrenamiento
        for train_index, test_index in leave_1_out_iterators:
            d_train = data_w_fratures[train_index]
            d_test = data_w_fratures[test_index]
            l_train = train_categ[train_index]
            l_test = train_categ[test_index]

            nbrs.fit(d_train, l_train)
            profit_v[feat] += 100*nbrs.score(d_test, l_test)

        # Desactivamos la característica activada
        solution[feat] = False

    return [profit_v.max(),profit_v.argmax()]



# Algoritmo greedy SFS que devuelve una selección de características
def greedySFS(train_data, train_categ):
    num_features = len(train_data[0])

    # Partimos de un vector que no selecciona ninguna característica
    solution = np.zeros(num_features, bool)
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        #Comparamos el número de aciertos actual y el máximo hasta ahora.
        current_profit, max_position = getBetterFeature(train_data, train_categ, solution)
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[max_position] = True
            previous_profit = current_profit

    return solution
