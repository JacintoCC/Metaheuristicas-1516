import numpy as np
import time
from BasicFunctions import *
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

# Algoritmo greedy SFS que devuelve una selección de características
def greedySFS(train_data, train_categ):
    num_features = len(train_data[0])

    # Partimos de un vector que no selecciona ninguna característica
    solution = np.zeros(num_features, bool)
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        # Tomamos las características no seleccionadas
        features = np.array(range(num_features))
        features = features[solution == False]
        profit_v = np.zeros(num_features, int)

        for feat in features:
            # Activamos cada una de estas características
            current_sol = np.array(solution)
            current_sol[feat] = True

            nbrs =  neighbors.KNeighborsClassifier(3)

            # Para cada dato, hacemos kNN con las características activas y el conjunto de entrenamiento
            for i in range(len(train_data)):
                train_feats = np.delete(train_data[:,current_sol],i,0)
                item = np.array([train_data[i,current_sol]], float)
                
                nbrs.fit(train_feats,np.delete(train_categ,i,0))
                profit_v[feat] += nbrs.score(item, [train_categ[i]])

        #Comparamos el número de aciertos actual y el máximo hasta ahora.
        current_profit = profit_v.max()
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[profit_v.argmax()] = True
            previous_profit = current_profit

    return solution


def runSFS(data, categories, iterations = 5, num_partitions = 2):
    results_table = np.empty([iterations*num_partitions,3], dtype=float)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    for i in range(iterations):
        print("Iteration ", i)
        partition  = makePartitions(data, categories)
        for j in range(num_partitions):
            print("Sub iteration ", j)
            start = time.time()

            training_data = partition[0][j]
            training_categ = partition[1][j]

            test_data = np.array([partition[0][k][l] for k in range(num_partitions) if k!=j for l in range(len(partition[0][k]))], float)
            test_categ = np.array([partition[1][k][l] for k in range(num_partitions) if k!=j for l in range(len(partition[1][k]))])

            solution = greedySFS(training_data, training_categ)

            nbrs =  neighbors.KNeighborsClassifier(3)
            nbrs.fit(training_data[:,solution],training_categ)
            rate = 100*nbrs.score(test_data[:,solution], test_categ)

            end = time.time()
            results_table[i*num_partitions+j,0] = rate
            results_table[i*num_partitions+j,1] = (1 - sum(solution)/len(training_data[0]))*100
            results_table[i*num_partitions+j,2] = end-start

            print("Rate = " + str(rate) + "\nTime = " + str(end-start) + " s")

    return results_table
