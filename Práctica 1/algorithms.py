from math import sqrt
import numpy as np
import time
from BasicFunctions import *

# Función para obtener la distancia entre dos vectores.
def distance(x,y):
    return sqrt(np.sum((x-y)*(x-y)))

# Dadas las posiciones donde están los vecinos más cercanos, da la categoría más votada.
def countingVotes(mins, categories):
    count_votes = dict((i, categories.tolist().count(i)) for i in [categories[j] for j in mins])
    max_nvotes = max(count_votes.values())
    most_voted = [ cat for cat in count_votes if count_votes[cat]==max_nvotes]

    for i in mins:
        if( categories[i] in most_voted):
            return categories[i]

# Busca los k vecions más cercanos y devuelve la categoría predominante.
def kNN(k, data, categories, item):
    distances =  np.array([[i, distance(data[i],item)] for i in range(len(data))], float)
    mins = []
    distances_i = np.array(distances)
    for i in range(k):
        loc_min_distance = np.array([row[1] for row in distances_i], float).argmin()
        mins.append(distances_i[loc_min_distance,0])
        distances_i = np.delete(distances_i,loc_min_distance,0)

    return countingVotes(mins, categories)

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
            # Para cada dato, hacemos kNN con las características activas y el conjunto de entrenamiento
            for i in range(len(train_data)):
                train_feats = np.delete(train_data[:,current_sol],i,0)
                item = np.array(train_data[i,current_sol], float)
                cat = kNN(3,train_feats, train_categ, item)
                correct = (cat == train_categ[i])
                if(correct):
                    profit_v[feat] += 1
        #Comparamos el número de aciertos actual y el máximo hasta ahora.
        current_profit = profit_v.max()
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[profit_v.argmax()] = True
            previous_profit = current_profit

    return [solution,current_profit]


def runSFS(data, categories, iterations = 5, num_partitions = 2):
    corrects_vector_train = []
    corrects_vector_test = []

    for i in range(iterations):
        print("Iteration ", i)
        partition  = makePartitions(data, categories)
        for j in range(num_partitions):
            print("Sub iteration ", i)
            start = time.time()

            training_data = partition[0][j]
            training_categ = partition[1][j]

            eval_data = np.array([partition[0][k][l] for k in range(num_partitions) if k!=j for l in range(len(partition[0][k]))], float)
            eval_categ = np.array([partition[1][k][l] for k in range(num_partitions) if k!=j for l in range(len(partition[1][k]))])

            solution = greedySFS(training_data, training_categ)
            corrects = 0
            train_feats = np.array([row[solution] for row in training_data], float)

            for k in range(len(eval_data)):
                item = eval_data[k,solution]
                if(kNN(3,train_feats, training_categ, item) == eval_categ[k]):
                    corrects += 1

            corrects_vector.append(corrects/len(eval_data)*100)
            end = time.time()
    print("Rate = " + str(corrects_vector) + "\nTime = " + str(end-start) + " s")

    return np.array(corrects_vector).mean()
